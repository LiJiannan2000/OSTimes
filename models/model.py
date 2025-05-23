import torch
from torch import nn
import torch.nn.functional as F

from OS_times_TFM.core.config import config
from OS_times_TFM.models.resnetEncoder import ResNetEncoder
from OS_times_TFM.models.Attention import CrossTrans, SelfTrans
from OS_times_TFM.models.TimesFM import StackedDecoder


class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=512):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, embedding_dim))  # 8x

    def forward(self, x, position_ids=None):
        position_embeddings = self.position_embeddings
        return x + position_embeddings


class Model(nn.Module):
    """
    models
    """

    def __init__(self, img_dim, patch_dim, embedding_dim, num_heads, num_layers,
                 dropout_rate=0.0, attn_dropout_rate=0.0, positional_encoding_type="learned"):
        super().__init__()
        self.resnet_encoder = ResNetEncoder(config)
        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.num_patches = 2016
        self.seq_length = self.num_patches
        self.hidden_dim = self.num_patches
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.crossAttention = CrossTrans(
            self.embedding_dim,
            heads=num_heads,
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
        )
        self.selfAttention = SelfTrans(
            self.embedding_dim,
            heads=num_heads,
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
        )
        self.timesFM = StackedDecoder(
            hidden_size=config.TFM.hidden_size,
            intermediate_size=config.TFM.intermediate_size,
            num_heads=config.TFM.num_heads,
            num_kv_heads=config.TFM.num_kv_heads,
            head_dim=config.TFM.head_dim,
            num_layers=config.TFM.num_layers,
            rms_norm_eps=config.TFM.rms_norm_eps,
        )
        self.pre_head_ln = nn.LayerNorm(self.embedding_dim)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.linear1 = nn.Linear(self.embedding_dim, 256)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64 + 28, 32)
        self.linear4 = nn.Linear(32, 1)

    def forward(self, x, x_def, info, masks):
        x = torch.chunk(x, 4, dim=1)

        tensor_origin = []
        for t in range(4):
            xt = self.resnet_encoder(x[t])
            xt = xt.permute(0, 2, 3, 4, 1).contiguous()
            xt = xt.view(xt.size(0), -1, self.embedding_dim)

            x_def_resized = F.interpolate(x_def[:, t, :, :, :].unsqueeze(1), size=(12, 14, 12), mode='trilinear', align_corners=False)
            x_def_resized = x_def_resized.squeeze(1)
            x_def_resized = x_def_resized.view(x_def_resized.size(0), -1).unsqueeze(2)
            x_alpha = xt * x_def_resized
            x_alpha = self.pe_dropout(x_alpha)
            x_beta = self.pe_dropout(xt)
            xt = self.crossAttention(x_alpha, x_beta)
            tensor_origin.append(xt)
        tensor_origin = torch.stack(tensor_origin, dim=1)

        tensor_prediction = []
        fusion_fea = []
        for t in range(3):
            x_f = tensor_origin[:, t, :, :]  # 当前时间步的图像
            if t == 0:
                tensor_prediction.append(x_f)
                fusion_fea.append(x_f)

            x_f = fusion_fea[-1].detach()
            x_f = self.position_encoding(x_f)
            x_f = self.pe_dropout(x_f)
            x_f = x_f.permute(0, 2, 1).contiguous()
            x_f = self.timesFM(x_f)
            x_f = x_f.permute(0, 2, 1).contiguous()
            x_f = self.pre_head_ln(x_f)
            tensor_prediction.append(x_f)

            # 如果下一时间步图像缺失，使用预测值
            if masks[t + 1] == 1:
                x_f = tensor_prediction[-1].detach()
            else:
                x_f = tensor_origin[:, t + 1, :, :]  # 当前时间步的图像
            x_f = 0.5 * x_f + 0.5 * fusion_fea[-1].detach()
            x_f = self.pe_dropout(x_f)
            x_f = self.selfAttention(x_f)
            fusion_fea.append(x_f)

        for i in range(4):
            x_f = fusion_fea[i]
            x_f = self._reshape_output(x_f)
            x_f = self.avg_pool(x_f).flatten(start_dim=1)
            fusion_fea[i] = x_f

        risks = []
        for t in range(4):
            x_fusion = self.relu(self.linear1(fusion_fea[t]))
            x_fusion = self.relu(self.linear2(x_fusion))
            x_fusion = torch.cat([x_fusion, info], dim=1)
            x_fusion = self.relu(self.linear3(x_fusion))
            x_fusion = self.linear4(x_fusion)
            risks.append(x_fusion)

        tensor_origin = tensor_origin[:, 1:, :, :]
        tensor_prediction = torch.stack(tensor_prediction, dim=1)
        tensor_prediction = tensor_prediction[:, 1:, :, :]
        risks = torch.stack(risks, dim=1).squeeze()

        return risks, tensor_origin, tensor_prediction

    def _reshape_output(self, x):
        x = x.view(x.size(0), 12, 14, 12, self.embedding_dim)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


def OSnet(_pe_type="learned"):
    model = Model(
        img_dim=config.MODEL.INPUT_SIZE,
        patch_dim=8,
        embedding_dim=512,
        num_heads=config.MODEL.NUM_HEADS,
        num_layers=config.MODEL.NUM_LAYERS,
        dropout_rate=config.MODEL.DROPOUT_RATE,
        attn_dropout_rate=config.MODEL.ATTN_DROPOUT_RATE,
        positional_encoding_type=_pe_type,
    )
    return model


if __name__ == '__main__':
    with torch.no_grad():
        import os

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((4, 16, 182, 218, 182), device=cuda0)
        x_def = torch.rand((4, 4, 182, 218, 182), device=cuda0)
        info = torch.rand((4, 28), device=cuda0)
        masks = torch.tensor([0, 0, 1, 1], device=cuda0)
        model = Model(config.MODEL.INPUT_SIZE, 8, 512, 4, config.MODEL.NUM_LAYERS, config.MODEL.DROPOUT_RATE, config.MODEL.ATTN_DROPOUT_RATE, "fixed")
        model.cuda()
        y, tensor_ori, tensor_pre = model(x, x_def, info, masks)
        print(y.shape)
        print(tensor_ori.shape)
        print(tensor_pre.shape)
