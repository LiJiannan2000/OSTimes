import torch
import torch.nn as nn
from OS_times_2.models import resnet


class ResNetEncoder(nn.Module):
    """
    ResNext from pytorch official.
    """
    def __init__(self, config):
        super(ResNetEncoder, self).__init__()

        """Load pretrained resnet."""
        self.model = resnet.resnet18(
                sample_input_W=28,
                sample_input_H=28,
                sample_input_D=14,
                shortcut_type='A',
                no_cuda=False,
                num_seg_classes=2)

        if config.MODEL.USE_PRETRAINED:
            net_dict = self.model.state_dict()
            pretrained_dict = torch.load(config.MODEL.PRETRAINED)

            # Modify pretrained_dict keys
            new_state_dict = {}
            for key, value in pretrained_dict['state_dict'].items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove the 'module.' prefix
                else:
                    new_key = key
                new_state_dict[new_key] = value

            # Filter out unnecessary keys
            pretrained_dict = {k: v for k, v in new_state_dict.items() if k in net_dict}

            # Update model_dict
            net_dict.update(pretrained_dict)
            self.model.load_state_dict(net_dict)

        """Modify conv1 input channel from 1 to 4."""
        self.model.conv1 = nn.Conv3d(4, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        """Add conv2 make output smaller."""
        self.model.conv2 = nn.Conv3d(64, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x
