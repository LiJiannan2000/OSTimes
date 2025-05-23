import torch
import torch.nn as nn
import torch.nn.functional as F


class CoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad = {}

    def save_grad(self, name):
        def hook(grad):
            self.grad[name] = grad
        return hook

    def forward(self, risk, ostime, death):
        risk = risk.float()

        risk_set = torch.cumsum(torch.exp(risk).flip(dims=(0,)), dim=0).flip(dims=(0,))

        diff = (risk - torch.log(risk_set)) * death
        loss = -torch.sum(diff)
        if torch.sum(death) == 0:
            return loss
        loss = loss / torch.sum(death)

        return loss


def cosine_similarity_loss(a, b):
    assert a.shape == b.shape, "a and b must have the same shape"

    a1, a2, a3 = a[:, 0], a[:, 1], a[:, 2]
    b1, b2, b3 = b[:, 0], b[:, 1], b[:, 2]

    loss1 = 1 - F.cosine_similarity(a1, b1)
    loss2 = 1 - F.cosine_similarity(a2, b2)
    loss3 = 1 - F.cosine_similarity(a3, b3)

    total_loss = loss1.mean() + loss2.mean() + loss3.mean()
    return total_loss


def listnet_loss(y_pred, y_true):
    P_true = F.softmax(y_true, dim=1)
    P_pred = F.softmax(y_pred, dim=1)

    loss = -torch.sum(P_true * torch.log(P_pred + 1e-10), dim=1)
    return loss.mean()


def order_constraint_loss(predictions, temperature=1.0):
    batch_size = predictions.size(0)

    y_true = torch.tensor([0.0, 1.0, 2.0, 3.0], device=predictions.device)

    y_true = y_true.expand(batch_size, -1)

    y_true = y_true * temperature

    loss = listnet_loss(predictions, y_true)

    return loss
