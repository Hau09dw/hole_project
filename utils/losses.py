import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        # preds = list of feature maps, targets = list of bboxes
        # Simplified: chỉ tính BCE objectness
        loss = sum(self.bce(p, torch.zeros_like(p)) for p in preds)
        return loss

class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        if target.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device)
        target = target.max(dim=0, keepdim=True)[0]
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum()
        dice = 1 - (2 * intersection + 1) / (pred_sig.sum() + target.sum() + 1)
        return bce + dice

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # sigmoid nếu chưa qua activation
        inputs = torch.sigmoid(inputs)

        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice

