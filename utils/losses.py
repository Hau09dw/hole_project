import torch
import torch.nn as nn
import torch.nn.functional as F

def bbox_iou(box1, box2, eps=1e-7):
    """
    box1: [N,4] xyxy
    box2: [M,4] xyxy
    return IoU [N,M]
    """
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])
    area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])
    union = area1[:, None] + area2 - inter_area + eps
    return inter_area / union


class DetectionLoss(nn.Module):
    def __init__(self, lambda_box=0.05, lambda_obj=1.0, lambda_cls=0.5):
        super().__init__()
        self.bce_obj = nn.BCEWithLogitsLoss(reduction="mean")
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="mean")
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls

    def forward(self, preds, targets):
        """
        preds: list of [B,(5+num_classes),H,W]
        targets: list length B, mỗi phần tử (N,5) [cls,cx,cy,w,h] normalized
        """
        box_loss, obj_loss, cls_loss = 0.0, 0.0, 0.0

        for p in preds:  # loop qua P3,P4,P5
            B, C, H, W = p.shape
            num_classes = C - 5
            p = p.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]
            p = p.view(B, -1, C)  # [B,HW,C]

            for b in range(B):
                pred = p[b]  # [HW,C]
                t = targets[b]

                obj_tgt = torch.zeros_like(pred[:, 4:5])  # background mặc định
                cls_tgt = torch.zeros_like(pred[:, 5:])   # multi-class target

                if t.numel() > 0:
                    for gt in t:
                        cls_id, cx, cy, w, h = gt.tolist()
                        gi, gj = int(cx * W), int(cy * H)
                        idx = gj * W + gi
                        obj_tgt[idx] = 1.0

                        if num_classes > 0:
                            cls_tgt[idx, int(cls_id)] = 1.0

                        # --- Box loss ---
                        px, py, pw, ph = pred[idx, 0:4]
                        pred_box = torch.tensor([
                            px - pw/2, py - ph/2,
                            px + pw/2, py + ph/2
                        ], device=pred.device).unsqueeze(0)

                        gt_box = torch.tensor([
                            (cx - w/2) * W, (cy - h/2) * H,
                            (cx + w/2) * W, (cy + h/2) * H
                        ], device=pred.device).unsqueeze(0)

                        iou = bbox_iou(pred_box, gt_box)
                        box_loss += (1.0 - iou).mean()

                # --- Obj loss ---
                obj_loss += self.bce_obj(pred[:, 4:5], obj_tgt)

                # --- Cls loss ---
                if num_classes > 0:
                    cls_loss += self.bce_cls(pred[:, 5:], cls_tgt)

        total_loss = (self.lambda_box * box_loss +
                      self.lambda_obj * obj_loss +
                      self.lambda_cls * cls_loss)

        return total_loss


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
    
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.a, self.b, self.s = alpha, beta, smooth
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        tp = (p*targets).sum()
        fp = (p*(1-targets)).sum()
        fn = ((1-p)*targets).sum()
        tversky = (tp + self.s) / (tp + self.a*fp + self.b*fn + self.s)
        return 1 - tversky

class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets):
        # focal BCE on probabilities
        p = torch.sigmoid(logits)
        bce = -(self.alpha*targets*torch.log(p+1e-6) + (1-self.alpha)*(1-targets)*torch.log(1-p+1e-6))
        mod = ((1-p)**self.gamma)*targets + (p**self.gamma)*(1-targets)
        return (bce*mod).mean()

