# trainers/mean_teacher.py
import torch
import torch.nn as nn
from models.yolov12_bifpn import YOLOv12_BiFPN
class MeanTeacherTrainer:
    def __init__(self, model, optimizer, cfg, loss_seg_fn):
        self.model = model
        self.teacher = YOLOv12_BiFPN(num_classes=cfg["data"]["num_classes"]).to(
            next(model.parameters()).device
        )
        self.teacher.load_state_dict(model.state_dict())
        self.optimizer = optimizer
        self.cfg = cfg
        self.loss_seg = loss_seg_fn
        self.ema_decay = cfg["semi"].get("ema_decay", 0.99)

    def update_teacher(self):
        for t_params, s_params in zip(self.teacher.parameters(), self.model.parameters()):
            t_params.data = self.ema_decay * t_params.data + (1 - self.ema_decay) * s_params.data

    def train_step(self, batch_l, batch_u):
        imgs_l, masks_l = batch_l["img"].to("cuda"), batch_l["masks"].to("cuda")
        imgs_u, _ = batch_u["img"].to("cuda"), batch_u["masks"].to("cuda")

        # đảm bảo mask shape [B,1,H,W]
        if masks_l.ndim == 3:
            masks_l = masks_l.unsqueeze(1)

        # supervised forward
        out_l = self.model(imgs_l)
        pred_l = out_l["seg"]

        # supervised loss (BCE + Dice)
        bce_val = nn.BCEWithLogitsLoss()(pred_l, masks_l)
        dice_val = (self.loss_seg(pred_l, masks_l) - bce_val)  # Dice thành phần
        sup_loss = bce_val + dice_val

        # consistency loss
        with torch.no_grad():
            out_u_teacher = self.teacher(imgs_u)
        out_u_student = self.model(imgs_u)

        cons_loss = torch.mean(
            (torch.sigmoid(out_u_student["seg"]) - torch.sigmoid(out_u_teacher["seg"]).detach()) ** 2
        )

        # weighting
        lambda_cons = self.cfg["semi"].get("lambda_cons", 0.1)
        total_loss = sup_loss + lambda_cons * cons_loss

        # pixel acc debug
        with torch.no_grad():
            pred_bin = (torch.sigmoid(pred_l) > 0.5).float()
            pixel_acc = (pred_bin == masks_l).float().mean().item()

        # backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.update_teacher()

        # in debug chi tiết
        print(f"[DEBUG] sup_loss={sup_loss.item():.4f}, BCE={bce_val.item():.4f}, "
              f"Dice={dice_val.item():.4f}, cons={cons_loss.item():.4f}, "
              f"total={total_loss.item():.4f}, pixel_acc={pixel_acc:.4f}")

        return (sup_loss.item(), cons_loss.item(), total_loss.item(),
        bce_val.item(), dice_val.item(), pixel_acc)
