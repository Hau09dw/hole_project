# trainers/mean_teacher.py
import torch
import torch.nn as nn
from models.yolov12_bifpn import YOLOv12_BiFPN

class MeanTeacherTrainer:
    def __init__(self, model, optimizer, cfg, loss_seg_fn, device):
        self.model = model
        self.teacher = YOLOv12_BiFPN(num_classes=cfg["data"]["num_classes"]).to(device)
        self.teacher.load_state_dict(model.state_dict())
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = device
        self.loss_seg = loss_seg_fn
        # đọc đúng nhánh config
        self.ema_decay = cfg["train"].get("ema_decay", 0.99)
        self.pseudo_thr = cfg["semi"].get("min_conf", 0.5)

    def update_teacher(self):
        for t_params, s_params in zip(self.teacher.parameters(), self.model.parameters()):
            t_params.data = self.ema_decay * t_params.data + (1 - self.ema_decay) * s_params.data

    def train_step(self, batch_l, batch_u):
        imgs_l, masks_l = batch_l["img"].to(self.device), batch_l["masks"].to(self.device)
        imgs_u = batch_u["img"].to(self.device)

        # supervised segmentation
        out_l = self.model(imgs_l)
        pred_seg_l = out_l["seg"]
        seg_loss = self.loss_seg(pred_seg_l, masks_l)

        # teacher pseudo labels (soft) + confidence mask
        with torch.no_grad():
            out_u_teacher = self.teacher(imgs_u)
            t_prob = torch.sigmoid(out_u_teacher["seg"])
            conf_thr = self.pseudo_thr
            conf_mask = ((t_prob > conf_thr) | (t_prob < (1.0 - conf_thr))).float()

        # student on unlabeled
        out_u_student = self.model(imgs_u)
        s_prob = torch.sigmoid(out_u_student["seg"])

        # soft consistency (MSE) chỉ ở vùng teacher tự tin
        consistency_loss = ((s_prob - t_prob) ** 2 * conf_mask).mean()

        # ramp-up lambda_cons (giảm nhiễu những epoch đầu)
        cur_epoch = self.cfg["train"].get("cur_epoch", 1)
        ramp = self.cfg["train"].get("cons_ramp", 5)
        base_cons = self.cfg["train"].get("lambda_cons", 0.5)
        lambda_cons = base_cons * min(1.0, cur_epoch / max(1, ramp))

        lambda_seg = self.cfg["train"].get("lambda_seg", 1.0)
        total_loss = lambda_seg * seg_loss + lambda_cons * consistency_loss

        # pixel acc (tham khảo)
        with torch.no_grad():
            pred_bin = (torch.sigmoid(pred_seg_l) > 0.5).float()
            pixel_acc = (pred_bin == masks_l).float().mean().item()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.update_teacher()

        # log BCE & DiceLoss
        with torch.no_grad():
            bce_val = torch.nn.functional.binary_cross_entropy_with_logits(pred_seg_l, masks_l).item()
            dice_val = 1.0 - (2 * (torch.sigmoid(pred_seg_l) * masks_l).sum().item() + 1) / (
                torch.sigmoid(pred_seg_l).sum().item() + masks_l.sum().item() + 1
            )

        print(f"[DEBUG] seg={seg_loss.item():.4f}, cons={consistency_loss.item():.4f}, "
              f"total={total_loss.item():.4f}, pixel_acc={pixel_acc:.4f}")

        return (seg_loss.item(), consistency_loss.item(), total_loss.item(),
                bce_val, dice_val, pixel_acc)
