import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.yolov12_bifpn import YOLOv12_BiFPN
from datasets.dataset import PotholeDataset
from utils.metrics import compute_iou, compute_map
from ultralytics.utils.ops import xywh2xyxy
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from prettytable import PrettyTable
@torch.no_grad()
def evaluate(cfg_path="configs/configs.yaml", weights="outputs/checkpoints/best.pt", split="val"):
    # Load config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    dataset = PotholeDataset(cfg["data"][split], img_size=cfg["data"]["img_size"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=PotholeDataset.collate_fn)

    # Model
    model = YOLOv12_BiFPN(num_classes=cfg["data"]["num_classes"]).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    # Metrics accumulators
    ious, dices, aps = [], [], []
    os.makedirs("outputs/vis_eval", exist_ok=True)

    for i, batch in enumerate(loader):
        img = batch["img"].to(device)
        masks_gt = batch["masks"].to(device)   # (B,1,H,W)
        bboxes_gt = batch["bboxes"]            # list
        path = batch["path"][0]

        out = model(img)
        seg_out, det_outs = out["seg"], out["det"]

        # Debug shapes
        if isinstance(det_outs, (list, tuple)):
            # Truy cập vào list bên trong (det_outs[0]) trước khi lặp
            inner_list = det_outs[0]
            det_shapes = [f.shape for f in inner_list]
        else:
            # Trường hợp det_outs là một tensor duy nhất
            det_shapes = [det_outs.shape]
        print(f"[DEBUG] Batch {i}: seg.shape={seg_out.shape}, det_outs={len(det_shapes)} maps: {det_shapes}")

        # --- segmentation prediction ---
        seg_pred = torch.sigmoid(seg_out).cpu()[0, 0].numpy()
        seg_pred_bin = (seg_pred > 0.5).astype(np.uint8)

        if masks_gt.shape[0] > 0:
            mask_gt = masks_gt[0, 0].cpu().numpy()
            iou = compute_iou(torch.tensor(seg_pred_bin), torch.tensor(mask_gt))
            dice = (2 * (seg_pred_bin * mask_gt).sum() + 1) / (seg_pred_bin.sum() + mask_gt.sum() + 1)
            ious.append(iou.item())
            dices.append(dice)

        # --- detection prediction ---
        pred_boxes = []
        preds = det_outs[0] if isinstance(det_outs, (list, tuple)) else det_outs
        preds = preds[0].cpu()  # batch 0, shape [num_preds, 5+num_classes]

        for p in preds:
            x, y, w, h, obj = p[:5].tolist()
            if obj > 0.5:
                box = xywh2xyxy(torch.tensor([[x, y, w, h]])).squeeze(0).numpy()
                pred_boxes.append((obj, box.tolist()))

        gt_boxes = []
        if len(bboxes_gt) > 0:
            for box in bboxes_gt[0]:
                _, cx, cy, bw, bh = box
                cx, cy, bw, bh = cx*cfg["data"]["img_size"], cy*cfg["data"]["img_size"], bw*cfg["data"]["img_size"], bh*cfg["data"]["img_size"]
                x1, y1 = int(cx - bw/2), int(cy - bh/2)
                x2, y2 = int(cx + bw/2), int(cy + bh/2)
                gt_boxes.append([x1, y1, x2, y2])

        if len(gt_boxes) > 0:
            aps.append(compute_map(pred_boxes, gt_boxes, iou_thresh=0.5))

        # --- visualization ---
        img_np = (img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask_color = np.zeros_like(img_np)
        mask_color[..., 1] = seg_pred_bin * 255
        blended = cv2.addWeighted(img_np, 0.7, mask_color, 0.3, 0)

        for gb in gt_boxes:  # GT bboxes (blue)
            cv2.rectangle(blended, (gb[0], gb[1]), (gb[2], gb[3]), (255, 0, 0), 2)
        for _, pb in pred_boxes:  # Pred bboxes (yellow)
            x1, y1, x2, y2 = map(int, pb)
            cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 255), 2)

        save_path = os.path.join("outputs/vis_eval", os.path.basename(path))
        cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    # --- Final metrics ---
    mean_iou = np.mean(ious) if len(ious) else 0
    mean_dice = np.mean(dices) if len(dices) else 0
    mean_map = np.mean(aps) if len(aps) else 0

    print(f"Evaluation results on {split} set:")
    print(f"  mIoU   = {mean_iou:.4f}")
    print(f"  Dice   = {mean_dice:.4f}")
    print(f"  mAP@0.5 = {mean_map:.4f}")
    print("Visualizations saved in outputs/vis_eval/")

if __name__ == "__main__":
    evaluate()
