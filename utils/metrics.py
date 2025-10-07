import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def compute_iou(pred_mask, gt_mask):
    pred = (pred_mask > 0.5).float()
    inter = (pred * gt_mask).sum()
    union = pred.sum() + gt_mask.sum() - inter
    return (inter + 1e-6) / (union + 1e-6)

def bbox_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / (union + 1e-6)

def compute_map(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    pred_boxes: [(score, [x1,y1,x2,y2])]
    gt_boxes: [(1.0, [x1,y1,x2,y2])]
    """
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 1.0
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0

    pred_boxes = sorted(pred_boxes, key=lambda x: x[0], reverse=True)
    tp, fp = np.zeros(len(pred_boxes)), np.zeros(len(pred_boxes))
    matched = set()

    for i, (score, pb) in enumerate(pred_boxes):
        best_iou, best_j = 0, -1
        for j, (_, gb) in enumerate(gt_boxes):
            iou = bbox_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j not in matched:
            tp[i] = 1
            matched.add(best_j)
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / (len(gt_boxes) + 1e-6)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

    ap = 0.0
    for r in np.linspace(0, 1, 11):
        p = precisions[recalls >= r].max() if np.any(recalls >= r) else 0
        ap += p / 11
    return ap

def compute_precision_recall(pred_boxes, gt_boxes, iou_thresh=0.5):
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 1.0, 1.0
    if len(pred_boxes) == 0:
        return 0.0, 0.0
    if len(gt_boxes) == 0:
        return 0.0, 0.0

    tp, fp = 0, 0
    matched = set()

    for score, pb in pred_boxes:
        best_iou, best_j = 0, -1
        for j, (_, gb) in enumerate(gt_boxes):
            iou = bbox_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j not in matched:
            tp += 1
            matched.add(best_j)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision, recall

def compute_seg_metrics(pred_mask, gt_mask):
    pred = (pred_mask > 0.5).float()
    inter = (pred * gt_mask).sum().item()
    union = (pred + gt_mask - pred * gt_mask).sum().item()
    dice = (2 * inter + 1e-6) / (pred.sum().item() + gt_mask.sum().item() + 1e-6)
    pixel_acc = (pred == gt_mask).float().mean().item()
    iou = (inter + 1e-6) / (union + 1e-6)
    return dice, pixel_acc, iou

def plot_confusion_matrix(pred_mask, gt_mask, save_path=None):
    pred = (pred_mask > 0.5).cpu().numpy().astype(int).flatten()
    gt = gt_mask.cpu().numpy().astype(int).flatten()
    cm = confusion_matrix(gt, pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Background","Pothole"])
    disp.plot(cmap="Blues", values_format="d")
    if save_path:
        plt.savefig(save_path)
    plt.close()
