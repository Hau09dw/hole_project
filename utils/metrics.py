import torch
import numpy as np

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
    gt_boxes: [[x1,y1,x2,y2]]
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
        for j, gb in enumerate(gt_boxes):
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
