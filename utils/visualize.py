import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

def visualize_sample(img, mask_pred=None, mask_gt=None, pred_boxes=None, gt_boxes=None, save_path=None, show=False):
    """
    Hiển thị kết quả dự đoán: ảnh gốc + mask + bounding boxes.

    Args:
        img (Tensor): (3,H,W) hoặc (B,3,H,W)
        mask_pred (Tensor): (1,H,W) hoặc (B,1,H,W)
        mask_gt (Tensor): (1,H,W) hoặc (B,1,H,W)
        pred_boxes (list): [(score, [x1,y1,x2,y2])]
        gt_boxes (list): [(1.0, [x1,y1,x2,y2])]
        save_path (str): nếu không None thì lưu ra file
        show (bool): nếu True thì plt.show()
    """
    if img.ndim == 4:  # (B,3,H,W) -> lấy ảnh đầu tiên
        img = img[0]
    img = img.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC

    if mask_pred is not None and mask_pred.ndim == 4:
        mask_pred = mask_pred[0]
    if mask_gt is not None and mask_gt.ndim == 4:
        mask_gt = mask_gt[0]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Ảnh gốc + boxes
    axs[0].imshow(img)
    axs[0].set_title("Image + Boxes")
    if pred_boxes:
        for score, box in pred_boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=2, edgecolor="r", facecolor="none")
            axs[0].add_patch(rect)
    if gt_boxes:
        for _, box in gt_boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=2, edgecolor="g", facecolor="none")
            axs[0].add_patch(rect)

    # Mask dự đoán
    if mask_pred is not None:
        axs[1].imshow(mask_pred.squeeze().cpu(), cmap="gray")
        axs[1].set_title("Predicted Mask")
    else:
        axs[1].axis("off")

    # Mask ground truth
    if mask_gt is not None:
        axs[2].imshow(mask_gt.squeeze().cpu(), cmap="gray")
        axs[2].set_title("Ground Truth Mask")
    else:
        axs[2].axis("off")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def save_curve(train_losses, val_losses, save_dir):
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.title("Loss Curve")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "training_curve.png"))
    plt.close()
