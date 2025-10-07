# scripts/eval.py
import os, yaml, torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from datasets.dataset import PotholeDataset
from models.yolov12_bifpn import YOLOv12_BiFPN
from utils.metrics import compute_seg_metrics, plot_confusion_matrix
from utils.visualize import visualize_sample
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import cv2
# === Overlay helpers ===
def to_uint8_img(t):
    """
    t: torch.Tensor (3,H,W) hoặc (H,W,3), giá trị [0,1] hoặc [0,255]
    -> numpy uint8 (H,W,3) RGB
    """
    import numpy as np
    if t.ndim == 3 and t.shape[0] == 3:  # (3,H,W) -> (H,W,3)
        t = t.permute(1, 2, 0)
    arr = t.detach().cpu().numpy()
    if arr.max() <= 1.0:
        arr = (arr * 255.0).clip(0, 255)
    return arr.astype(np.uint8)

def overlay_masks_on_image(img_rgb_u8, pred_bin, gt_bin=None, alpha=0.45):
    """
    img_rgb_u8: np.uint8 (H,W,3), RGB
    pred_bin: np.bool_(H,W) dự đoán
    gt_bin:   np.bool_(H,W) ground-truth (tùy chọn)
    alpha: độ trong suốt của vùng màu

    Màu quy ước:
      - TP (pred & gt): vàng (255,255,0)
      - FP (pred & ~gt): xanh lá (0,255,0)
      - FN (~pred & gt): đỏ (255,0,0)
      - Nếu không có gt_bin: toàn bộ pred = xanh lá
    """
    import numpy as np
    h, w, _ = img_rgb_u8.shape
    color = np.zeros_like(img_rgb_u8, dtype=np.uint8)

    if gt_bin is None:
        # Chỉ có pred: tô xanh lá
        color[pred_bin] = (0, 255, 0)
    else:
        tp = pred_bin & gt_bin
        fp = pred_bin & (~gt_bin)
        fn = (~pred_bin) & gt_bin
        color[tp] = (255, 255, 0)   # vàng
        color[fp] = (0, 255, 0)     # xanh lá
        color[fn] = (255, 0, 0)     # đỏ

    # alpha blend: out = (1-alpha)*img + alpha*color
    out = img_rgb_u8.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)

def evaluate():
    # --- Load config (UTF-8) ---
    with open("configs/configs.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Dataset / Loader ---
    dataset = PotholeDataset(cfg["data"]["val"], img_size=cfg["data"]["img_size"])
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=cfg["data"]["num_workers"], collate_fn=PotholeDataset.collate_fn,
    )

    # --- Model / Checkpoint ---
    model = YOLOv12_BiFPN(num_classes=cfg["data"]["num_classes"]).to(device)
    ckpt_path = os.path.join(cfg["train"]["save_dir"], "best.pt")  # dùng best.pt
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    os.makedirs("outputs/eval_vis", exist_ok=True)

    # --- Running metrics ---
    dices, pixel_accs, ious = [], [], []
    cm_total = np.zeros((2, 2), dtype=np.int64)  # [[TN, FP],[FN, TP]]

    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs = batch["img"].to(device)
            masks_gt = batch["masks"].to(device)
            out = model(imgs)

            # dùng PROB thay vì LOGIT
            seg_prob = torch.sigmoid(out["seg"])

            # segmentation metrics (Dice/IoU/PixelAcc)
            d, pa, iu = compute_seg_metrics(seg_prob, masks_gt)
            dices.append(d); pixel_accs.append(pa); ious.append(iu)

            # accumulate confusion matrix (pixel-level)
            pred_bin = (seg_prob > 0.5).cpu().numpy().astype(np.uint8).ravel()
            gt_bin   = (masks_gt > 0.5).cpu().numpy().astype(np.uint8).ravel()
            cm_total += confusion_matrix(gt_bin, pred_bin, labels=[0, 1])

            # visualize vài mẫu
            # if i < 10:
            #     #plot_confusion_matrix(seg_prob[0].cpu(), masks_gt[0].cpu(),
            #     #                      save_path=f"outputs/eval_vis/confmat_{i}.png")
            #     visualize_sample(
            #         imgs, mask_pred=seg_prob, mask_gt=masks_gt,
            #         save_path=f"outputs/eval_vis/sample_{i}.png"
            #     )
            if i < 10: 
                
                save_dir = "outputs/eval_vis"
                os.makedirs(save_dir, exist_ok=True)

                img_u8 = to_uint8_img(imgs[0])  # (H,W,3) RGB uint8

                # xTạo mask nhị phân (H,W)
                pred = (seg_prob[0, 0] > 0.5).cpu().numpy().astype(bool)
                gt   = (masks_gt[0, 0] > 0.5).cpu().numpy().astype(bool)

                # Overlay dự đoán + GT (TP= vàng, FP= xanh lá, FN= đỏ)
                overlay_both = overlay_masks_on_image(img_u8, pred_bin=pred, gt_bin=gt, alpha=0.45)
                cv2.imwrite(os.path.join(save_dir, f"overlay_pred_gt_{i}.png"), cv2.cvtColor(overlay_both, cv2.COLOR_RGB2BGR))

    # aggregate results
    results = {}
    if dices:       results["Dice"] = sum(dices)/len(dices)
    if pixel_accs:  results["PixelAcc"] = sum(pixel_accs)/len(pixel_accs)
    if ious:        results["IoU"] = sum(ious)/len(ious)

    TN, FP = int(cm_total[0, 0]), int(cm_total[0, 1])
    FN, TP = int(cm_total[1, 0]), int(cm_total[1, 1])
    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    f1_pixel  = 2 * precision * recall / (precision + recall + 1e-6)

    results.update({"PixelPrecision": precision, "PixelRecall": recall, "PixelF1": f1_pixel,
                    "TN": TN, "FP": FP, "FN": FN, "TP": TP})

    for k, v in results.items():
        print(f"{k}: {v if isinstance(v, int) else f'{v:.4f}'}")

    pd.DataFrame([results]).to_csv("outputs/eval_results.csv", index=False)
    print("Saved results to outputs/eval_results.csv")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_total, display_labels=["Background","Pothole"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Aggregate Confusion Matrix (Val)")
    plt.savefig("outputs/eval_vis/confmat_aggregate.png", bbox_inches="tight")
    plt.close()

    # dataset stats
    try:
        def count_lines(p):
            with open(p, "r", encoding="utf-8") as f:
                return sum(1 for x in f if x.strip())
        nL = count_lines(cfg["data"]["train_l"])
        nU = count_lines(cfg["data"]["train_u"])
        nV = count_lines(cfg["data"]["val"])
        nT = count_lines(cfg["data"]["test"])
        ratio = f"{nL}:{nU}" if nU else f"{nL}:0"
        print(f"\n[DATASET] Labeled={nL}, Unlabeled={nU} (ratio {ratio}), Val={nV}, Test={nT}")
        print(f"[AUG] augment={cfg['train'].get('augment', False)}, img_size={cfg['data']['img_size']}")
    except Exception as e:
        print(f"[WARN] dataset stats skipped: {e}")

if __name__ == "__main__":
    evaluate()
