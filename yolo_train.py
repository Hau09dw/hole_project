from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === PATH DATASET ===
dataset_path = r"C:\Users\hauhm\Downloads\hole_project\hole_project\data_segment\data.yaml"
data_root = Path(r"C:\Users\hauhm\Downloads\hole_project\hole_project\data_segment")


# ==================== TRAIN MODEL ====================
def train_model():
    model = YOLO("yolo12-seg.yaml")

    results = model.train(
        data=dataset_path,
        epochs=20,
        imgsz=640,
        batch=16,
        patience=20,
        mosaic=1.0,
        mixup=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        amp=False,
        device="0" if torch.cuda.is_available() else "cpu",
        project="runs/train",
        name="yolov12l-pothole",
        plots=True,
    )
    return model, results


# ==================== EVALUATE ====================
def evaluate_model(model, dataset=dataset_path):
    metrics = model.val(data=dataset, split="val", plots=True)
    print("Validation metrics:", metrics)

    save_dir = Path(getattr(model.trainer, "save_dir", "runs/val/exp"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Lưu metrics ra file
    with open(save_dir / "metrics.txt", "w") as f:
        f.write(str(metrics))

    # === Confusion Matrix ===
    preds, gts = [], []
    val_imgs = data_root / "val"
    for img_path in val_imgs.glob("*.jpg"):  
        results = model.predict(
            source=img_path, conf=0.25, imgsz=640, verbose=False
        )
        for r in results:
            if len(r.boxes.cls) > 0:
                preds.extend(r.boxes.cls.cpu().numpy().astype(int).tolist())

            # Ground truth đọc từ file label
            label_file = (
                str(img_path)
                .replace("images", "labels")
                .rsplit(".", 1)[0]
                + ".txt"
            )
            if Path(label_file).exists():
                with open(label_file) as lf:
                    for line in lf:
                        cls_id = int(line.split()[0])
                        gts.append(cls_id)

    if len(preds) > 0 and len(gts) > 0:
        cm = confusion_matrix(gts, preds, labels=np.unique(gts + preds))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.savefig(save_dir / "confusion_matrix.png")
        plt.close()


# ==================== PLOT TRAINING CURVES ====================
def plot_training_curves(save_dir="runs/train/yolov12l-pothole"):
    csv_file = Path(save_dir) / "results.csv"
    if not csv_file.exists():
        print("⚠️ results.csv not found, cannot plot curves")
        return

    df = pd.read_csv(csv_file)
    epochs = df.index + 1

    # Loss curves
    plt.plot(epochs, df["train/box_loss"], label="Train Box Loss")
    plt.plot(epochs, df["train/cls_loss"], label="Train Cls Loss")
    if "train/dfl_loss" in df.columns:
        plt.plot(epochs, df["train/dfl_loss"], label="Train DFL Loss")
    plt.plot(epochs, df["val/box_loss"], label="Val Box Loss")
    plt.plot(epochs, df["val/cls_loss"], label="Val Cls Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(Path(save_dir) / "loss_curve.png")
    plt.close()

    # mAP curve (accuracy proxy)
    if "metrics/mAP50-95(B)" in df.columns:
        plt.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP50-95")
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.legend()
        plt.title("Validation mAP Curve")
        plt.savefig(Path(save_dir) / "map_curve.png")
        plt.close()


# ==================== VISUALIZE TEST RESULTS ====================
def visualize_results(model, dataset=dataset_path):
    results = model.val(data=dataset, split="test", plots=True)
    print("Test metrics:", results)

    test_images = data_root / "images" / "test"
    sample_imgs = list(test_images.glob("*.jpg"))[:5] 
    for img in sample_imgs:
        model.predict(
            img,
            save=True,
            project="runs/predict",
            name="pothole_demo",
            imgsz=640,
            conf=0.25,
        )


# ==================== MAIN ====================
if __name__ == "__main__":
    model, results = train_model()
    evaluate_model(model)
    plot_training_curves("runs/train/yolov12l-pothole")
    visualize_results(model)
