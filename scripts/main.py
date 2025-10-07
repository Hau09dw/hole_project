import torch
from torch.utils.data import DataLoader
from datasets.dataset import PotholeDataset
from utils.losses import FocalBCELoss, DiceLoss 
from models.yolov12_bifpn import YOLOv12_BiFPN
from trainers.mean_teacher import MeanTeacherTrainer
import yaml
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from utils.logger import save_log
import re

def plot_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 1. Loss curves
    plt.figure()
    plt.plot(history["total"], label="Total Loss")
    plt.plot(history["bce"], label="BCE Loss")
    plt.plot(history["dice"], label="Dice Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.title("Training Loss Curves")
    plt.savefig(os.path.join(save_dir, "loss_curves.png")); plt.close()

    # 2. Pixel accuracy
    plt.figure()
    plt.plot(history["pixel_acc"], label="Pixel Accuracy", color="green")
    plt.xlabel("Epoch"); plt.ylabel("Pixel Acc"); plt.title("Pixel Accuracy per Epoch")
    plt.legend(); plt.savefig(os.path.join(save_dir, "pixel_acc.png")); plt.close()

    # 3. Seg vs Consistency loss
    plt.figure()
    plt.plot(history["seg"], label="Seg Loss")
    plt.plot(history["cons"], label="Consistency Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Seg vs Consistency Loss")
    plt.legend(); plt.savefig(os.path.join(save_dir, "seg_vs_cons.png")); plt.close()

def get_latest_checkpoint(save_dir):
    """Tìm checkpoint epochN.pt mới nhất trong thư mục save_dir"""
    ckpts = [f for f in os.listdir(save_dir) if re.match(r"epoch(\d+)\.pt", f)]
    if not ckpts:
        return None, 0
    # lấy số epoch lớn nhất
    epochs = [int(re.findall(r"epoch(\d+)\.pt", f)[0]) for f in ckpts]
    max_epoch = max(epochs)
    latest_ckpt = os.path.join(save_dir, f"epoch{max_epoch}.pt")
    return latest_ckpt, max_epoch

def main():
    # Load config (UTF-8)
    with open("configs/configs.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset (nối augment từ config)
    dataset_l = PotholeDataset(cfg["data"]["train_l"], img_size=cfg["data"]["img_size"],
                               use_aug=cfg["train"].get("augment", False))
    dataset_u = PotholeDataset(cfg["data"]["train_u"], img_size=cfg["data"]["img_size"],
                               use_aug=cfg["train"].get("augment", False))

    loader_l = DataLoader(
        dataset_l, batch_size=cfg["train"]["batch"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], collate_fn=PotholeDataset.collate_fn,
    )
    loader_u = DataLoader(
        dataset_u, batch_size=cfg["train"]["batch"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], collate_fn=PotholeDataset.collate_fn,
    )

    # Model
    model = YOLOv12_BiFPN(num_classes=cfg["data"]["num_classes"]).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))

    # Segmentat                 ion loss = Focal-BCE + Dice
    bce = FocalBCELoss(alpha=0.75, gamma=2.0)
    dice = DiceLoss()
    def seg_loss_fn(pred, target): return bce(pred, target) + dice(pred, target)

    # Trainer
    trainer = MeanTeacherTrainer(model, optimizer, cfg, seg_loss_fn, device=device)

    # Directories
    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)

    # Loss history (resume từ file CSV nếu có)
    hist_path = os.path.join(cfg["train"]["save_dir"], "training_history.csv")
    if os.path.exists(hist_path):
        history = pd.read_csv(hist_path).to_dict(orient="list")
    else:
        history = {"seg": [], "cons": [], "total": [], "bce": [], "dice": [], "pixel_acc": []}

    # Checkpoint resume
    latest_ckpt, last_epoch = get_latest_checkpoint(cfg["train"]["save_dir"])
    if latest_ckpt:
        print(f"[INFO] Resume from checkpoint {latest_ckpt} (epoch {last_epoch})")
        model.load_state_dict(torch.load(latest_ckpt, map_location=device))
        start_epoch = last_epoch
    else:
        print("[INFO] No checkpoint found, start training from scratch")
        start_epoch = 0

    # Chọn best theo **min DiceLoss**
    best_dice = min(history["dice"]) if len(history["dice"]) > 0 else float('inf')
    best_epoch = -1

    # Training loop
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        cfg["train"]["cur_epoch"] = epoch + 1

        seg_epoch = cons_epoch = total_epoch = 0.0
        bce_epoch = dice_epoch = pixel_acc_epoch = 0.0
        n_batches = 0

        pbar = tqdm(
            zip(loader_l, loader_u),
            total=min(len(loader_l), len(loader_u)),
            desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}",
            unit="batch",
        )

        for batch_l, batch_u in pbar:
            seg_loss, cons_loss, total_loss, bce_val, dice_val, pixel_acc = trainer.train_step(batch_l, batch_u)

            seg_epoch += seg_loss; cons_epoch += cons_loss; total_epoch += total_loss
            bce_epoch += bce_val;  dice_epoch += dice_val;  pixel_acc_epoch += pixel_acc
            n_batches += 1

            pbar.set_postfix({"seg": f"{seg_loss:.2f}", "cons": f"{cons_loss:.2f}", "total": f"{total_loss:.2f}"})

        # averages
        seg_epoch /= n_batches; cons_epoch /= n_batches; total_epoch /= n_batches
        bce_epoch /= n_batches; dice_epoch /= n_batches; pixel_acc_epoch /= n_batches

        # save history
        history["seg"].append(seg_epoch)
        history["cons"].append(cons_epoch)
        history["total"].append(total_epoch)
        history["bce"].append(bce_epoch)
        history["dice"].append(dice_epoch)
        history["pixel_acc"].append(pixel_acc_epoch)

        # save checkpoint mỗi epoch
        ckpt_path = os.path.join(cfg["train"]["save_dir"], f"epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)

        # save best model theo **min DiceLoss**
        if dice_epoch < best_dice:
            best_dice = dice_epoch
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(cfg["train"]["save_dir"], "best.pt"))

        # luôn lưu last model
        torch.save(model.state_dict(), os.path.join(cfg["train"]["save_dir"], "last.pt"))

        print(
            f"\n[Epoch {epoch+1}] seg={seg_epoch:.4f}, cons={cons_epoch:.4f}, "
            f"total={total_epoch:.4f}, BCE={bce_epoch:.4f}, "
            f"Dice={dice_epoch:.4f}, PixelAcc={pixel_acc_epoch:.4f}"
        )
        save_log("outputs/train_log.txt", epoch + 1, seg_epoch, cons_epoch, total_epoch,
                 bce_epoch, dice_epoch, pixel_acc_epoch)

    # vẽ biểu đồ
    plot_curves(history, save_dir=cfg["train"]["save_dir"])

    # lưu history CSV
    pd.DataFrame(history).to_csv(os.path.join(cfg["train"]["save_dir"], "training_history.csv"), index=False)

    print(f"\nBest model saved at epoch {best_epoch} with DiceLoss={best_dice:.4f}")
    print("Training curves and metrics saved!")

if __name__ == "__main__":
    main()
