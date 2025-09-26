# utils/logger.py
import os

def save_log(log_path, epoch, sup, cons, total, bce, dice, pixel_acc):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(
            f"[Epoch {epoch}] sup={sup:.4f}, cons={cons:.4f}, total={total:.4f}, "
            f"BCE={bce:.4f}, Dice={dice:.4f}, PixelAcc={pixel_acc:.4f}\n"
        )
