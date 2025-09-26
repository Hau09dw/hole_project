import os
import matplotlib.pyplot as plt

def save_curve(train_losses, val_losses, save_dir):
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.title("Loss Curve")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "training_curve.png"))
    plt.close()
