import re
import matplotlib.pyplot as plt

# Đọc file log
log_file = r"C:\Users\hauhm\Downloads\hole_project\hole_project\outputs\checkpoints\best.pt"
with open(log_file, "r") as f:
    lines = f.readlines()

# Khởi tạo list
epochs = []
sup_loss, cons_loss, total_loss = [], [], []
bce_loss, dice_loss, pixel_acc = [], [], []

# Regex để lấy số liệu
pattern = (r"\[Epoch (\d+)\] sup=([\d\.]+), cons=([\d\.]+), total=([\d\.]+), "
           r"BCE=([\d\.]+), Dice=([\d\.]+), PixelAcc=([\d\.]+)")

for line in lines:
    match = re.search(pattern, line)
    if match:
        epochs.append(int(match.group(1)))
        sup_loss.append(float(match.group(2)))
        cons_loss.append(float(match.group(3)))
        total_loss.append(float(match.group(4)))
        bce_loss.append(float(match.group(5)))
        dice_loss.append(float(match.group(6)))
        pixel_acc.append(float(match.group(7)))

# Vẽ biểu đồ
plt.figure(figsize=(12,8))

# Loss curves
plt.subplot(2,1,1)
plt.plot(epochs, total_loss, label="Total Loss")
plt.plot(epochs, sup_loss, label="Sup Loss")
plt.plot(epochs, cons_loss, label="Cons Loss")
plt.plot(epochs, bce_loss, label="BCE Loss")
plt.plot(epochs, dice_loss, label="Dice Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curves")
plt.legend()
plt.grid(True)

# Pixel Accuracy
plt.subplot(2,1,2)
plt.plot(epochs, pixel_acc, label="Pixel Accuracy", color="purple")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Pixel Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
