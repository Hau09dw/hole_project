import os
import random

def generate_split_files(root, output_labeled, output_unlabeled, labeled_ratio=0.3):
    """
    root: thư mục gốc chứa data_segment/train (có images/ và labels/)
    output_labeled: file txt chứa 30% ảnh có nhãn
    output_unlabeled: file txt chứa 70% ảnh chỉ có ảnh
    labeled_ratio: tỉ lệ ảnh có nhãn (mặc định 0.3)
    """
    img_dir = os.path.join(root, "images")
    label_dir = os.path.join(root, "labels")

    samples = []
    for img_file in os.listdir(img_dir):
        if img_file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
            if os.path.exists(label_path):
                samples.append((img_path, label_path))
            else:
                print(f"[WARN] Missing label for {img_file}")

    # Shuffle và split
    random.shuffle(samples)
    num_labeled = int(len(samples) * labeled_ratio)
    labeled_samples = samples[:num_labeled]
    unlabeled_samples = samples[num_labeled:]

    # Lưu file labeled (ảnh + nhãn)
    with open(output_labeled, "w") as f:
        for img_path, label_path in labeled_samples:
            f.write(f"{img_path} {label_path}\n")

    # Lưu file unlabeled (chỉ ảnh)
    with open(output_unlabeled, "w") as f:
        for img_path, _ in unlabeled_samples:
            f.write(f"{img_path}\n")

    print(f"Saved {len(labeled_samples)} labeled samples to {output_labeled}")
    print(f"Saved {len(unlabeled_samples)} unlabeled samples to {output_unlabeled}")


def generate_eval_files(root, split, output_file):
    """
    root: thư mục gốc data_segment
    split: valid/test
    output_file: file txt cần ghi
    """
    img_dir = os.path.join(root, split, "images")
    label_dir = os.path.join(root, split, "labels")

    samples = []
    for img_file in os.listdir(img_dir):
        if img_file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
            if os.path.exists(label_path):
                samples.append(f"{img_path} {label_path}\n")

    with open(output_file, "w") as f:
        f.writelines(samples)

    print(f"Saved {len(samples)} samples to {output_file}")


if __name__ == "__main__":
    root = "data_segment"

    # Chia train thành labeled + unlabeled
    generate_split_files(
        root=os.path.join(root, "train"),
        output_labeled="train_l.txt",
        output_unlabeled="train_u.txt",
        labeled_ratio=0.3
    )

    # Tạo file eval
    generate_eval_files(root, "valid", "val.txt")
    generate_eval_files(root, "test", "test.txt")
