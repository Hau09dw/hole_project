import os
import numpy as np
from datasets.dataset import LabeledDataset, UnlabeledDataset

def analyze_dataset(list_file, img_size=640, labeled=True):
    if labeled:
        ds = LabeledDataset(list_file, img_size=img_size)
    else:
        ds = UnlabeledDataset(list_file, img_size=img_size)

    num_images = len(ds)
    print(f"📊 {list_file}")
    print(f"   - Số ảnh: {num_images}")

    if labeled:
        num_objects = 0
        class_count = {}
        box_areas, mask_areas = [], []

        for i in range(num_images):
            _, target = ds[i]
            num_objects += len(target["labels"])
            for c in target["labels"]:
                c = int(c.item())
                class_count[c] = class_count.get(c, 0) + 1

            for box in target["boxes"]:
                x1, y1, x2, y2 = box.tolist()
                box_areas.append((x2-x1)*(y2-y1))

            if target["masks"].numel() > 0:
                mask_areas.append(target["masks"].sum().item())

        print(f"   - Số đối tượng: {num_objects}")
        print(f"   - Số đối tượng trung bình/ảnh: {num_objects/num_images:.2f}")
        print(f"   - Phân bố class: {class_count}")
        print(f"   - Diện tích box trung bình: {np.mean(box_areas) if box_areas else 0:.1f}")
        print(f"   - Diện tích mask trung bình: {np.mean(mask_areas) if mask_areas else 0:.1f}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    analyze_dataset("data_segment/train_l.txt", labeled=True)
    analyze_dataset("data_segment/val.txt", labeled=True)
    analyze_dataset("data_segment/train_u.txt", labeled=False)
