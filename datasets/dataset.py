import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class PotholeDataset(Dataset):
    def __init__(self, list_file, img_size=640):
        with open(list_file, "r") as f:
            self.samples = [line.strip().split() for line in f.readlines()]
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        parts = self.samples[idx]
        img_path = parts[0]
        label_path = None if len(parts) == 1 or parts[1] == "-1" else parts[1]

        # đọc ảnh
        img = cv2.imread(img_path)
        assert img is not None, f"Không load được ảnh {img_path}"
        h0, w0 = img.shape[:2]

        # resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img).float()

        bboxes, masks = [], []

        if label_path and os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                cls = int(parts[0])
                coords = list(map(float, parts[1:]))
                poly = np.array(coords).reshape(-1, 2)

                # scale polygon sang img_size
                poly_x = poly[:, 0] * self.img_size
                poly_y = poly[:, 1] * self.img_size
                poly_scaled = np.stack([poly_x, poly_y], axis=1).astype(np.int32)

                # bbox
                xmin, ymin = poly_x.min(), poly_y.min()
                xmax, ymax = poly_x.max(), poly_y.max()
                bw, bh = xmax - xmin, ymax - ymin
                cx, cy = xmin + bw / 2, ymin + bh / 2
                bbox = [cls, cx / self.img_size, cy / self.img_size,
                        bw / self.img_size, bh / self.img_size]
                bboxes.append(bbox)

                # mask
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                cv2.fillPoly(mask, [poly_scaled], 1)
                masks.append(mask)
        if len(masks) > 0:
            merged_mask = np.max(np.stack(masks, axis=0), axis=0)  # (H,W)
        else:
            merged_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        merged_mask = torch.from_numpy(merged_mask).unsqueeze(0).float()  # (1,H,W)

        return {
            "img": img_tensor,
            "bboxes": torch.tensor(bboxes, dtype=torch.float32),
            "masks": merged_mask,  # chỉ 1 mask/ảnh
            "path": img_path
        }
    def collate_fn(batch):
        imgs = torch.stack([item["img"] for item in batch], dim=0)
        bboxes = [item["bboxes"] for item in batch]
        masks = torch.stack([item["masks"] for item in batch], dim=0)  # (B,1,H,W)
        paths = [item["path"] for item in batch]
        return {"img": imgs, "bboxes": bboxes, "masks": masks, "path": paths}
