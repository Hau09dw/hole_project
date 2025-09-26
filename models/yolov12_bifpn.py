import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.head import Detect

MODEL_YAML = os.path.join("yolov12", "ultralytics", "cfg", "models", "v12", "yolov12.yaml")


# -----------------------------
# BiFPN
# -----------------------------
class BiFPN(nn.Module):
    def __init__(self, channels, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(ch, out_channels, 1, bias=False) for ch in channels])
        self.lateral_bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in channels])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats):
        P3, P4, P5 = [self.relu(bn(conv(f))) for conv, bn, f in zip(self.lateral_convs, self.lateral_bns, feats)]
        # Top-down
        P4_ = P4 + F.interpolate(P5, scale_factor=2, mode="nearest")
        P3_ = P3 + F.interpolate(P4_, scale_factor=2, mode="nearest")
        # Bottom-up
        P4_out = P4_ + F.max_pool2d(P3_, 2)
        P5_out = P5 + F.max_pool2d(P4_out, 2)
        return [P3_, P4_out, P5_out]


# -----------------------------
# YOLOv12 + BiFPN + Segmentation
# -----------------------------

class YOLOv12_BiFPN(nn.Module):
    def __init__(self, num_classes=1, seg_dim=1, img_size=640):
        super().__init__()
        full_model = DetectionModel(cfg=MODEL_YAML)
        self.backbone = nn.Sequential(*list(full_model.model)[:-1])

        dummy = torch.zeros(1, 3, img_size, img_size)
        cache, x, chs = {}, dummy, []
        for i, layer in enumerate(self.backbone):
            if hasattr(layer, "f"):
                f = layer.f if isinstance(layer.f, list) else [layer.f]
                inputs = [x if j == -1 else cache[j] for j in f]
                x = layer(inputs[0] if len(inputs) == 1 else inputs)
            else:
                x = layer(x)
            cache[i] = x
            if i in [14, 17, 20]:
                chs.append(x.shape[1])

        # BiFPN
        self.neck = BiFPN(chs, out_channels=256)

        # Detection head (không dùng loss chính)
        self.detect = Detect(nc=num_classes, ch=[256, 256, 256])

        # Segmentation head (loss chính)
        self.seg_head = nn.Sequential(
            nn.Conv2d(256 * 3, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(size=(img_size, img_size), mode="bilinear", align_corners=False),
            nn.Conv2d(128, seg_dim, 1),
        )

    def forward(self, x):
        feats, cache = [], {}
        for i, layer in enumerate(self.backbone):
            if hasattr(layer, "f"):
                f = layer.f if isinstance(layer.f, list) else [layer.f]
                inputs = [x if j == -1 else cache[j] for j in f]
                x = layer(inputs[0] if len(inputs) == 1 else inputs)
            else:
                x = layer(x)
            cache[i] = x
            if i in [14, 17, 20]:
                feats.append(x)

        fused_feats = self.neck(feats)

        # Segmentation out (main task)
        up_feats = [F.interpolate(f, size=(fused_feats[0].shape[2], fused_feats[0].shape[3]), mode="nearest")
                    for f in fused_feats]
        seg_in = torch.cat(up_feats, dim=1)
        seg_out = self.seg_head(seg_in)

        # Detection out (optional)
        det_outs = self.detect(fused_feats)

        return {"seg": seg_out, "det": det_outs}