import torch
from ultralytics.nn.tasks import DetectionModel
import os

MODEL_YAML = os.path.join("yolov12", "ultralytics", "cfg", "models", "v12", "yolov12.yaml")

def check_backbone(img_size=640):
    # Load YOLOv12 backbone
    model = DetectionModel(cfg=MODEL_YAML)
    backbone = model.model[:-1]  # bỏ Detect head

    x = torch.zeros(1, 3, img_size, img_size)
    cache = {}

    print("=== Backbone Layers ===")
    for i, layer in enumerate(backbone):
        # xác định input theo layer.f
        if hasattr(layer, "f"):
            f = layer.f
            if isinstance(f, int):
                f = [f]
            inputs = []
            for j in f:
                if j == -1:
                    inputs.append(x)
                else:
                    inputs.append(cache[j])
            if len(inputs) == 1:
                inp = inputs[0]
            else:
                inp = inputs
            x = layer(inp)
        else:
            x = layer(x)

        cache[i] = x
        print(f"Layer {i:02d}: {layer.__class__.__name__:<20} -> {tuple(x.shape)}")

    print("========================")

if __name__ == "__main__":
    check_backbone()
