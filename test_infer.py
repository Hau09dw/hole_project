import torch
import cv2
from models.yolov12_bifpn import YOLOv12_BiFPN
from utils.decode import decode_yolo_output_multi
from utils.visualize import overlay_segmentation, draw_boxes
from datasets.dataset import LabeledDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = YOLOv12_BiFPN(num_classes=1)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.to(device).eval()

# Load ảnh
img_path = "data_segment/test/img001.jpg"
img = cv2.imread(img_path)
img_resized = cv2.resize(img, (640,640))
img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2,0,1).unsqueeze(0)/255.
img_tensor = img_tensor.to(device)

with torch.no_grad():
    out = model(img_tensor)

# --- Detection ---
batch_boxes, batch_logits = decode_yolo_output_multi(out["det"], num_classes=1, img_size=640)
boxes, logits = batch_boxes[0], batch_logits[0]
scores, labels = torch.max(torch.sigmoid(logits), dim=1)

# Vẽ box
vis_img = draw_boxes(img_tensor[0].cpu(), boxes.cpu(), labels)
cv2.imwrite("out_det.jpg", vis_img)

# --- Segmentation ---
mask_pred = out["seg"][0,0].cpu()
overlay_segmentation(img_tensor[0].cpu(), mask_pred>0.5, mask_pred, "out_seg.jpg")
