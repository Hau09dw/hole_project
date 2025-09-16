from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from pathlib import Path
MODEL_WEIGHTS = r"C:\Users\hauhm\Downloads\hole_project\hole_project\yolov12\ultralytics\cfg\models\v12\yolov12.yaml"  
dataset_path = r"C:\Users\hauhm\Downloads\hole_project\hole_project\data\data.yaml"
def train_model():
    
    model = YOLO(MODEL_WEIGHTS)

    results = model.train(data=dataset_path,
                        epochs=400,
                        imgsz=640,
                        batch=16,
                        patience=20,
                        device='0' if torch.cuda.is_available() else 'cpu',
                        project="runs/train",
                        name="yolov12l-pothole",
                        plots=True)
    return model, results

def evaluate_model(model, results):
    metrics = model.val()
    print(metrics)

    # Lấy đường dẫn từ results hoặc model.trainer
    save_dir = Path(getattr(model.trainer, 'save_dir', 'runs/val/exp'))
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "metrics.txt", "w") as f:
        f.write(str(metrics))
        
def visualize_results(model, dataset="./data/data.yaml"):
    # Trực quan hóa kết quả trên test set
    results = model.val(data=dataset, split="test", plots=True)
    print("Test metrics:", results)

    # Hiển thị 1 số ảnh dự đoán
    test_images = Path("./data/images/test")
    sample_imgs = list(test_images.glob("*.png"))[:5]
    for img in sample_imgs:
        model.predict(img, save=True, project="runs/predict", name="pothole_demo", imgsz=640, conf=0.25)
if __name__ == "__main__":
    model, results = train_model()
    evaluate_model(model, results)
    visualize_results(model)
