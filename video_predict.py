from ultralytics import YOLO
import torch
model = YOLO(r"C:\Users\hauhm\Downloads\hole_project\hole_project\runs\train\yolov12l-pothole4\weights\best.pt")
results = model.predict(source=r"C:\Users\hauhm\Downloads\hole_video_2.mp4",
                        save=True,
                        project="runs/predict",
                        name="pothole_demo",
                        conf=0.25,
                        device='0' if torch.cuda.is_available() else 'cpu',
                    
                        show=True)