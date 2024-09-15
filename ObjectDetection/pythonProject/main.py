from ultralytics import YOLO

model = YOLO('yolo8n.pt')
results = model('images/img.png', show=True)