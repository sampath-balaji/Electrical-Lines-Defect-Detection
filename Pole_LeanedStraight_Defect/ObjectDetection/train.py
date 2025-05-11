import os
from ultralytics import YOLO

# Load YOLOv12 model
model = YOLO('yolo12m.pt')

# Device selection
device = 'cuda'  # or 'cpu'

# Define dataset path
dataset_dir = r"/home/line_quality/line_quality/ElectricPoles_StraightLeaned-Defects" # Use complete path
data_yaml = os.path.join(dataset_dir, "data.yaml")

# Train the model
model.train(
    data=data_yaml,
    batch=16,
    imgsz=640,
    patience=0,    # Use a non-zero value to enable early stopping
    epochs=200,
    device=device,
    half=True,
    workers=0,
    optimizer='auto'
)

# Validate and test after training
val_metrics = model.val(split='val')
test_metrics = model.val(split='test')

print("Validation metrics:", val_metrics)
print("Test metrics:", test_metrics)
