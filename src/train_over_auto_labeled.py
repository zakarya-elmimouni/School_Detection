from ultralytics import YOLO

# Path to your YAML file
#DATA_YAML = 'dataset/brazil/fixed_bb_and_manual_data/generated_dataset1/data.yaml'
DATA_YAML = 'dataset/usa/dataset_yolo_auto_labeling/data.yaml'
# Model: YOLO10n
model = YOLO('yolov8n.pt')

# Training
results = model.train(
    data=DATA_YAML,
    epochs=120,
    imgsz=500 ,
    batch=64,
    lr0=0.001,
    lrf=0.01,
    pretrained=True,
    seed=0,
    device=0,
    project='results/usa/rslt_yolo8n_auto_labeling',
    name='exp',
    save=True,
    plots=True,
    patience=10,  # early stopping
    save_period=20,
    verbose=True,
    translate=0.3,
    degrees=15,
    max_det=1,
    #auto_augment=None
)

print(f"? Best model saved at: {results.save_dir}/weights/best.pt")
print(f"? All training plots saved at: {results.save_dir}")