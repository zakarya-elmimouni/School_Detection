from ultralytics import YOLO

# Path to your YAML file
DATA_YAML = 'dataset/usa/golden_data/data.yaml'

model = YOLO('runs/detect/results/usa/rslt_yolo26n_auto_labeling/exp/weights/best.pt') # existing model to finetune
# Training

results = model.train(
    data=DATA_YAML,
    epochs=120,
    imgsz=500 ,
    batch=64,
    lr0=0.01,
    lrf=0.001,
    pretrained=True,
    seed=0,
    device=0,
    project='results/usa/rslt_yolo26n_finetune_auto_on_golden',
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
