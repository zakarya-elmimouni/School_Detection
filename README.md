# School Detection from Satellite Imagery

This repository contains the codebase for the weakly supervised object detection pipeline used to detect school buildings from satellite images, in the context of digital connectivity and infrastructure mapping. The approach relies on a large dataset of auto-labeled images and a smaller manually annotated set (golden data), with segmentation-based labeling, training, and fine-tuning strategies.

---
## 🎥 Demo
![Demo](output_motion1.gif)

## 🏫 School Detection Web App

We provide an interactive **Streamlit application** that lets you detect schools on satellite images using our trained models on USA.

### 1️⃣ Features
- Upload any satellite image (PNG/JPG); it is automatically resized to **500 × 500 px**.
- Select the model you want to use among our different models.
- Get an **annotated image** with a red bounding box if a school is detected.
- Download the processed image in one click.
### 2️⃣ How to run the Application 
1-Check the models in school-detector folder.  
```bash
school-detector/
└─ models/
   ├─ satlas/best.pt
   ├─ yolo/best.pt
   ├─ FasterRcnn/best.pt
   ...
   └─ big_global_model/best.pt
```
2- Run the App: start the Streamlit server by running:
```bash
streamlit run app.py
```

  
##  Project Structure & Key Features

- Automatic data download from Google Static Maps API (requires API key)
- Outlier cleaning (blur, vegetation, sea, desert)
- Bounding box generation using LangSAM segmentation
- Dataset construction
- Fine-tuning with high-quality golden data
- Training, evaluation, and hyperparameter tuning (ECP)

---

##  How to Run the Code

### 1. Install dependencies

run:

```bash
pip install -r requirements.txt
```
### 2. Download satellite images 

⚠️ **We do not share image files directly.**  
Google's usage policy does not allow redistribution of Static Map images.  
To reproduce the dataset:

- Get a [Google Static Maps API key](https://developers.google.com/maps/documentation/maps-static/get-api-key)
- For each country run the code provided in scripts/download_data_from_static_maps_api.

### 2. Clean and generate automatic labels (bounding boxes) 
- Clean outlier images (blurred, vegetation, sea, desert) and generate the bounding boxes with segementation using the codes provided in the folder scripts/cleaning_scripts
### 3. Prepare dataset (Auto-labeled)
 - Build the  dataset with auto-labeled bounding boxes and apply standard augmentations
### 4. Prepare Golden Dataset
-The golden dataset consists of manually annotated labels in YOLO format.
- You will find label files in:

 ```bash
dataset/{country}/manaully_labeled_data/labels
```
To use this dataset:
- Copy only the matching images from data/{country}/satellite/ (same filenames as labels).
- Build the golden dataset for each country

### 7. Prepare Global Datasets
- run the code in scripts/create_global_dataset_with_txt_files.py to create the global golden dataset or the african regional dataset.

### 8. Train and Evaluate YOLO models
Once the datasets are ready you can lunch the training and evaluation.
### 9. Hyperparameters (ECP)

We optimized the training hyperparameters using the [ECP algorithm](https://arxiv.org/abs/2502.04290), a black-box optimization method well-suited for tuning costly deep learning models.

All optimal hyperparameter for the Finetuned models with ECP, are stored in the folder: hyperparamaters.


