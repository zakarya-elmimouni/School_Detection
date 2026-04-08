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
```
2- Run the App: start the Streamlit server by running:
```bash
streamlit run app.py
```

  
##  Project Structure & Key Features
-Collect school and non school positions from GIGA website and overpass API
-Automatic data download from NAIP Imagery 
- Outlier cleaning (vegetation, sea, desert) (automatic)
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
To reproduce the dataset:

- we provide the dataset of the positions in the folder data/usa
- To download images run the code provided in scripts/download_data.py.

### 2. Clean and generate automatic labels (bounding boxes) 
- Clean outlier images (blurred, vegetation, sea, desert) and generate the bounding boxes with segementation using the codes provided in the folder scripts/cleaning_scripts
### 3. Prepare dataset (Auto-labeled)
 - Build the  dataset with auto-labeled bounding boxes run the code scripts/clean_data.py
### 4. Prepare Golden Dataset
-The golden dataset consists of manually annotated labels in YOLO format.
- You will find label files in:

 ```bash
dataset/{country}/golden/labels
```
To use this dataset:
- Copy only the matching images from data/usa/satellite/ (same filenames as labels).
- Build the golden dataset using the code provided in scripts/build_golden_dataset.py 

### 5. Train and Evaluate models
Once the datasets are ready you can lunch the training and evaluation. Training scripts are available in the folder src, three models are available to train YOLO26n and Satals model and Faster rcnn model. the code are available for training on automatic labeled data and also on golden data, we provide also code for finetuning hyperparameters with ECP for the three models. (make sure to modify whathever is needed to adapt to your case).
### 6. Hyperparameters (ECP)

We optimized the training hyperparameters using the [ECP algorithm](https://arxiv.org/abs/2502.04290), a black-box optimization method well-suited for tuning costly deep learning models.

All optimal hyperparameter for the Finetuned models with ECP, are stored in the folder: hyperparamaters.


