import io
import os
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
import streamlit as st
from ultralytics import YOLO
from typing import Optional, Tuple

# Import pour Faster R-CNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Import pour Satlas (Assurez-vous que satlaspretrain_models est installé)
try:
    import satlaspretrain_models as spm
except ImportError:
    st.error("Le package 'satlaspretrain_models' est manquant. Installez-le pour utiliser le modèle Satlas.")

# ------------------------------
# Configuration
# ------------------------------
st.set_page_config(page_title="School Detection USA", layout="wide")

MODELS_USA = {
    "YOLOv26 Nano (USA)": {"path": "models/usa/best_yolo26n.pt", "type": "yolo"},
    "Faster R-CNN (USA)": {"path": "models/usa/best_faster_rcnn.pt", "type": "rcnn"},
    "Satlas Swin-B (USA)": {"path": "models/usa/best_satlas.pt", "type": "satlas"}
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 500 # Taille standardisée pour l'inférence
NUM_CLASSES = 1 # School uniquement

# ------------------------------
# Chargeurs de Modèles (Cached)
# ------------------------------
@st.cache_resource
def load_yolo(path):
    return YOLO(path)

@st.cache_resource
def load_rcnn(path):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_satlas(path):
    weights = spm.Weights()
    # On utilise Aerial_SwinB_SI comme dans votre script de prédiction
    model = weights.get_pretrained_model(
        "Aerial_SwinB_SI", fpn=True, head=spm.Head.DETECT, num_categories=NUM_CLASSES + 1, device=DEVICE
    )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ------------------------------
# Fonctions de Prétraitement & Dessin
# ------------------------------
def preprocess_image(pil_img: Image.Image):
    """Redimensionne et convertit l'image en tenseur pour R-CNN et Satlas."""
    img_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    
    # Normalisation standard (ImageNet) souvent utilisée par R-CNN et Satlas
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
    return img_resized, tensor

def draw_boxes(img_rgb: np.ndarray, boxes: np.ndarray, scores: np.ndarray, threshold: float):
    annotated = img_rgb.copy()
    for i, box in enumerate(boxes):
        if scores[i] >= threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2) # Bleu pour changer un peu
            label = f"School: {scores[i]:.2f}"
            cv2.putText(annotated, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return annotated

# ------------------------------
# Inférence par Architecture
# ------------------------------
def run_inference(model_info, pil_img, conf_th):
    m_type = model_info["type"]
    m_path = model_info["path"]
    
    if not os.path.exists(m_path):
        st.error(f"Fichier de poids introuvable : {m_path}")
        return None, False, [], []

    # 1. Inférence YOLO
    if m_type == "yolo":
        model = load_yolo(m_path)
        img_input = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
        results = model(img_input, conf=conf_th)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        
    # 2. Inférence Faster R-CNN
    elif m_type == "rcnn":
        model = load_rcnn(m_path)
        img_resized, tensor = preprocess_image(pil_img)
        img_input = np.array(img_resized)
        with torch.no_grad():
            output = model(tensor)[0]
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

    # 3. Inférence Satlas
    elif m_type == "satlas":
        model = load_satlas(m_path)
        img_resized, tensor = preprocess_image(pil_img)
        img_input = np.array(img_resized)
        with torch.no_grad():
            preds = model(tensor)
            if isinstance(preds, tuple): preds = preds[0]
            # Satlas renvoie souvent une liste de dictionnaires pour le batch
            pred = preds[0] 
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

    # Filtrage et Dessin
    keep = scores >= conf_th
    final_boxes = boxes[keep]
    final_scores = scores[keep]
    
    has_det = len(final_boxes) > 0
    annotated_img = draw_boxes(img_input, final_boxes, final_scores, conf_th)
    
    return annotated_img, has_det, final_boxes, final_scores

# ------------------------------
# Interface UI
# ------------------------------
st.title("🇺🇸 School Detection - USA Special Edition")
st.markdown("Cette application compare les performances de **YOLOv8**, **Faster R-CNN** et **Satlas Pretrain** sur le territoire américain.")

with st.sidebar:
    st.header("Paramètres USA")
    model_name = st.selectbox("Choisir l'architecture", list(MODELS_USA.keys()))
    conf_th = st.slider("Seuil de confiance", 0.01, 0.95, 0.25, 0.05)
    st.info(f"Mode : {MODELS_USA[model_name]['type'].upper()}")

uploaded_file = st.file_uploader("Charger une image satellite", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Entrée (Originale)")
        st.image(img, use_container_width=True)
    
    with col2:
        st.subheader("Détection")
        if st.button("Lancer l'analyse 🚀"):
            with st.spinner(f"Calcul en cours avec {model_name}..."):
                result_img, found, b, s = run_inference(MODELS_USA[model_name], img, conf_th)
                
                if result_img is not None:
                    st.image(result_img, use_container_width=True)
                    if found:
                        st.success(f"✅ {len(b)} école(s) détectée(s) !")
                    else:
                        st.warning("⚠️ Aucune école détectée avec ce seuil.")
                    
                    # Bouton de téléchargement
                    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                    st.download_button(
                        label="Télécharger le résultat",
                        data=buffer.tobytes(),
                        file_name="detection_usa.png",
                        mime="image/png"
                    )