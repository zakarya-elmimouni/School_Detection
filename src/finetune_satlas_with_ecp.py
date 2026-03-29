import os
import sys
import glob
import csv
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou

import satlaspretrain_models as spm

# ==========================================
# Clone ECP
# ==========================================

if not os.path.exists("ECP"):
    os.system("git clone https://github.com/fouratifares/ECP.git")

sys.path.append(os.path.abspath("ECP"))
from optimizers.ECP import ECP

# ==========================================
# CONFIG
# ==========================================

DATA_ROOT = "dataset/usa/golden_data_small_train"

IMG_DIR_TRAIN = f"{DATA_ROOT}/images/train"
LBL_DIR_TRAIN = f"{DATA_ROOT}/labels/train"

IMG_DIR_VAL = f"{DATA_ROOT}/images/val"
LBL_DIR_VAL = f"{DATA_ROOT}/labels/val"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "Aerial_SwinB_SI"

IMG_SIZE = 400
NUM_CLASSES = 1

EPOCHS = 15
PATIENCE = 3

IOU_THRESHOLD = 0.5

os.makedirs("results/usa/satlas_small_train_ecp/models_trials", exist_ok=True)

CSV_LOG = "results/usa/satlas_small_train_ecp/ecp_small_direct_golden_log.csv"

# ==========================================
# DATASET
# ==========================================

def load_yolo_txt(lbl_path, img_w, img_h):

    if not os.path.exists(lbl_path):
        return torch.zeros((0,4)), torch.zeros((0,),dtype=torch.int64)

    boxes=[]
    labels=[]

    with open(lbl_path) as f:
        lines=f.readlines()

    for line in lines:

        cls,cx,cy,w,h = map(float,line.split())

        x1=(cx-w/2)*img_w
        y1=(cy-h/2)*img_h
        x2=(cx+w/2)*img_w
        y2=(cy+h/2)*img_h

        boxes.append([x1,y1,x2,y2])
        labels.append(int(cls)+1)

    if len(boxes)==0:
        return torch.zeros((0,4)),torch.zeros((0,),dtype=torch.int64)

    return torch.tensor(boxes),torch.tensor(labels)


class YoloDataset(Dataset):

    def __init__(self,img_dir,lbl_dir):

        self.img_paths=sorted(glob.glob(os.path.join(img_dir,"*.*")))
        self.lbl_dir=lbl_dir

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,idx):

        img_path=self.img_paths[idx]

        img=Image.open(img_path).convert("RGB")
        img=img.resize((IMG_SIZE,IMG_SIZE))

        img=torch.tensor(np.array(img)/255.).permute(2,0,1).float()

        lbl_path=os.path.join(self.lbl_dir,Path(img_path).stem+".txt")

        boxes,labels=load_yolo_txt(lbl_path,IMG_SIZE,IMG_SIZE)

        return img,{"boxes":boxes,"labels":labels}


def collate_fn(batch):
    return tuple(zip(*batch))


# ==========================================
# MODEL
# ==========================================

def build_model():

    weights = spm.Weights()

    model = weights.get_pretrained_model(
        MODEL_ID,
        fpn=True,
        head=spm.Head.DETECT,
        num_categories=NUM_CLASSES+1,
        device=DEVICE
    )

    return model


# ==========================================
# METRIC
# ==========================================

def total_loss_from_model_output(out):
    """
    Improved version with optional debug
    """
    if isinstance(out, dict):
        print("c'est ca ce qu'il faut regarder",out)
        total_loss = 0.0
        loss_count = 0
        for key, value in out.items():
            if 'loss' in key.lower() and torch.is_tensor(value):
                total_loss += value
                loss_count += 1
                # print(f"Loss {key}: {value.item():.4f}")  # Uncomment for debug
        if loss_count == 0:
            for key, value in out.items():
                if torch.is_tensor(value):
                    total_loss += value
                    loss_count += 1
        return total_loss
    elif isinstance(out, (list, tuple)):
        if len(out) >= 1 and isinstance(out[0], dict):
            return total_loss_from_model_output(out[0])
        else:
            total_loss = 0.0
            for item in out:
                if torch.is_tensor(item):
                    total_loss += item
            return total_loss
    elif torch.is_tensor(out):
        return out
    else:
        print(f"WARNING: Unrecognized output format: {type(out)}")
        return torch.tensor(0.0, device=DEVICE)


def compute_f1_50(model,loader,score_thresh):

    TP,FP,FN=0,0,0

    model.eval()

    with torch.no_grad():

        for images,targets in loader:

            images=torch.stack(images).to(DEVICE)

            outputs=model(images)[0]

            for output,target in zip(outputs,targets):

                pred_boxes=output["boxes"].cpu()
                pred_scores=output["scores"].cpu()

                gt_boxes=target["boxes"]

                keep=pred_scores>=score_thresh
                pred_boxes=pred_boxes[keep]

                if len(gt_boxes)==0:

                    FP+=len(pred_boxes)
                    continue

                if len(pred_boxes)==0:

                    FN+=len(gt_boxes)
                    continue

                ious=box_iou(pred_boxes,gt_boxes)

                matched=set()

                for i in range(len(pred_boxes)):

                    max_iou,idx=torch.max(ious[i],dim=0)

                    if max_iou>=IOU_THRESHOLD and idx.item() not in matched:

                        TP+=1
                        matched.add(idx.item())

                    else:

                        FP+=1

                FN+=len(gt_boxes)-len(matched)

    precision=TP/(TP+FP+1e-6)
    recall=TP/(TP+FN+1e-6)

    f1=2*precision*recall/(precision+recall+1e-6)

    return f1


# ==========================================
# ECP OBJECTIVE
# ==========================================

class SatlasObjective:

    def __init__(self):

        self.bounds=np.array([

            [1e-5,5e-4],
            [1e-6,1e-3],
            [0.01,0.4],
            [0.4,0.8]


        ])

        self.dimensions=self.bounds.shape[0]

        train_ds=YoloDataset(IMG_DIR_TRAIN,LBL_DIR_TRAIN)
        val_ds=YoloDataset(IMG_DIR_VAL,LBL_DIR_VAL)

        self.train_ds=train_ds
        self.val_ds=val_ds

        self.global_best=0
        self.global_model_path="results/usa/satlas_tiny_train_ecp/best_global_model.pt"

        with open(CSV_LOG,"w",newline="") as f:

            writer=csv.writer(f)
            writer.writerow([
                "F1",
                "lr",
                "weight_decay",
                "conf_thresh",
                "nms_thresh",
            ])

    def __call__(self,x):

        lr,wd,conf,nms=x

        bs=4
        warmup=2

        trial_name=f"lr{lr:.1e}_wd{wd:.1e}_conf{conf}_nms{nms}"

        trial_model_path=f"results/usa/satlas_tiny_train_ecp/models_trials/{trial_name}.pt"

        train_loader=DataLoader(
            self.train_ds,
            batch_size=bs,
            shuffle=True,
            collate_fn=collate_fn
        )

        val_loader=DataLoader(
            self.val_ds,
            batch_size=bs,
            shuffle=False,
            collate_fn=collate_fn
        )

        model=build_model().to(DEVICE)

        optimizer=torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd
        )

        best_epoch_f1=0
        patience_counter=0

        for epoch in range(EPOCHS):

            model.train()

            for images,targets in train_loader:

                images=torch.stack(images).to(DEVICE)

                targets=[{k:v.to(DEVICE) for k,v in t.items()} for t in targets]

                out=model(images,targets)

#                loss=sum(v for v in out.values())
                loss = total_loss_from_model_output(out)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            f1=compute_f1_50(model,val_loader,conf)

            print(f"Epoch {epoch+1} F1={f1:.4f}")

            if f1>best_epoch_f1:

                best_epoch_f1=f1
                patience_counter=0

#                torch.save(model.state_dict(),trial_model_path)

            else:

                patience_counter+=1

            if patience_counter>=PATIENCE:

                print("Early stopping")
                break

        # global best
        if best_epoch_f1>self.global_best:

            self.global_best=best_epoch_f1

            torch.save(
                model.state_dict(),
                self.global_model_path
            )

            print("NEW GLOBAL BEST MODEL")

        # CSV log
        with open(CSV_LOG,"a",newline="") as f:

            writer=csv.writer(f)

            writer.writerow([
                best_epoch_f1,
                lr,
                wd,
                conf,
                nms
            ])

        return best_epoch_f1


# ==========================================
# MAIN
# ==========================================

if __name__=="__main__":

    objective=SatlasObjective()

    n_evals=20

    points,values,eps=ECP(objective,n=n_evals)

    best_idx=np.argmax(values)

    print("\nOptimization finished")

    print("Best params:",points[best_idx])
    print("Best F1:",values[best_idx])