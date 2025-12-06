import os
import math
import json
import random
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork

SEED = 42
BASE_DIR = Path(".")
IMG_DIRS = [BASE_DIR / f"images{i}" for i in range(7)]  
TRAIN_JSON = BASE_DIR / "training.json"
TEST_JSON  = BASE_DIR / "testing.json"

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
CHECKPOINT = "resnet50_fpn_binary.pth"
PRED_CSV   = "test_predictions_resnet50_fpn.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = (DEVICE == "cuda")
NUM_WORKERS = 0  
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# helper functions
def find_image_path(file_name: str, img_dirs: List[Path] = IMG_DIRS) -> Optional[Path]:
    for d in img_dirs:
        p = d / file_name
        if p.exists():
            return p
    p = BASE_DIR / "images" / file_name
    if p.exists():
        return p
    return None

def robust_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def make_label(entry: Dict[str, Any]) -> int:
    sev_val = robust_float(entry.get("severity", 0))
    return int(sev_val >= 1)


assert TRAIN_JSON.exists(), f"{TRAIN_JSON} nicht gefunden"
with open(TRAIN_JSON, "r") as f:
    tr = json.load(f)

train_items = []
src = tr.get("images", tr) if isinstance(tr, dict) else tr
for d in src:
    fn = d.get("file_name")
    p = find_image_path(fn)
    if p is not None:
        train_items.append({
            "file_name": fn,
            "path": str(p),
            "label": make_label(d),
            "width": d.get("width"),
            "height": d.get("height"),
        })

df = pd.DataFrame(train_items)
print(f"Train Samples: {len(df)} | Positives: {df['label'].sum()} | Negatives: {(1-df['label']).sum()}")

if len(df["label"].unique()) > 1:
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=VAL_SPLIT, random_state=SEED, stratify=df["label"]
    )
else:
    idx = np.arange(len(df))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - VAL_SPLIT))
    train_df = df.iloc[idx[:cut]]
    val_df   = df.iloc[idx[cut:]]

print(f"Train: {len(train_df)}  Val: {len(val_df)}")

# pos_weight 
pos = int((train_df["label"] == 1).sum())
neg = int((train_df["label"] == 0).sum())
pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=DEVICE)
print("pos_weight =", float(pos_weight))

train_tfms = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=0),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=15, border_mode=0, value=0, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])
val_tfms = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=0),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

# Dataset
class AerialWasteBinaryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def _load_image(self, path: str) -> np.ndarray:
        with Image.open(path) as im:
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            elif im.mode == "L":
                im = im.convert("RGB")
            arr = np.array(im)
        return arr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_image(row["path"])
        label = np.array([row["label"]], dtype=np.float32)

        aug = self.transforms(image=img)
        img_t = aug["image"]  

        return img_t, torch.from_numpy(label)

train_ds = AerialWasteBinaryDataset(train_df, transforms=train_tfms)
val_ds   = AerialWasteBinaryDataset(val_df,   transforms=val_tfms)

# Weighted Sampler 
class_counts = train_df["label"].value_counts().to_dict()
weights = [1.0 / class_counts[int(row["label"])] for _, row in train_df.iterrows()]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
# Model
class ResNet50FPNBinary(nn.Module):
    def __init__(self, pretrained: bool = False, fpn_out_channels: int = 256, dropout: float = 0.2):
        super().__init__()
        self.backbone = resnet50(weights=(ResNet50_Weights.IMAGENET1K_V2 if pretrained else None))
        self.backbone.fc = nn.Identity()

        self.feat_extractor = create_feature_extractor(
            self.backbone,
            return_nodes={"layer1": "c2", "layer2": "c3", "layer3": "c4", "layer4": "c5"}
        )
        in_channels_list = [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=fpn_out_channels)

        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(4 * fpn_out_channels, 1)  

    def forward(self, x):
        feats = self.feat_extractor(x)    # dict: c2..c5
        fpn_out = self.fpn(feats)         

        pooled = []
        for key in ["c5", "c4", "c3", "c2"]:
            p = fpn_out[key]
            gap = F.adaptive_avg_pool2d(p, 1).flatten(1)
            pooled.append(gap)
        z = torch.cat(pooled, dim=1)
        z = self.dropout(z)
        return self.head(z)               

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return {"acc": acc, "f1": f1, "auc": auc, "thr": thr}

# Training
model = ResNet50FPNBinary(pretrained=False).to(DEVICE)
def set_backbone_requires_grad(m: ResNet50FPNBinary, req: bool):
    for p in m.backbone.parameters():
        p.requires_grad = req
    for p in m.feat_extractor.parameters():
        p.requires_grad = req

FREEZE_BACKBONE_EPOCHS = 3
set_backbone_requires_grad(model, False)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
best_auc = -1.0

autocast_enabled = (DEVICE == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=autocast_enabled)

for epoch in range(1, EPOCHS + 1):
    if epoch == FREEZE_BACKBONE_EPOCHS + 1:
        set_backbone_requires_grad(model, True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        print("-> Backbone unfrozen.")

    # train
    model.train(True)
    running_loss = 0.0
    y_true_tr, y_prob_tr = [], []

    for imgs, labels in train_loader:
        imgs   = imgs.to(DEVICE, non_blocking=False)
        labels = labels.to(DEVICE, non_blocking=False).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
            logits = model(imgs)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight.to(DEVICE))
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        y_prob_tr.extend(probs.tolist())
        y_true_tr.extend(labels.detach().cpu().numpy().ravel().tolist())

    train_loss = running_loss / max(len(train_loader.dataset), 1)
    train_metrics = compute_metrics(np.array(y_true_tr), np.array(y_prob_tr), thr=0.5)

    # validation
    model.eval()
    y_true_v, y_prob_v = [], []
    v_loss_sum, n_v = 0.0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.to(DEVICE, non_blocking=False)
            labels = labels.to(DEVICE, non_blocking=False).float()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                logits = model(imgs)
                v_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight.to(DEVICE))
            v_loss_sum += v_loss.item() * imgs.size(0)
            n_v += imgs.size(0)
            y_prob_v.extend(torch.sigmoid(logits).cpu().numpy().ravel().tolist())
            y_true_v.extend(labels.cpu().numpy().ravel().tolist())

    val_loss = v_loss_sum / max(n_v, 1)
    val_metrics = compute_metrics(np.array(y_true_v), np.array(y_prob_v), thr=0.5)

    # Mapping: AP@0.5 -> F1@0.5, mIoU(TP) -> ROC-AUC
    ap50_like = val_metrics["f1"]
    miou_like = 0.0 if math.isnan(val_metrics["auc"]) else val_metrics["auc"]
    print(f"Val: AP@0.5={ap50_like:.4f} | mIoU(TP)={miou_like:.4f} | train_loss={train_loss:.4f}")

    if not math.isnan(miou_like) and miou_like > best_auc:
        best_auc = miou_like
        state = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in model.state_dict().items()}
        torch.save(state, CHECKPOINT)
        print(f"Saved BEST checkpoint → {CHECKPOINT} (AUC={best_auc:.4f})")


if TEST_JSON.exists():
    with open(TEST_JSON, "r") as f:
        te = json.load(f)
    test_items = []
    src_te = te.get("images", te) if isinstance(te, dict) else te
    for d in src_te:
        fn = d.get("file_name")
        p = find_image_path(fn)
        if p is not None:
            test_items.append({"file_name": fn, "path": str(p)})
    test_df = pd.DataFrame(test_items)
    class TestDataset(Dataset):
        def __init__(self, df, transforms):
            self.df = df.reset_index(drop=True); self.transforms = transforms
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            with Image.open(row["path"]) as im:
                if im.mode not in ("RGB","L"):
                    im = im.convert("RGB")
                elif im.mode == "L":
                    im = im.convert("RGB")
                arr = np.array(im)
                img_t = val_tfms(image=arr)["image"]
            return img_t, row["file_name"]

    test_loader = DataLoader(TestDataset(test_df, val_tfms), batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    best_model = ResNet50FPNBinary(pretrained=False).to(DEVICE)
    best_model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE), strict=True)
    best_model.eval()

    pred_rows = []
    with torch.no_grad():
        for imgs, fns in test_loader:
            imgs = imgs.to(DEVICE, non_blocking=False)
            logits = best_model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            for fn, p in zip(fns, probs):
                pred_rows.append({"file_name": fn, "waste_proba": float(p), "waste_pred": int(p >= 0.5)})
    pd.DataFrame(pred_rows).to_csv(PRED_CSV, index=False)
    print("Saved:", PRED_CSV)