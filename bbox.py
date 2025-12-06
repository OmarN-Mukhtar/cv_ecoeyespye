import os
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import torchvision
from torchvision.ops import box_iou
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

SEED = 42
BASE_DIR = Path(".")
PATCH_DIR = BASE_DIR / "patches"
PATCHES_JSON = BASE_DIR / "patches_bbox.json"
IMG_DIRS = [BASE_DIR / f"images{i}" for i in range(7)] + [BASE_DIR / "images"]
CHECKPOINT = "resnet50_bbox_best.pth"
EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
NUM_WORKERS = 0
PRINT_FREQ = 50
SCORE_THRESH = 0.05  
if hasattr(torch.backends, 'mps') :
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

print(f"Using device: {DEVICE}")
print(f"PATCH_DIR exists: {PATCH_DIR.exists()}  | PATCHES_JSON exists: {PATCHES_JSON.exists()}")

# Helper functions

def find_image_path(file_name: str, patch_dir=PATCH_DIR, img_dirs=IMG_DIRS):
    """Bevorzuge 'patches/'. Fällt danach auf images*, images zurück."""
    if patch_dir is not None:
        p = Path(patch_dir) / file_name
        if p.exists():
            return p
    for d in img_dirs:
        p = Path(d) / file_name
        if p.exists():
            return p
    return None


def coco_xywh_to_xyxy(b: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = b
    return (x, y, x + w, y + h)


def clip_box(xyxy, img_w, img_h):
    x1, y1, x2, y2 = xyxy
    return (
        max(0.0, min(float(x1), img_w - 1.0)),
        max(0.0, min(float(y1), img_h - 1.0)),
        max(0.0, min(float(x2), img_w - 1.0)),
        max(0.0, min(float(y2), img_h - 1.0)),
    )

class PatchDetDataset(Dataset):
    """
    Detection-Dataset aus patches_bbox.json.

    Unterstützt einen einzelnen Ziel-Klasse ("waste"=1). Negative Beispiele haben leere bbox-Liste.
    Gibt (image_tensor, target_dict) zurück. target_dict enthält Keys:
      - boxes (FloatTensor[N,4] in xyxy)
      - labels (LongTensor[N])
      - image_id (Tensor[1])
      - iscrowd (Tensor[N])
      - area (Tensor[N])
    """

    def __init__(self, items: List[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def _safe_open(self, path: str) -> Image.Image:
        im = Image.open(path)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        elif im.mode == "L":
            im = im.convert("RGB")
        return im

    def __getitem__(self, idx):
        it = self.items[idx]
        p = find_image_path(it["file_name"])
        assert p is not None, f"Image not found for {it['file_name']}"

        img = self._safe_open(p)
        w, h = img.size
        img_t = TF.to_tensor(img)  

        b = it.get("bbox", []) or []
        label = int(it.get("label", 0))

        if label == 1 and len(b) == 4:
            x1, y1, x2, y2 = coco_xywh_to_xyxy(b)
            x1, y1, x2, y2 = clip_box((x1, y1, x2, y2), w, h)
            boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.int64)
            iscrowd = torch.zeros((1,), dtype=torch.int64)
            area = torch.tensor([(x2 - x1) * (y2 - y1)], dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([int(it.get("image_id", idx))], dtype=torch.int64),
            "iscrowd": iscrowd,
            "area": area,
            "file_name": it["file_name"],  
        }
        return img_t, target


def collate_fn(batch):
    imgs, targs = list(zip(*batch))
    return list(imgs), list(targs)


assert PATCHES_JSON.exists(), f"patches_bbox.json nicht gefunden unter {PATCHES_JSON}"
with open(PATCHES_JSON, "r") as f:
    raw = json.load(f)

if isinstance(raw, dict) and "patches" in raw:
    items = raw["patches"]
elif isinstance(raw, list):
    items = raw
else:
    for k in ("items", "images"):
        if isinstance(raw, dict) and k in raw:
            items = raw[k]
            break
    else:
        raise ValueError("Unerwartetes Format in patches_bbox.json")

items = [it for it in items if find_image_path(it.get("file_name", "")) is not None]

pos = [it for it in items if int(it.get("label", 0)) == 1]
neg = [it for it in items if int(it.get("label", 0)) == 0]

n_val_pos = max(1, int(len(pos) * VAL_SPLIT))
n_val_neg = max(1, int(len(neg) * VAL_SPLIT))

train_items = pos[n_val_pos:] + neg[n_val_neg:]
val_items = pos[:n_val_pos] + neg[:n_val_neg]

rng = np.random.default_rng(SEED)
rng.shuffle(train_items)
rng.shuffle(val_items)

print(f"Train: {len(train_items)} | Val: {len(val_items)} | Pos(train)={sum(int(i.get('label',0)) for i in train_items)} | Pos(val)={sum(int(i.get('label',0)) for i in val_items)}")

train_ds = PatchDetDataset(train_items)
val_ds = PatchDetDataset(val_items)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)


# Model
model = fasterrcnn_resnet50_fpn(pretrained=True)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)

model.to(DEVICE)

# Optimizer 
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# Metrics (AP@0.5 and mIoU)

def compute_iou_mat(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)
    pb = torch.as_tensor(pred_boxes, dtype=torch.float32)
    gb = torch.as_tensor(gt_boxes, dtype=torch.float32)
    iou = box_iou(pb, gb).cpu().numpy()
    return iou


def eval_ap50_single_class(all_preds: Dict[str, Any], all_gts: Dict[str, Any]) -> Tuple[float, float]:
    """
    Compute AP@0.5 for a single class across the dataset and mean IoU of matched TPs.

    all_preds: {"boxes": Nx4 xyxy, "scores": N}
    all_gts: {"boxes": Mx4 xyxy}
    """
    records = [] 

    n_positives = 0
    for fn, gt in all_gts.items():
        gt_boxes = gt["boxes"]
        n_positives += gt_boxes.shape[0]

    for fn, pred in all_preds.items():
        boxes_p = pred["boxes"]
        scores = pred["scores"]
        order = np.argsort(-scores)
        boxes_p = boxes_p[order]
        scores = scores[order]

        gt_boxes = all_gts.get(fn, {"boxes": np.zeros((0,4), dtype=np.float32)})["boxes"]
        matched = np.zeros((gt_boxes.shape[0],), dtype=bool)
        iou_mat = compute_iou_mat(boxes_p, gt_boxes)

        for i, (b, s) in enumerate(zip(boxes_p, scores)):
            if gt_boxes.shape[0] == 0:
                records.append((fn, float(s), 0, 0.0))
                continue
            best_gt = int(np.argmax(iou_mat[i]))
            best_iou = float(iou_mat[i, best_gt])
            if best_iou >= 0.5 and not matched[best_gt]:
                matched[best_gt] = True
                records.append((fn, float(s), 1, best_iou))
            else:
                records.append((fn, float(s), 0, best_iou))

    if n_positives == 0:
        return 0.0, 0.0

    # Sort all predictions by score desc
    records.sort(key=lambda x: -x[1])
    tps = np.array([r[2] for r in records], dtype=np.float32)
    fps = 1.0 - tps
    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    recalls = cum_tp / max(n_positives, 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

    mprec = np.maximum.accumulate(precisions[::-1])[::-1]
    ap = float(np.trapz(mprec, recalls))

    tp_ious = [r[3] for r in records if r[2] == 1]
    miou = float(np.mean(tp_ious)) if len(tp_ious) > 0 else 0.0
    return ap, miou


# Training & Validation

def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(loader, 1):
        images = [img.to(DEVICE) for img in images]
        t_dev = []
        for t in targets:
            t = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in t.items() if k != "file_name"}
            t_dev.append(t)

        loss_dict = model(images, t_dev)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += float(losses.detach().cpu())
        if i % PRINT_FREQ == 0:
            print(f"[Epoch {epoch}] step {i}/{len(loader)} loss={running_loss/i:.4f}")
    return running_loss / max(len(loader), 1)


def validate(model, loader):
    model.eval()
    all_preds: Dict[str, Dict[str, np.ndarray]] = {}
    all_gts: Dict[str, Dict[str, np.ndarray]] = {}

    with torch.no_grad():
        for images, targets in loader:
            images_dev = [img.to(DEVICE) for img in images]
            outputs = model(images_dev)

            for img, targ, out in zip(images, targets, outputs):
                fn = targ["file_name"]
                # GT
                gt_boxes = targ["boxes"].cpu().numpy().astype(np.float32)
                all_gts[fn] = {"boxes": gt_boxes}
                # Predictions (apply a score threshold but keep fairly low)
                keep = out["scores"].cpu().numpy() >= SCORE_THRESH
                pred_boxes = out["boxes"].cpu().numpy()[keep].astype(np.float32)
                pred_scores = out["scores"].cpu().numpy()[keep].astype(np.float32)
                all_preds[fn] = {"boxes": pred_boxes, "scores": pred_scores}

    ap50, miou = eval_ap50_single_class(all_preds, all_gts)
    return ap50, miou


best_ap = -1.0
for epoch in range(1, EPOCHS + 1):
    print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
    tr_loss = train_one_epoch(model, train_loader, optimizer, epoch)
    ap50, miou = validate(model, val_loader)
    print(f"Val: AP@0.5={ap50:.4f} | mIoU(TP)={miou:.4f} | train_loss={tr_loss:.4f}")

    if ap50 > best_ap:
        best_ap = ap50
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"Saved BEST checkpoint → {CHECKPOINT} (AP@0.5={best_ap:.4f})")

    lr_scheduler.step()

# Final evaluation on val with best checkpoint
print("\nLoading best checkpoint for final eval…")
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE)
model.eval()
final_ap50, final_miou = validate(model, val_loader)
print(f"FINAL Val: AP@0.5={final_ap50:.4f} | mIoU(TP)={final_miou:.4f}")

@torch.no_grad()
def dump_val_predictions_csv(out_csv="val_predictions_det.csv"):
    rows = []
    for images, targets in val_loader:
        images_dev = [img.to(DEVICE) for img in images]
        outputs = model(images_dev)
        for targ, out in zip(targets, outputs):
            fn = targ["file_name"]
            boxes = out["boxes"].cpu().numpy().astype(float)
            scores = out["scores"].cpu().numpy().astype(float)
            for b, s in zip(boxes, scores):
                x1, y1, x2, y2 = b.tolist()
                rows.append({
                    "file_name": fn,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "score": s
                })
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)

# dump_val_predictions_csv()
