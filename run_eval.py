"""
run_eval.py — Chạy đánh giá model trên tập Val, xuất per-class IoU / Precision / Recall / F1
Chạy: python run_eval.py
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import json

# ── Cấu hình ──────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
VAL_ROOT    = "./LoveDA_patch/Val"
MODEL_PATH  = "./last_generator.pth"
IMG_SIZE    = 512
NUM_CLASSES = 7
BATCH_SIZE  = 4

CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren", "Forest", "Agricultural"]

print(f"Device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

class PatchDataset(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir  = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.images   = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img  = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, name)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE).astype(np.int32)
        mask[mask == 255] = 1
        mask = np.clip(mask - 1, 0, NUM_CLASSES - 1).astype(np.uint8)
        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out["image"]
            mask = out["mask"].long()
        else:
            img  = torch.tensor(img / 255.0).permute(2,0,1).float()
            mask = torch.tensor(mask).long()
        return img, mask

val_ds     = PatchDataset(VAL_ROOT, transform=val_transform)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
print(f"Val set: {len(val_ds):,} samples")

# ── Load model ────────────────────────────────────────────────────────
G = smp.DeepLabV3Plus(
    encoder_name="resnet50", encoder_weights=None,
    in_channels=3, classes=NUM_CLASSES, activation=None,
).to(DEVICE)
G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
G.eval()
print(f"Loaded model from {MODEL_PATH}")

# ── Tính confusion matrix ─────────────────────────────────────────────
cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

with torch.no_grad():
    for i, (imgs, masks) in enumerate(val_loader):
        imgs  = imgs.to(DEVICE)
        masks = torch.clamp(masks, 0, NUM_CLASSES - 1)
        logits = G(imgs)
        preds  = logits.argmax(dim=1).cpu().numpy()
        tgts   = masks.numpy()
        for p, t in zip(preds, tgts):
            np.add.at(cm, (t.ravel(), p.ravel()), 1)
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(val_loader)}] batches done...")

print("Confusion matrix computed!")

# ── Tính metrics từ confusion matrix ─────────────────────────────────
C = NUM_CLASSES
iou       = np.zeros(C)
precision = np.zeros(C)
recall    = np.zeros(C)
f1        = np.zeros(C)

for c in range(C):
    tp = cm[c, c]
    fp = cm[:, c].sum() - tp
    fn = cm[c, :].sum() - tp

    den_iou = tp + fp + fn
    den_pre = tp + fp
    den_rec = tp + fn

    iou[c]       = tp / den_iou if den_iou > 0 else float("nan")
    precision[c] = tp / den_pre if den_pre > 0 else float("nan")
    recall[c]    = tp / den_rec if den_rec > 0 else float("nan")

    if not (np.isnan(precision[c]) or np.isnan(recall[c])) and (precision[c] + recall[c]) > 0:
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
    else:
        f1[c] = float("nan")

miou           = float(np.nanmean(iou))
mean_precision = float(np.nanmean(precision))
mean_recall    = float(np.nanmean(recall))
mean_f1        = float(np.nanmean(f1))
pixel_acc      = float(np.diag(cm).sum() / cm.sum())

# ── In kết quả ────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  Per-class Metrics on Validation Set")
print("="*65)
print(f"  {'Class':<18} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-"*65)

def fmt(v):
    return f"{v:.4f}" if not np.isnan(v) else "  N/A"

results = {}
for i, name in enumerate(CLASS_NAMES):
    print(f"  [{i}] {name:<15} {fmt(iou[i]):>8} {fmt(precision[i]):>10} {fmt(recall[i]):>8} {fmt(f1[i]):>8}")
    results[name] = {
        "IoU": round(float(iou[i]), 4) if not np.isnan(iou[i]) else None,
        "Precision": round(float(precision[i]), 4) if not np.isnan(precision[i]) else None,
        "Recall": round(float(recall[i]), 4) if not np.isnan(recall[i]) else None,
        "F1": round(float(f1[i]), 4) if not np.isnan(f1[i]) else None,
    }

print("-"*65)
print(f"  {'Mean':<18} {fmt(miou):>8} {fmt(mean_precision):>10} {fmt(mean_recall):>8} {fmt(mean_f1):>8}")
print(f"\n  Overall Pixel Accuracy: {pixel_acc:.4f}")
print("="*65)

# ── Lưu kết quả ra JSON ───────────────────────────────────────────────
output = {
    "per_class": results,
    "mean": {
        "mIoU": round(miou, 4),
        "mean_precision": round(mean_precision, 4),
        "mean_recall": round(mean_recall, 4),
        "mean_f1": round(mean_f1, 4),
        "pixel_accuracy": round(pixel_acc, 4),
    }
}
with open("eval_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print("\nKết quả đã lưu vào eval_results.json")
