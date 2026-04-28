"""
segGAN_improved.py  —  DeepLabV3+ ResNet50 Generator + PatchGAN Discriminator
==============================================================================
Kiến trúc:
  - Generator : DeepLabV3+ với encoder ResNet50 pretrained ImageNet (từ smp)
  - Discriminator: PatchGAN 70×70 với Spectral Normalization

Cải tiến so với segGAN.ipynb gốc:
  1. Fix NUM_CLASSES = 7
  2. DeepLabV3+ ResNet50 pretrained (thay UNet tự build)
  3. Augmentation mạnh với albumentations (chuẩn LoveDA repo)
  4. Normalize đúng ImageNet mean/std
  5. CrossEntropyLoss với class_weight (median frequency)
  6. CosineAnnealingLR scheduler
  7. Metric mIoU chuẩn LoveDA thay vì chỉ val_loss

Cài đặt (một lần):
  pip install segmentation-models-pytorch albumentations
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 1 — Imports                                               ║
# ╚══════════════════════════════════════════════════════════════════╝
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.spectral_norm as spectral_norm

import segmentation_models_pytorch as smp   # DeepLabV3+

import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 2 — Cấu hình                                             ║
# ╚══════════════════════════════════════════════════════════════════╝
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

TRAIN_ROOT  = "./LoveDA_patch/Train"
VAL_ROOT    = "./LoveDA_patch/Val"
TEST_ROOT   = "./Test/Test"

IMG_SIZE    = 512          # DeepLabV3+ hoạt động tốt hơn ở 512 (ResNet50 stride 16)
BATCH_SIZE  = 4            # Giảm xuống do model lớn hơn
EPOCHS      = 100
NUM_CLASSES = 7            # Background, Building, Road, Water, Barren, Forest, Agricultural

LR_G        = 1e-4
LR_D        = LR_G / 4
LAMBDA_ADV  = 0.01         # Trọng số adversarial loss

# LoveDA official color map
COLOR_MAP = np.array([
    [255, 255, 255],   # 0: Background
    [255,   0,   0],   # 1: Building
    [255, 255,   0],   # 2: Road
    [  0,   0, 255],   # 3: Water
    [159, 129, 183],   # 4: Barren
    [  0, 255,   0],   # 5: Forest
    [255, 195, 128],   # 6: Agricultural
], dtype=np.uint8)

CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren", "Forest", "Agricultural"]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 3 — Augmentation & Dataset                                ║
# ╚══════════════════════════════════════════════════════════════════╝
train_transform = A.Compose([
    A.RandomCrop(IMG_SIZE, IMG_SIZE),

    # ── Geometric (giữ nguyên, phù hợp ảnh vệ tinh top-down) ──────
    A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomRotate90(p=1.0),
        A.Transpose(p=1.0),
    ], p=0.75),

    # ── Color / texture ────────────────────────────────────────────
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    ], p=0.5),

    # ── Thêm mới: tương phản cục bộ (rất hiệu quả cho ảnh vệ tinh) ─
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),

    # ── Thêm mới: noise cảm biến vệ tinh ──────────────────────────
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

    # ── Thêm mới: Cutout — regularization, tránh overfitting ───────
    A.CoarseDropout(
        max_holes=8, max_height=32, max_width=32,
        min_holes=1, min_height=8,  min_width=8,
        fill_value=0, p=0.3,
    ),

    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


class PatchDataset(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir   = os.path.join(root, "images")
        self.mask_dir  = os.path.join(root, "masks")
        self.images    = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img  = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, name)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE).astype(np.int32)

        # Mapping nhãn LoveDA: 1-7 → 0-6, ignore (0 và 255) → 0
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


class LoveDATestDataset(Dataset):
    def __init__(self, root):
        self.images = []
        for area in ["Rural", "Urban"]:
            d = os.path.join(root, area, "images_png")
            for f in sorted(os.listdir(d)):
                self.images.append(os.path.join(d, f))

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img  = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return img, path


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 4 — DataLoaders                                           ║
# ╚══════════════════════════════════════════════════════════════════╝
train_ds = PatchDataset(TRAIN_ROOT, transform=train_transform)
val_ds   = PatchDataset(VAL_ROOT,   transform=val_transform)
test_ds  = LoveDATestDataset(TEST_ROOT)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=1)

print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 5 — Class Weights (median frequency balancing)            ║
# ╚══════════════════════════════════════════════════════════════════╝
def compute_class_weights(dataset, num_classes):
    print("Đang tính class weights từ tập Train...")
    freq = np.zeros(num_classes, dtype=np.float64)
    for _, mask in tqdm(dataset, desc="Counting pixels"):
        m = mask.numpy() if hasattr(mask, "numpy") else np.array(mask)
        for c in range(num_classes):
            freq[c] += (m == c).sum()
    freq_norm = freq / freq.sum()
    med = np.median(freq_norm[freq_norm > 0])
    w   = np.where(freq_norm > 0, med / freq_norm, 0.0)
    w   = w / w.max()  # scale về [0, 1] rồi nhân num_classes
    w   = w / w.sum() * num_classes
    print(f"Class weights: {np.round(w, 3)}")
    print(f"  {'  '.join(f'{n}={v:.2f}' for n,v in zip(CLASS_NAMES, w))}")
    return torch.tensor(w, dtype=torch.float32).to(device)

class_weights = compute_class_weights(train_ds, NUM_CLASSES)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 6 — Model: DeepLabV3+ ResNet50 Generator                  ║
# ╚══════════════════════════════════════════════════════════════════╝
#
#  smp.DeepLabV3Plus:
#    - encoder   : ResNet50 pretrained ImageNet (frozen stages đầu)
#    - decoder   : ASPP (Atrous Spatial Pyramid Pooling) + low-level features
#    - output    : logits shape [N, NUM_CLASSES, H, W]

G = smp.DeepLabV3Plus(
    encoder_name        = "resnet50",
    encoder_weights     = "imagenet",   # Pretrained ImageNet — tăng mạnh hiệu năng
    in_channels         = 3,
    classes             = NUM_CLASSES,
    activation          = None,         # Raw logits — để tương thích với CE loss
).to(device)

num_params_G = sum(p.numel() for p in G.parameters()) / 1e6
print(f"Generator (DeepLabV3+ ResNet50): {num_params_G:.1f}M params")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 7 — PatchGAN Discriminator (70×70)                        ║
# ╚══════════════════════════════════════════════════════════════════╝
#
#  Input: [image (3ch) | softmax_mask (7ch)] = 10 channels
#  Output: patch map [N, 1, H', W'] — real/fake score từng patch

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3 + NUM_CLASSES):
        super().__init__()
        def block(ic, oc, stride=2):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(ic, oc, 4, stride=stride, padding=1)),
                nn.BatchNorm2d(oc) if oc != 64 else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.model = nn.Sequential(
            block(in_ch, 64,  stride=2),   # No BN on first layer
            block(64,    128, stride=2),
            block(128,   256, stride=2),
            block(256,   512, stride=1),   # stride=1 ở đây để giữ receptive field
            spectral_norm(nn.Conv2d(512, 1, 4, stride=1, padding=1)),
        )

    def forward(self, x):
        return self.model(x)

D = PatchDiscriminator(in_ch=3 + NUM_CLASSES).to(device)
num_params_D = sum(p.numel() for p in D.parameters()) / 1e6
print(f"Discriminator (PatchGAN 70x70): {num_params_D:.1f}M params")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 8 — Loss Functions                                        ║
# ╚══════════════════════════════════════════════════════════════════╝
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs  = torch.softmax(logits, dim=1)
        onehot = F.one_hot(targets, logits.shape[1]).permute(0,3,1,2).float()
        inter  = (probs * onehot).sum(dim=(2,3))
        union  = probs.sum(dim=(2,3)) + onehot.sum(dim=(2,3))
        dice   = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

gan_loss  = nn.BCEWithLogitsLoss()
ce_loss   = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
dice_loss = DiceLoss()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 9 — Optimizer & Scheduler                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
# Dùng lr khác nhau cho encoder (pretrained) và decoder (train từ đầu)
opt_G = torch.optim.AdamW([
    {"params": G.encoder.parameters(), "lr": LR_G * 0.1},   # Fine-tune encoder nhẹ
    {"params": G.decoder.parameters(), "lr": LR_G},
    {"params": G.segmentation_head.parameters(), "lr": LR_G},
], betas=(0.9, 0.999), weight_decay=1e-4)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))

# CosineAnnealingLR — giảm LR mượt từ đỉnh xuống eta_min
sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=EPOCHS, eta_min=1e-6)
sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=EPOCHS, eta_min=1e-7)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 10 — mIoU Metric & Validation                             ║
# ╚══════════════════════════════════════════════════════════════════╝
def compute_miou(logits, targets, num_classes):
    """Tính mIoU per-class rồi lấy trung bình (bỏ qua class không xuất hiện)."""
    preds = logits.argmax(dim=1).cpu().numpy()
    tgts  = targets.cpu().numpy()
    ious  = []
    for c in range(num_classes):
        pred_c   = preds == c
        target_c = tgts  == c
        inter    = (pred_c & target_c).sum()
        union    = (pred_c | target_c).sum()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0


def validate():
    G.eval()
    total_loss, total_miou = 0.0, 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs  = imgs.to(device)
            masks = torch.clamp(masks.to(device), 0, NUM_CLASSES - 1)
            logits = G(imgs)
            loss   = ce_loss(logits, masks) + dice_loss(logits, masks)
            miou   = compute_miou(logits, masks, NUM_CLASSES)
            total_loss += loss.item()
            total_miou += miou
    n = len(val_loader)
    return total_loss / n, total_miou / n



# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 12 — Training Loop (AMP enabled)                          ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "="*65)
print("  GAN Satellite Segmentation  |  DeepLabV3+ ResNet50 + PatchGAN")
print("="*65)

# AMP scaler — tự động chuyển float16/float32, tiết kiệm ~40% VRAM
scaler_G = torch.cuda.amp.GradScaler()
scaler_D = torch.cuda.amp.GradScaler()

history  = {"g_loss": [], "d_loss": [], "val_loss": [], "val_miou": []}
best_miou = 0.0

for epoch in range(EPOCHS):
    G.train(); D.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{EPOCHS}")
    ep_g, ep_d = 0.0, 0.0

    for imgs, masks in loop:
        imgs  = imgs.to(device)
        masks = torch.clamp(masks.to(device), 0, NUM_CLASSES - 1)

        # ── 1. Train Generator ─────────────────────────────────────
        with torch.cuda.amp.autocast():
            logits    = G(imgs)                             # [N, 7, H, W]
            soft_pred = torch.softmax(logits, dim=1)        # [N, 7, H, W]
            fake_in   = torch.cat([imgs, soft_pred], dim=1) # [N, 10, H, W]

            d_fake = D(fake_in)
            g_adv  = gan_loss(d_fake, torch.ones_like(d_fake))
            g_ce   = ce_loss(logits, masks)
            g_dice = dice_loss(logits, masks)
            g_loss = LAMBDA_ADV * g_adv + g_ce + g_dice

        opt_G.zero_grad()
        scaler_G.scale(g_loss).backward()
        scaler_G.unscale_(opt_G)                            # unscale trước khi clip
        nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        scaler_G.step(opt_G)
        scaler_G.update()

        # ── 2. Train Discriminator ──────────────────────────────────
        with torch.cuda.amp.autocast():
            real_onehot = F.one_hot(masks, NUM_CLASSES).permute(0,3,1,2).float()
            real_in  = torch.cat([imgs, real_onehot], dim=1)
            fake_in2 = torch.cat([imgs, soft_pred.detach()], dim=1)

            d_real = D(real_in)
            d_fake = D(fake_in2)
            d_loss = (
                gan_loss(d_real, torch.ones_like(d_real) * 0.9) +  # label smoothing
                gan_loss(d_fake, torch.zeros_like(d_fake))
            ) * 0.5

        # Chỉ update D khi còn yếu
        if d_loss.item() > 0.3:
            opt_D.zero_grad()
            scaler_D.scale(d_loss).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

        ep_g += g_loss.item(); ep_d += d_loss.item()
        loop.set_postfix(G=f"{g_loss.item():.4f}", D=f"{d_loss.item():.4f}")

    # ── Schedulers step ────────────────────────────────────────────
    sched_G.step(); sched_D.step()

    # ── Validation ─────────────────────────────────────────────────
    val_loss, val_miou = validate()
    lr_now = sched_G.get_last_lr()[0]
    n = len(train_loader)

    print(f"\n  G_loss={ep_g/n:.4f} | D_loss={ep_d/n:.4f} | "
          f"val_loss={val_loss:.4f} | val_mIoU={val_miou:.4f} | lr={lr_now:.2e}")

    history["g_loss"].append(ep_g / n)
    history["d_loss"].append(ep_d / n)
    history["val_loss"].append(val_loss)
    history["val_miou"].append(val_miou)

    # ── Lưu best model (cả G và D) khi val_mIoU cao nhất ─────────────────
    if val_miou > best_miou:
        best_miou = val_miou
        torch.save(G.state_dict(), "best_generator.pth")
        torch.save(D.state_dict(), "best_discriminator.pth")
        print(f"  💾 Best model saved  |  mIoU = {best_miou:.4f}")

    # ── Lưu last checkpoint sau mỗi epoch ──────────────────────────────
    torch.save(G.state_dict(), "last_generator.pth")
    torch.save(D.state_dict(), "last_discriminator.pth")

print(f"\n✅ Huấn luyện hoàn tất!  Best mIoU: {best_miou:.4f}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 13 — Plot Training History                                ║
# ╚══════════════════════════════════════════════════════════════════╝
def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["g_loss"], label="G loss"); axes[0].set_title("Generator Loss")
    axes[1].plot(history["d_loss"], label="D loss", color="orange"); axes[1].set_title("Discriminator Loss")
    axes[2].plot(history["val_miou"], label="val mIoU", color="green"); axes[2].set_title("Validation mIoU")
    for ax in axes:
        ax.legend(); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()

plot_history(history)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 14 — Inference & Visualization                            ║
# ╚══════════════════════════════════════════════════════════════════╝
def predict_and_visualize(model_path, image_path, save_path=None, mask_path=None):
    """
    Load checkpoint và dự đoán mask cho ảnh vệ tinh bất kỳ.

    Args:
        model_path : đường dẫn file .pth checkpoint Generator.
        image_path : đường dẫn ảnh vệ tinh RGB đầu vào.
        save_path  : (tuỳ chọn) lưu figure kết quả ra file.
        mask_path  : (tuỳ chọn) đường dẫn ground-truth mask (grayscale
                     LoveDA format: nhãn 1-7) để tính và hiển thị IoU.

    Luồng xử lý:
        1. Pad ảnh và chia thành các patch IMG_SIZE×IMG_SIZE (sliding window,
           stride = IMG_SIZE//2, overlap 50%) để xử lý ảnh kích thước bất kỳ.
        2. Mỗi patch normalize theo ImageNet rồi forward qua Generator.
        3. Cộng dồn logits (float32) lên logit_sum và đếm số lần mỗi pixel
           được dự đoán (count_map), sau đó lấy argmax của trung bình logits.
           → Khử artifact đường biên giữa các patch.
        4. Hiển thị: ảnh gốc | mask tô màu | overlay (alpha=0.5)
           + biểu đồ phân bố lớp (%).
        5. Nếu có mask_path: thêm cột ground-truth và in per-class IoU / mIoU.
    """
    # ── 1. Load model ──────────────────────────────────────────────
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50", encoder_weights=None,
        in_channels=3, classes=NUM_CLASSES, activation=None,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ── 2. Đọc ảnh gốc ─────────────────────────────────────────────
    orig = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    H, W = orig.shape[:2]

    # ── 3. Sliding-window inference ────────────────────────────────
    _norm = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    STRIDE = IMG_SIZE // 2   # 50% overlap → khử artifact biên patch

    # Pad đủ để patch cuối cùng không bị cắt ngắn
    pad_h  = (IMG_SIZE - H % IMG_SIZE) % IMG_SIZE
    pad_w  = (IMG_SIZE - W % IMG_SIZE) % IMG_SIZE
    # Thêm một stride nữa để sliding window luôn phủ hết mọi pixel
    pad_h += STRIDE
    pad_w += STRIDE
    padded = np.pad(orig, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    pH, pW = padded.shape[:2]

    logit_sum = np.zeros((pH, pW, NUM_CLASSES), dtype=np.float32)  # tổng logits
    count_map = np.zeros((pH, pW),              dtype=np.float32)  # số lần mỗi pixel được predict

    with torch.no_grad():
        for y in range(0, pH - IMG_SIZE + 1, STRIDE):
            for x in range(0, pW - IMG_SIZE + 1, STRIDE):
                patch = padded[y:y + IMG_SIZE, x:x + IMG_SIZE]
                inp   = _norm(image=patch)["image"].unsqueeze(0).to(device)
                # logits: [1, C, H, W] → [H, W, C] trên CPU
                logit = model(inp).squeeze(0).permute(1, 2, 0).cpu().numpy()
                logit_sum[y:y + IMG_SIZE, x:x + IMG_SIZE] += logit
                count_map[y:y + IMG_SIZE, x:x + IMG_SIZE] += 1

    # Chia trung bình logits rồi lấy argmax
    count_map = np.maximum(count_map, 1)                           # tránh chia 0
    avg_logit = logit_sum / count_map[..., np.newaxis]             # [pH, pW, C]
    pred_map  = avg_logit.argmax(axis=-1).astype(np.uint8)[:H, :W]  # cắt về kích thước gốc

    # ── 4. Tô màu và tạo overlay ───────────────────────────────────
    colored = COLOR_MAP[pred_map]                                    # [H, W, 3] uint8
    overlay = (orig * 0.5 + colored * 0.5).astype(np.uint8)

    # ── 5. (Tuỳ chọn) Ground-truth & per-class IoU ─────────────────
    gt_mask   = None
    class_iou = None
    miou      = None
    if mask_path is not None and os.path.exists(mask_path):
        gt_raw  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)
        gt_raw[gt_raw == 255] = 1
        gt_mask = np.clip(gt_raw - 1, 0, NUM_CLASSES - 1).astype(np.uint8)

        class_iou = []
        for c in range(NUM_CLASSES):
            inter = ((pred_map == c) & (gt_mask == c)).sum()
            union = ((pred_map == c) | (gt_mask == c)).sum()
            class_iou.append(inter / union if union > 0 else float("nan"))
        miou = np.nanmean(class_iou)

        print(f"mIoU = {miou:.4f}")
        for i, (name, iou) in enumerate(zip(CLASS_NAMES, class_iou)):
            tag = f"{iou:.4f}" if not np.isnan(iou) else "  N/A"
            print(f"  [{i}] {name:<15s}: IoU = {tag}")

    # ── 6. Thống kê tỉ lệ class ────────────────────────────────────
    total_px     = pred_map.size
    class_ratios = [(pred_map == c).sum() / total_px * 100 for c in range(NUM_CLASSES)]

    # ── 7. Vẽ figure ───────────────────────────────────────────────
    has_gt  = gt_mask is not None
    n_img   = 4 if has_gt else 3               # số cột ảnh
    fig_w   = 5 * n_img + 4                    # tổng chiều rộng

    fig = plt.figure(figsize=(fig_w, 6))
    gs  = fig.add_gridspec(
        1, n_img + 1,
        width_ratios=[4] * n_img + [3],
        wspace=0.35,
    )

    # Ảnh gốc
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(orig)
    ax0.set_title("Ảnh vệ tinh gốc", fontsize=12, fontweight="bold")
    ax0.axis("off")

    # Mask dự đoán tô màu
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(colored)
    ax1.set_title("Phân đoạn (GAN)", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Overlay
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(overlay)
    ax2.set_title("Overlay (α=0.5)", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # Ground-truth (tuỳ chọn)
    if has_gt:
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.imshow(COLOR_MAP[gt_mask])
        gt_title = f"Ground Truth\n(mIoU = {miou:.3f})" if miou is not None else "Ground Truth"
        ax3.set_title(gt_title, fontsize=12, fontweight="bold")
        ax3.axis("off")

    # Biểu đồ phân bố class
    ax_bar = fig.add_subplot(gs[0, n_img])
    bar_colors = [COLOR_MAP[i] / 255.0 for i in range(NUM_CLASSES)]
    bars = ax_bar.barh(
        range(NUM_CLASSES), class_ratios,
        color=bar_colors, edgecolor="grey", linewidth=0.5,
    )
    ax_bar.set_yticks(range(NUM_CLASSES))
    ax_bar.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax_bar.set_xlabel("Tỉ lệ (%)", fontsize=9)
    ax_bar.set_title("Phân bố lớp", fontsize=11, fontweight="bold")
    ax_bar.invert_yaxis()
    ax_bar.grid(axis="x", alpha=0.3)
    for bar, ratio in zip(bars, class_ratios):
        if ratio > 1.5:
            ax_bar.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{ratio:.1f}%", va="center", fontsize=8,
            )

    # Legend chung
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP[i] / 255.0, label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center", ncol=NUM_CLASSES,
        fontsize=9, framealpha=0.9, edgecolor="#ccc",
        bbox_to_anchor=(0.45, -0.04),
    )
    plt.suptitle(
        f"Kết quả phân đoạn: {os.path.basename(image_path)}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Đã lưu kết quả → {save_path}")
    plt.show()
    return pred_map


# Ví dụ chạy inference sau khi train xong:
# predict_and_visualize(
#     "best_generator.pth",
#     "./Test/Test/Urban/images_png/5167.png",
#     save_path="./predictions/5167_result.png",
#     mask_path=None,  # hoặc truyền đường dẫn mask GT để tính IoU
# )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Cell 15 — Visualize Dataset Samples                            ║
# ╚══════════════════════════════════════════════════════════════════╝
def visualize_dataset_samples(root, n_samples=4, indices=None, split_name="Train", save_path=None):
    """
    Hiển thị các cặp (ảnh, mask tô màu) từ tập Train hoặc Val.

    Args:
        root       : đường dẫn thư mục dataset (VD: "./LoveDA_patch/Train").
        n_samples  : số cặp muốn xem (bị bỏ qua nếu truyền `indices`).
        indices    : list chỉ số cụ thể muốn xem, VD [0, 5, 12].
                     Nếu None → chọn ngẫu nhiên n_samples mẫu.
        split_name : tên split để hiển thị trên tiêu đề ("Train" / "Val").
        save_path  : (tuỳ chọn) lưu figure kết quả ra file.

    Layout mỗi hàng:
        [Ảnh gốc] | [Mask tô màu] | [Overlay α=0.5] | [Phân bố lớp (%)]
    """
    img_dir  = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    names    = sorted(os.listdir(img_dir))

    # Chọn index
    if indices is not None:
        chosen = [names[i] for i in indices]
    else:
        chosen = list(np.random.choice(names, size=min(n_samples, len(names)), replace=False))

    n = len(chosen)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]   # đảm bảo axes luôn là 2D

    col_titles = ["Ảnh gốc", "Mask (Ground Truth)", "Overlay (α=0.5)", "Phân bố lớp (%)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=8)

    for row, name in enumerate(chosen):
        # ── Đọc ảnh và mask ──────────────────────────────────────────
        img  = cv2.cvtColor(cv2.imread(os.path.join(img_dir, name)), cv2.COLOR_BGR2RGB)
        mask_raw = cv2.imread(os.path.join(mask_dir, name), cv2.IMREAD_GRAYSCALE).astype(np.int32)

        # Mapping nhãn LoveDA: 1-7 → 0-6, ignore (0 & 255) → 0
        mask_raw[mask_raw == 255] = 1
        mask = np.clip(mask_raw - 1, 0, NUM_CLASSES - 1).astype(np.uint8)

        # ── Tô màu & overlay ─────────────────────────────────────────
        colored = COLOR_MAP[mask]
        overlay = (img * 0.5 + colored * 0.5).astype(np.uint8)

        # ── Thống kê tỉ lệ class ─────────────────────────────────────
        total_px     = mask.size
        class_ratios = [(mask == c).sum() / total_px * 100 for c in range(NUM_CLASSES)]

        # ── Vẽ cột 0: ảnh gốc ────────────────────────────────────────
        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(name, fontsize=8, rotation=0, labelpad=120, va="center")
        axes[row, 0].axis("off")

        # ── Vẽ cột 1: mask tô màu ─────────────────────────────────────
        axes[row, 1].imshow(colored)
        axes[row, 1].axis("off")

        # ── Vẽ cột 2: overlay ────────────────────────────────────────
        axes[row, 2].imshow(overlay)
        axes[row, 2].axis("off")

        # ── Vẽ cột 3: biểu đồ cột nằm ngang ─────────────────────────
        ax_bar = axes[row, 3]
        bar_colors = [COLOR_MAP[i] / 255.0 for i in range(NUM_CLASSES)]
        bars = ax_bar.barh(
            range(NUM_CLASSES), class_ratios,
            color=bar_colors, edgecolor="grey", linewidth=0.5,
        )
        ax_bar.set_yticks(range(NUM_CLASSES))
        ax_bar.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax_bar.set_xlabel("Tỉ lệ (%)", fontsize=9)
        ax_bar.invert_yaxis()
        ax_bar.grid(axis="x", alpha=0.3)
        for bar, ratio in zip(bars, class_ratios):
            if ratio > 1.5:
                ax_bar.text(
                    bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{ratio:.1f}%", va="center", fontsize=8,
                )

    # ── Legend chung ─────────────────────────────────────────────────
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP[i] / 255.0, label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center", ncol=NUM_CLASSES,
        fontsize=9, framealpha=0.9, edgecolor="#ccc",
        bbox_to_anchor=(0.45, -0.02),
    )
    plt.suptitle(
        f"Dataset samples — {split_name}  ({n} ảnh)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Đã lưu → {save_path}")
    plt.show()


# Ví dụ sử dụng:
# visualize_dataset_samples(TRAIN_ROOT, n_samples=4, split_name="Train")
# visualize_dataset_samples(VAL_ROOT,   n_samples=3, split_name="Val")
# visualize_dataset_samples(TRAIN_ROOT, indices=[0, 10, 50], split_name="Train")
