#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib
matplotlib.use("Agg")   # Headless backend — bắt buộc khi train qua SSH (không có display)
import matplotlib.pyplot as plt

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

import albumentations as A
from albumentations.pytorch import ToTensorV2


# In[5]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

TRAIN_ROOT  = "./LoveDA_patch/Train"
VAL_ROOT    = "./LoveDA_patch/Val"
TEST_ROOT   = "./Test/Test"

IMG_SIZE    = 512          # DeepLabV3+ hoạt động tốt hơn ở 512 (ResNet50 stride 16)
BATCH_SIZE  = 4            # Giảm xuống do model lớn hơn
EPOCHS      = 300 
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


# In[6]:


train_transform = A.Compose([
    A.RandomCrop(IMG_SIZE, IMG_SIZE),
    A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomRotate90(p=1.0),
        A.Transpose(p=1.0),
    ], p=0.75),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    ], p=0.5),
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



# In[7]:


train_ds = PatchDataset(TRAIN_ROOT, transform=train_transform)
val_ds   = PatchDataset(VAL_ROOT,   transform=val_transform)
test_ds  = LoveDATestDataset(TEST_ROOT)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=1)

print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")


# In[8]:


# ===== FINAL VERSION KHÔNG BAO GIỜ ĐEN =====

import albumentations as A
import numpy as np
import torch

vis_ds = PatchDataset(TRAIN_ROOT, transform=None)
img, mask = vis_ds[0]

# ===== convert chuẩn =====
if isinstance(img, torch.Tensor):
    img = img.permute(1, 2, 0).cpu().numpy()

if isinstance(mask, torch.Tensor):
    mask = mask.cpu().numpy()

# FIX scale
if img.max() <= 1.0:
    img = (img * 255).astype(np.uint8)
else:
    img = img.astype(np.uint8)

mask = mask.astype(np.uint8)

# ===== augment =====
geom_transform = A.Compose([
    A.RandomRotate90(p=1.0),
    A.HorizontalFlip(p=1.0),
])

optical_transform = A.Compose([
    A.RandomBrightnessContrast(p=1.0),
    A.HueSaturationValue(p=1.0),
])

blur_transform = A.Compose([
    A.GaussianBlur(blur_limit=(5,7), p=1.0),
])

aug_geom = geom_transform(image=img, mask=mask)
aug_opt  = optical_transform(image=img, mask=mask)
aug_blur = blur_transform(image=img, mask=mask)

# ===== mask to color =====
def mask_to_color(mask):
    return COLOR_MAP[mask]

# ===== plot =====
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

titles = ["Original Image", "Geometric Augmentation", "Optical Augmentation", "Gaussian Blur"]

axes[0,0].imshow(img)
axes[0,1].imshow(aug_geom["image"])
axes[0,2].imshow(aug_opt["image"])
axes[0,3].imshow(aug_blur["image"])

axes[1,0].imshow(mask_to_color(mask))
axes[1,1].imshow(mask_to_color(aug_geom["mask"]))
axes[1,2].imshow(mask_to_color(aug_opt["mask"]))
axes[1,3].imshow(mask_to_color(aug_blur["mask"]))

for i in range(4):
    axes[0,i].set_title(titles[i])
    axes[0,i].axis("off")
    axes[1,i].axis("off")

plt.suptitle("LoveDA Satellite Image Data Augmentation", fontsize=16)
plt.tight_layout()
os.makedirs("./predictions", exist_ok=True)
plt.savefig("./predictions/augmentation_preview.png", dpi=150, bbox_inches="tight")
print("Đã lưu augmentation preview → ./predictions/augmentation_preview.png")
plt.close()   # Giải phóng RAM


# In[9]:


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


# In[10]:


G = smp.DeepLabV3Plus(
    encoder_name        = "resnet50",
    encoder_weights     = "imagenet",   # Pretrained ImageNet — tăng mạnh hiệu năng
    in_channels         = 3,
    classes             = NUM_CLASSES,
    activation          = None,         # Raw logits — để tương thích với CE loss
).to(device)

num_params_G = sum(p.numel() for p in G.parameters()) / 1e6
print(f"Generator (DeepLabV3+ ResNet50): {num_params_G:.1f}M params")


# In[11]:


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


# In[12]:


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


# In[13]:


opt_G = torch.optim.AdamW([
    {"params": G.encoder.parameters(), "lr": LR_G * 0.1},   # Fine-tune encoder nhẹ
    {"params": G.decoder.parameters(), "lr": LR_G},
    {"params": G.segmentation_head.parameters(), "lr": LR_G},
], betas=(0.9, 0.999), weight_decay=1e-4)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))

# CosineAnnealingLR — giảm LR mượt từ đỉnh xuống eta_min
sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=EPOCHS, eta_min=1e-6)
sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=EPOCHS, eta_min=1e-7)


# In[14]:


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


# In[15]:


# class EarlyStopping:
#     def __init__(self, patience=15, path="best_generator.pth"):
#         self.patience   = patience
#         self.counter    = 0
#         self.best_miou  = None
#         self.early_stop = False
#         self.path       = path

#     def __call__(self, miou, model):
#         if self.best_miou is None or miou > self.best_miou:
#             self.best_miou = miou
#             torch.save(model.state_dict(), self.path)
#             self.counter = 0
#             print(f"  ✓ New best mIoU: {miou:.4f}  →  Saved to {self.path}")
#         else:
#             self.counter += 1
#             print(f"  EarlyStopping [{self.counter}/{self.patience}]  (best: {self.best_miou:.4f})")
#             if self.counter >= self.patience:
#                 self.early_stop = True

# early_stopping = EarlyStopping(patience=15, path="best_generator.pth")


# In[ ]:


print("\n" + "="*65)
print("  GAN Satellite Segmentation  |  DeepLabV3+ ResNet50 + PatchGAN")
print("="*65)

history = {"g_loss": [], "d_loss": [], "val_loss": [], "val_miou": []}

best_miou = 0.0

for epoch in range(EPOCHS):

    G.train()
    D.train()

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{EPOCHS}")
    ep_g, ep_d = 0.0, 0.0

    for imgs, masks in loop:

        imgs  = imgs.to(device)
        masks = torch.clamp(masks.to(device), 0, NUM_CLASSES - 1)

        # ───────────── Train Generator ─────────────
        logits = G(imgs)                         # [N,7,H,W]
        soft_pred = torch.softmax(logits, dim=1)

        fake_in = torch.cat([imgs, soft_pred], dim=1)

        d_fake = D(fake_in)

        g_adv  = gan_loss(d_fake, torch.ones_like(d_fake))
        g_ce   = ce_loss(logits, masks)
        g_dice = dice_loss(logits, masks)

        g_loss = LAMBDA_ADV * g_adv + g_ce + g_dice

        opt_G.zero_grad()
        g_loss.backward()

        nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)

        opt_G.step()

        # ───────────── Train Discriminator ─────────────
        real_onehot = F.one_hot(masks, NUM_CLASSES).permute(0,3,1,2).float()

        real_in  = torch.cat([imgs, real_onehot], dim=1)
        fake_in2 = torch.cat([imgs, soft_pred.detach()], dim=1)

        d_real = D(real_in)
        d_fake = D(fake_in2)

        d_loss = (
            gan_loss(d_real, torch.ones_like(d_real) * 0.9) +
            gan_loss(d_fake, torch.zeros_like(d_fake))
        ) * 0.5

        if d_loss.item() > 0.3:
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

        ep_g += g_loss.item()
        ep_d += d_loss.item()

        loop.set_postfix(
            G=f"{g_loss.item():.4f}",
            D=f"{d_loss.item():.4f}"
        )

    # ───────────── Scheduler step ─────────────
    sched_G.step()
    sched_D.step()

    # ───────────── Validation ─────────────
    val_loss, val_miou = validate()

    lr_now = sched_G.get_last_lr()[0]
    n = len(train_loader)

    g_epoch = ep_g / n
    d_epoch = ep_d / n

    print(
        f"\n  G_loss={g_epoch:.4f} | "
        f"D_loss={d_epoch:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_mIoU={val_miou:.4f} | "
        f"lr={lr_now:.2e}"
    )

    history["g_loss"].append(g_epoch)
    history["d_loss"].append(d_epoch)
    history["val_loss"].append(val_loss)
    history["val_miou"].append(val_miou)

    # ───────────── Save BEST model ─────────────
    if val_miou > best_miou:
        best_miou = val_miou

        torch.save(G.state_dict(), "best_generator.pth")
        torch.save(D.state_dict(), "best_discriminator.pth")

        print(f"💾 Saved BEST model | mIoU = {best_miou:.4f}")

    # ───────────── Save LAST checkpoint ─────────────
    torch.save(G.state_dict(), "last_generator.pth")
    torch.save(D.state_dict(), "last_discriminator.pth")

print("\n✅ Huấn luyện hoàn tất!")
print("🏆 Best mIoU:", round(best_miou, 4))


# In[ ]:


def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["g_loss"], label="G loss"); axes[0].set_title("Generator Loss")
    axes[1].plot(history["d_loss"], label="D loss", color="orange"); axes[1].set_title("Discriminator Loss")
    axes[2].plot(history["val_miou"], label="val mIoU", color="green"); axes[2].set_title("Validation mIoU")
    for ax in axes:
        ax.legend(); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs("./predictions", exist_ok=True)
    plt.savefig("./predictions/training_history.png", dpi=150, bbox_inches="tight")
    print("Đã lưu training history → ./predictions/training_history.png")
    plt.close()   # Giải phóng RAM

plot_history(history)


# In[ ]:


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
           stride = IMG_SIZE, không overlap) để xử lý ảnh kích thước bất kỳ.
        2. Mỗi patch normalize theo ImageNet rồi forward qua Generator.
        3. Ghép kết quả argmax thành pred_map toàn ảnh, cắt về kích thước gốc.
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

    pad_h  = (IMG_SIZE - H % IMG_SIZE) % IMG_SIZE
    pad_w  = (IMG_SIZE - W % IMG_SIZE) % IMG_SIZE
    padded = np.pad(orig, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    pH, pW = padded.shape[:2]

    pred_map = np.zeros((pH, pW), dtype=np.uint8)
    with torch.no_grad():
        for y in range(0, pH, IMG_SIZE):
            for x in range(0, pW, IMG_SIZE):
                patch = padded[y:y + IMG_SIZE, x:x + IMG_SIZE]
                inp   = _norm(image=patch)["image"].unsqueeze(0).to(device)
                out   = model(inp).argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
                pred_map[y:y + IMG_SIZE, x:x + IMG_SIZE] = out

    pred_map = pred_map[:H, :W]   # cắt về kích thước gốc

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

    # Luôn lưu ảnh (SSH không có display)
    if save_path is None:
        os.makedirs("./predictions", exist_ok=True)
        save_path = f"./predictions/{os.path.splitext(os.path.basename(image_path))[0]}_result.png"
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Đã lưu kết quả → {save_path}")
    plt.close()   # Giải phóng RAM
    return pred_map


# In[ ]:


predict_and_visualize(
    model_path = "best_generator.pth",
    image_path = "./Train/Train/Rural/images_png/1000.png",  # ảnh train cụ thể
    mask_path  = "./Train/Train/Rural/masks_png/1000.png",   # mask thật của ảnh đó
)


# In[ ]:


predict_and_visualize(
    model_path = "last_generator.pth",
    image_path = "./Train/Train/Rural/images_png/1000.png",  # ảnh train cụ thể
    mask_path  = "./Train/Train/Rural/masks_png/1000.png",   # mask thật của ảnh đó
)


# In[ ]:


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

    # Luôn lưu ảnh (SSH không có display)
    if save_path is None:
        os.makedirs("./predictions", exist_ok=True)
        save_path = f"./predictions/{os.path.splitext(os.path.basename(image_path))[0]}_result.png"
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Đã lưu kết quả → {save_path}")
    plt.close()   # Giải phóng RAM
    return pred_map


# Ví dụ chạy inference sau khi train xong:
# predict_and_visualize(
#     "best_generator.pth",
#     "./Test/Test/Urban/images_png/5167.png",
#     save_path="./predictions/5167_result.png",
#     mask_path=None,  # hoặc truyền đường dẫn mask GT để tính IoU
# )


# In[ ]:


predict_and_visualize(
    model_path = "last_generator.pth",
    image_path = "./Train/Train/Rural/images_png/1000.png",  # ảnh train cụ thể
    mask_path  = "./Train/Train/Rural/masks_png/1000.png",   # mask thật của ảnh đó
)


# In[ ]:


import seaborn as sns   # pip install seaborn  (nếu chưa có)


# ── Hàm 1: Tính confusion matrix trên toàn bộ val set ──────────────
def compute_confusion_matrix(model, loader, num_classes, device):
    """
    Duyệt qua val_loader, cộng dồn confusion matrix pixel-level.

    Returns:
        cm : ndarray shape (num_classes, num_classes), dtype int64
             cm[true_class, pred_class]
    """
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(device)
            masks = torch.clamp(masks, 0, num_classes - 1)
            logits = model(imgs)
            preds  = logits.argmax(dim=1).cpu().numpy()   # [N, H, W]
            tgts   = masks.numpy()                         # [N, H, W]
            for p, t in zip(preds, tgts):
                # np.add.at để cộng dồn nhanh
                np.add.at(cm, (t.ravel(), p.ravel()), 1)
    return cm


# ── Hàm 2: Tính tất cả metric từ confusion matrix ──────────────────
def metrics_from_cm(cm):
    """
    Tính per-class IoU, Precision, Recall, F1 và Overall Pixel Accuracy.

    Args:
        cm : ndarray (C, C)  — cm[true, pred]

    Returns: dict với các key:
        iou, precision, recall, f1  : ndarray (C,)
        miou, mean_precision, mean_recall, mean_f1, pixel_acc : float
    """
    C = cm.shape[0]
    iou = np.zeros(C)
    precision = np.zeros(C)
    recall    = np.zeros(C)
    f1        = np.zeros(C)

    for c in range(C):
        tp  = cm[c, c]
        fp  = cm[:, c].sum() - tp   # dự đoán là c nhưng thực ra không phải
        fn  = cm[c, :].sum() - tp   # thực ra là c nhưng không dự đoán được
        den_iou  = tp + fp + fn
        den_prec = tp + fp
        den_rec  = tp + fn

        iou[c]       = tp / den_iou  if den_iou  > 0 else float("nan")
        precision[c] = tp / den_prec if den_prec > 0 else float("nan")
        recall[c]    = tp / den_rec  if den_rec  > 0 else float("nan")

        p, r = precision[c], recall[c]
        if not (np.isnan(p) or np.isnan(r)) and (p + r) > 0:
            f1[c] = 2 * p * r / (p + r)
        else:
            f1[c] = float("nan")

    pixel_acc    = cm.diagonal().sum() / cm.sum()
    miou         = np.nanmean(iou)
    mean_prec    = np.nanmean(precision)
    mean_recall  = np.nanmean(recall)
    mean_f1      = np.nanmean(f1)

    return {
        "iou"          : iou,
        "precision"    : precision,
        "recall"       : recall,
        "f1"           : f1,
        "miou"         : miou,
        "mean_precision": mean_prec,
        "mean_recall"  : mean_recall,
        "mean_f1"      : mean_f1,
        "pixel_acc"    : pixel_acc,
    }


# ── Hàm 3: In bảng kết quả ra console ─────────────────────────────
def print_metrics_table(metrics, class_names):
    print("\n" + "="*72)
    print(f"  {'Class':<18} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("="*72)
    for i, name in enumerate(class_names):
        iou = metrics["iou"][i]
        pre = metrics["precision"][i]
        rec = metrics["recall"][i]
        f1  = metrics["f1"][i]
        fmt = lambda v: f"{v:.4f}" if not np.isnan(v) else "  N/A "
        print(f"  [{i}] {name:<15} {fmt(iou):>8} {fmt(pre):>10} {fmt(rec):>8} {fmt(f1):>8}")
    print("="*72)
    print(f"  {'Mean (excl. NaN)':<18} "
          f"{metrics['miou']:>8.4f} "
          f"{metrics['mean_precision']:>10.4f} "
          f"{metrics['mean_recall']:>8.4f} "
          f"{metrics['mean_f1']:>8.4f}")
    print(f"\n  Overall Pixel Accuracy : {metrics['pixel_acc']:.4f}  "
          f"({metrics['pixel_acc']*100:.2f}%)")
    print("="*72)


# ── Hàm 4: Visualize toàn diện ─────────────────────────────────────
def plot_full_evaluation(metrics, cm, class_names, history=None,
                         color_map=None, save_path=None):
    """
    Vẽ figure đánh giá đầy đủ gồm 5 panel:

        Panel 1 (trái trên) : Bar chart per-class IoU
        Panel 2 (giữa trên) : Grouped bar chart Precision & Recall
        Panel 3 (phải trên) : Bar chart per-class F1-score
        Panel 4 (trái dưới) : Normalized Confusion Matrix (heatmap)
        Panel 5 (phải dưới) : Đường cong huấn luyện (loss + mIoU)
    """
    bar_colors = [color_map[i] / 255.0 for i in range(len(class_names))]
    C   = len(class_names)
    x   = np.arange(C)
    iou = metrics["iou"]
    pre = metrics["precision"]
    rec = metrics["recall"]
    f1  = metrics["f1"]

    fig = plt.figure(figsize=(22, 14))
    gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    # ── Panel 1: Per-class IoU ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(x, np.nan_to_num(iou), color=bar_colors,
                   edgecolor="grey", linewidth=0.6)
    ax1.axhline(metrics["miou"], color="red", linewidth=1.5,
                linestyle="--", label=f"mIoU = {metrics['miou']:.4f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("IoU")
    ax1.set_title("Per-class IoU", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, np.nan_to_num(iou)):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # ── Panel 2: Precision & Recall (grouped) ─────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    w   = 0.35
    ax2.bar(x - w/2, np.nan_to_num(pre), width=w, label="Precision",
            color=[(*c[:3], 0.85) for c in bar_colors], edgecolor="grey", linewidth=0.5)
    ax2.bar(x + w/2, np.nan_to_num(rec), width=w, label="Recall",
            color=[(*c[:3], 0.50) for c in bar_colors], edgecolor="grey", linewidth=0.5)
    ax2.axhline(metrics["mean_precision"], color="steelblue", linewidth=1.2,
                linestyle="--", label=f"Mean Prec = {metrics['mean_precision']:.3f}")
    ax2.axhline(metrics["mean_recall"],    color="darkorange", linewidth=1.2,
                linestyle=":",  label=f"Mean Rec  = {metrics['mean_recall']:.3f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("Score")
    ax2.set_title("Per-class Precision & Recall", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3: Per-class F1-score ───────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(x, np.nan_to_num(f1), color=bar_colors,
                    edgecolor="grey", linewidth=0.6)
    ax3.axhline(metrics["mean_f1"], color="purple", linewidth=1.5,
                linestyle="--", label=f"Mean F1 = {metrics['mean_f1']:.4f}")
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("F1-score")
    ax3.set_title("Per-class F1-score", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars3, np.nan_to_num(f1)):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # ── Panel 4: Normalized Confusion Matrix ──────────────────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    cm_norm = cm.astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm_norm / row_sum, 0)   # normalize theo hàng (true class)

    sns.heatmap(
        cm_norm, ax=ax4,
        annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.4, linecolor="lightgrey",
        cbar_kws={"shrink": 0.8, "label": "Tỉ lệ dự đoán"},
        annot_kws={"size": 9},
        vmin=0, vmax=1,
    )
    ax4.set_xlabel("Predicted Class", fontsize=11)
    ax4.set_ylabel("True Class", fontsize=11)
    ax4.set_title("Normalized Confusion Matrix\n(hàng = true class, ô = tỉ lệ dự đoán)",
                  fontsize=13, fontweight="bold")
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0, fontsize=9)

    # ── Panel 5: Training curves ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    if history is not None:
        epochs_range = range(1, len(history["g_loss"]) + 1)
        ax5_twin = ax5.twinx()

        ax5.plot(epochs_range, history["g_loss"],
                 color="steelblue", linewidth=1.5, label="G Loss (train)")
        ax5.plot(epochs_range, history["val_loss"],
                 color="tomato", linewidth=1.5, linestyle="--", label="Val Loss")
        ax5_twin.plot(epochs_range, history["val_miou"],
                      color="seagreen", linewidth=2, label="Val mIoU")

        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Loss", color="steelblue")
        ax5_twin.set_ylabel("mIoU", color="seagreen")
        ax5_twin.set_ylim(0, 1)
        ax5_twin.tick_params(axis="y", labelcolor="seagreen")
        ax5.tick_params(axis="y", labelcolor="steelblue")

        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
        ax5.grid(alpha=0.3)
        ax5.set_title("Đường cong huấn luyện\n(Loss + Val mIoU)", fontsize=13, fontweight="bold")
    else:
        ax5.text(0.5, 0.5, "history=None\n(không có dữ liệu training)",
                 ha="center", va="center", fontsize=11, color="grey")
        ax5.set_title("Đường cong huấn luyện", fontsize=13)

    # ── Tiêu đề tổng + summary metrics ───────────────────────────
    summary = (f"mIoU = {metrics['miou']:.4f}   |   "
               f"Mean F1 = {metrics['mean_f1']:.4f}   |   "
               f"Pixel Accuracy = {metrics['pixel_acc']*100:.2f}%")
    fig.suptitle(
        f"Đánh giá mô hình DeepLabV3+ ResNet50 trên tập Validation\n"
        f"{summary}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    if save_path is None:
        os.makedirs("./predictions", exist_ok=True)
        save_path = "./predictions/full_evaluation.png"
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Đã lưu biểu đồ đánh giá → {save_path}")
    plt.close()   # Giải phóng RAM


# ── Hàm 5: Training history nâng cao (thay thế plot_history) ───────
def plot_training_history(history, save_path=None):
    """
    Vẽ 4 đường cong huấn luyện chi tiết:
        - Generator Loss
        - Discriminator Loss
        - Validation Loss
        - Validation mIoU (highlight best epoch)
    """
    epochs = range(1, len(history["g_loss"]) + 1)
    best_ep = int(np.argmax(history["val_miou"])) + 1
    best_miou_val = max(history["val_miou"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Lịch sử huấn luyện — GAN Satellite Segmentation",
                 fontsize=14, fontweight="bold")

    configs = [
        (axes[0, 0], history["g_loss"],   "steelblue",  "Generator Loss (train)",      "Loss"),
        (axes[0, 1], history["d_loss"],   "darkorange", "Discriminator Loss (train)",  "Loss"),
        (axes[1, 0], history["val_loss"], "tomato",     "Validation Loss",             "Loss"),
        (axes[1, 1], history["val_miou"], "seagreen",   "Validation mIoU",             "mIoU"),
    ]
    for ax, data, color, title, ylabel in configs:
        ax.plot(epochs, data, color=color, linewidth=2)
        if title == "Validation mIoU":
            ax.axhline(best_miou_val, color="red", linewidth=1, linestyle="--",
                       label=f"Best = {best_miou_val:.4f} (epoch {best_ep})")
            ax.scatter([best_ep], [best_miou_val], color="red", s=60, zorder=5)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        os.makedirs("./predictions", exist_ok=True)
        save_path = "./predictions/training_history_detail.png"
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Đã lưu training history → {save_path}")
    plt.close()   # Giải phóng RAM


# In[ ]:


# ── CHẠY ĐÁNH GIÁ ────────────────────────────────────────────────────
# Uncomment khối dưới đây sau khi huấn luyện xong để chạy đánh giá:

# 1. Load best model
G_eval = smp.DeepLabV3Plus(
    encoder_name="resnet50", encoder_weights=None,
    in_channels=3, classes=NUM_CLASSES, activation=None,
).to(device)
G_eval.load_state_dict(torch.load("best_generator.pth", map_location=device))

# 2. Tính confusion matrix trên val set
print("Đang tính confusion matrix trên tập Validation...")
cm = compute_confusion_matrix(G_eval, val_loader, NUM_CLASSES, device)

# 3. Tính tất cả metrics
metrics = metrics_from_cm(cm)

# 4. In bảng kết quả
print_metrics_table(metrics, CLASS_NAMES)

# 5. Vẽ toàn bộ biểu đồ đánh giá
plot_full_evaluation(
    metrics, cm, CLASS_NAMES,
    history=history,           # truyền history từ training loop
    color_map=COLOR_MAP,
    save_path="./predictions/full_evaluation.png",
)

# 6. (Tùy chọn) Vẽ riêng training history chi tiết
plot_training_history(history, save_path="./predictions/training_history_detail.png")


