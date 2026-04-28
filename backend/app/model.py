"""
model.py — Load và chạy inference với DeepLabV3+ ResNet50 Generator
Được tách riêng để main.py load model 1 lần lúc startup.
"""

import io
import os
import logging
import numpy as np
from PIL import Image

import torch
import segmentation_models_pytorch as smp

import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

# ── Cấu hình (phải khớp với lúc train) ──────────────────────────────────────
NUM_CLASSES = 7
IMG_SIZE    = 512
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren", "Forest", "Agricultural"]

# LoveDA official color map (RGB)
COLOR_MAP = np.array([
    [255, 255, 255],   # 0: Background
    [255,   0,   0],   # 1: Building
    [255, 255,   0],   # 2: Road
    [  0,   0, 255],   # 3: Water
    [159, 129, 183],   # 4: Barren
    [  0, 255,   0],   # 5: Forest
    [255, 195, 128],   # 6: Agricultural
], dtype=np.uint8)

# ── Transform (giống val_transform lúc train) ────────────────────────────────
_infer_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ── Biến toàn cục giữ model ──────────────────────────────────────────────────
_generator = None


def load_model(model_path: str = "/code/last_generator.pth") -> None:
    """
    Load Generator từ checkpoint vào bộ nhớ (gọi 1 lần khi startup).
    """
    global _generator

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy file model tại: {model_path}\n"
            "Hãy đảm bảo volume mount đúng trong docker-compose.yml."
        )

    logger.info(f"Loading model từ {model_path} trên {DEVICE}...")

    model = smp.DeepLabV3Plus(
        encoder_name    = "resnet50",
        encoder_weights = None,      # Không cần pretrained khi load checkpoint
        in_channels     = 3,
        classes         = NUM_CLASSES,
        activation      = None,
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    _generator = model
    logger.info("✅ Model loaded thành công!")


def get_model():
    """Trả về model đã load, ném lỗi nếu chưa load."""
    if _generator is None:
        raise RuntimeError("Model chưa được load. Gọi load_model() trước.")
    return _generator


def predict(image_bytes: bytes) -> dict:
    """
    Chạy inference trên ảnh vệ tinh đầu vào.

    Args:
        image_bytes: raw bytes của ảnh upload (PNG / JPEG / TIFF).

    Returns:
        dict với:
          - "mask_image": bytes ảnh PNG của mask tô màu (LoveDA palette)
          - "overlay_image": bytes ảnh PNG overlay (ảnh gốc + mask, alpha=0.5)
          - "class_stats": list[dict] thống kê tỉ lệ từng class (%)
    """
    model = get_model()

    # ── 1. Đọc ảnh đầu vào ──────────────────────────────────────────────────
    orig_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_np  = np.array(orig_pil)            # [H, W, 3] uint8
    H, W     = orig_np.shape[:2]

    # ── 2. Overlapping sliding-window inference ────────────────────────────────
    # Dùng stride = IMG_SIZE // 2 (overlap 50%) + Gaussian blending
    # để xóa viền cứng giữa các tile.
    STRIDE = IMG_SIZE // 2

    # Padding để mọi pixel đều được bao phủ ít nhất bởi 1 tile đầy đủ
    pad_h = (IMG_SIZE - H % IMG_SIZE) % IMG_SIZE + STRIDE
    pad_w = (IMG_SIZE - W % IMG_SIZE) % IMG_SIZE + STRIDE
    padded = np.pad(orig_np, ((STRIDE, pad_h), (STRIDE, pad_w), (0, 0)), mode="reflect")
    pH, pW = padded.shape[:2]

    # Gaussian weight window — trọng tâm cao, biên thấp → blend mượt
    def _gaussian_window(size: int) -> np.ndarray:
        sigma = size / 4.0
        ax    = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
        g1d   = np.exp(-0.5 * (ax / sigma) ** 2)
        g2d   = np.outer(g1d, g1d)
        return (g2d / g2d.max()).astype(np.float32)

    gauss_w = _gaussian_window(IMG_SIZE)  # [IMG_SIZE, IMG_SIZE]

    # Accumulators: logit scores weighted-sum + weight sum
    logit_acc  = np.zeros((pH, pW, NUM_CLASSES), dtype=np.float32)
    weight_acc = np.zeros((pH, pW),              dtype=np.float32)

    with torch.no_grad():
        for y in range(0, pH - IMG_SIZE + 1, STRIDE):
            for x in range(0, pW - IMG_SIZE + 1, STRIDE):
                patch = padded[y : y + IMG_SIZE, x : x + IMG_SIZE]
                inp   = _infer_transform(image=patch)["image"].unsqueeze(0).to(DEVICE)
                # Lấy logits (trước argmax) để blend
                logits = model(inp).squeeze(0).cpu().numpy()  # [C, H, W]
                logits = logits.transpose(1, 2, 0)            # [H, W, C]
                logit_acc[y : y + IMG_SIZE, x : x + IMG_SIZE] += logits * gauss_w[:, :, None]
                weight_acc[y : y + IMG_SIZE, x : x + IMG_SIZE] += gauss_w

    # Normalize và lấy argmax
    weight_acc = np.maximum(weight_acc, 1e-8)  # tránh chia 0
    blended    = logit_acc / weight_acc[:, :, None]
    pred_map   = blended.argmax(axis=-1).astype(np.uint8)

    # Cắt về kích thước gốc (bỏ padding STRIDE đã thêm)
    pred_map = pred_map[STRIDE : STRIDE + H, STRIDE : STRIDE + W]

    # ── 3. Tô màu theo COLOR_MAP ─────────────────────────────────────────────
    colored_np = COLOR_MAP[pred_map]                         # [H, W, 3] uint8
    overlay_np = (orig_np * 0.5 + colored_np * 0.5).astype(np.uint8)

    # ── 4. Encode sang PNG bytes ─────────────────────────────────────────────
    def _to_png(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    mask_bytes    = _to_png(colored_np)
    overlay_bytes = _to_png(overlay_np)

    # ── 5. Tính thống kê class ───────────────────────────────────────────────
    total_px = pred_map.size
    class_stats = [
        {
            "id"     : int(c),
            "name"   : CLASS_NAMES[c],
            "color"  : f"#{COLOR_MAP[c][0]:02x}{COLOR_MAP[c][1]:02x}{COLOR_MAP[c][2]:02x}",
            "percent": round(float((pred_map == c).sum() / total_px * 100), 2),
        }
        for c in range(NUM_CLASSES)
    ]

    return {
        "mask_image"   : mask_bytes,
        "overlay_image": overlay_bytes,
        "class_stats"  : class_stats,
    }
