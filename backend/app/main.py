import base64
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.model import load_model, predict, CLASS_NAMES, COLOR_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lifecycle: load model 1 lần khi khởi động ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model("/code/last_generator.pth")
    except FileNotFoundError as e:
        logger.error(f"⚠️  {e}")
        logger.warning("Server vẫn chạy nhưng /api/predict sẽ trả lỗi 503 cho đến khi model được mount.")
    yield   # server đang chạy
    logger.info("Shutting down…")


app = FastAPI(
    title       = "GAN Satellite Segmentation API",
    version     = "2.0",
    description = "DeepLabV3+ ResNet50 Generator — phân đoạn ảnh vệ tinh LoveDA",
    lifespan    = lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Trên production: thay bằng domain cụ thể
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "model": "DeepLabV3+ ResNet50"}


# ── Thông tin classes ─────────────────────────────────────────────────────────
@app.get("/api/classes", tags=["Info"])
def get_classes():
    """Trả về danh sách 7 class LoveDA và màu tương ứng."""
    return [
        {
            "id"   : i,
            "name" : CLASS_NAMES[i],
            "color": f"#{COLOR_MAP[i][0]:02x}{COLOR_MAP[i][1]:02x}{COLOR_MAP[i][2]:02x}",
        }
        for i in range(len(CLASS_NAMES))
    ]


# ── Predict endpoint ────────────────────────────────────────────────────────
@app.post("/api/predict", tags=["Inference"])
async def predict_segmentation(image: UploadFile = File(...)):
    """
    Nhận ảnh vệ tinh, trả về:
    - mask_image   : ảnh PNG tô màu theo class (base64)
    - overlay_image: ảnh PNG overlay ảnh gốc + mask (base64)
    - class_stats  : tỉ lệ % từng class trong ảnh
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File upload phải là ảnh (PNG / JPEG / TIFF).")

    try:
        image_bytes = await image.read()
        result      = predict(image_bytes)

        return JSONResponse({
            "mask_image"   : base64.b64encode(result["mask_image"]).decode(),
            "overlay_image": base64.b64encode(result["overlay_image"]).decode(),
            "class_stats"  : result["class_stats"],
        })

    except RuntimeError as e:
        # Model chưa load (file không tìm thấy lúc startup)
        logger.error(f"Model error: {e}")
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng. Kiểm tra volume mount của last_generator.pth.")

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý ảnh: {str(e)}")
