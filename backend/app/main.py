import base64
import io
import logging
from contextlib import asynccontextmanager

from PIL import Image as PILImage

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os

from app.model    import load_model, predict, CLASS_NAMES, COLOR_MAP, get_active_model_path
from app.database import (
    init_db, save_record, list_records, get_record, delete_record,
    create_user, get_user_by_username, get_user_by_id,
    list_all_users, update_user_status, admin_delete_user,
    get_user_password_hash, update_user_password,
)
from app.auth     import verify_password, get_password_hash, create_access_token, get_current_user_id, require_admin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lifecycle: load model + init DB ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo bảng PostgreSQL
    try:
        init_db()
        logger.info("✅ Database initialized.")
    except Exception as e:
        logger.error(f"⚠️  DB init error: {e}")

    # Load model
    try:
        load_model("/code/last_generator.pth")
    except FileNotFoundError as e:
        logger.error(f"⚠️  {e}")
        logger.warning("Server vẫn chạy nhưng /api/predict sẽ trả lỗi 503 cho đến khi model được mount.")
    yield
    logger.info("Shutting down…")


app = FastAPI(
    title       = "GAN Satellite Segmentation API",
    version     = "2.1",
    description = "DeepLabV3+ ResNet50 Generator — phân đoạn ảnh vệ tinh LoveDA",
    lifespan    = lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "model": "DeepLabV3+ ResNet50"}


# ── Xác thực & Phân quyền (Auth) ─────────────────────────────────────────────
class UserRegister(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

@app.post("/api/auth/register", tags=["Auth"])
def register_user(user: UserRegister):
    hashed_pw = get_password_hash(user.password)
    user_id = create_user(user.username, hashed_pw, role="user")
    if user_id is None:
        raise HTTPException(status_code=400, detail="Username đã tồn tại.")
    return JSONResponse({"message": "Đăng ký thành công", "user_id": user_id})

@app.post("/api/auth/login", tags=["Auth"])
def login_user(user: UserLogin):
    db_user = get_user_by_username(user.username)
    if not db_user or not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(status_code=401, detail="Sai username hoặc password.")
    
    token = create_access_token({"sub": str(db_user["id"]), "username": db_user["username"], "role": db_user["role"], "is_premium": bool(db_user.get("is_premium", 0))})
    return JSONResponse({
        "access_token": token,
        "token_type": "bearer",
        "username": db_user["username"],
        "role": db_user["role"],
        "is_premium": bool(db_user.get("is_premium", 0))
    })


# ── Quản lý Model (Admin) ───────────────────────────────────────────────────
class ModelSwitchReq(BaseModel):
    model_name: str

@app.get("/api/models", tags=["Model"])
def list_available_models():
    """Liệt kê các file .pth trong thư mục /code/models/ (và file mặc định)."""
    models = ["/code/last_generator.pth"]
    models_dir = "/code/models"
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".pth"):
                models.append(os.path.join(models_dir, f))
    active = get_active_model_path()
    return {"models": models, "active_model": active}

@app.post("/api/models/switch", tags=["Model"])
def switch_model(req: ModelSwitchReq, admin_id: int = Depends(require_admin)):
    """Đổi model đang chạy (Chỉ Admin)."""
    if not os.path.exists(req.model_name):
        raise HTTPException(status_code=404, detail="File model không tồn tại.")
    try:
        load_model(req.model_name)
        return {"message": f"Đã chuyển sang model {os.path.basename(req.model_name)}"}
    except Exception as e:
        logger.error(f"Switch model error: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi load model: {e}")


# ── Admin: Quản lý người dùng ─────────────────────────────────────────────────
class UserUpdateReq(BaseModel):
    role: str | None = None
    is_premium: bool | None = None

@app.get("/api/admin/users", tags=["Admin"])
def admin_list_users(admin_id: int = Depends(require_admin)):
    """Liệt kê tất cả tài khoản (Chỉ Admin)."""
    users = list_all_users()
    # Không trả về password_hash
    return users

@app.patch("/api/admin/users/{user_id}", tags=["Admin"])
def admin_update_user(user_id: int, req: UserUpdateReq, admin_id: int = Depends(require_admin)):
    """Cập nhật role hoặc quyền Premium cho 1 user (Chỉ Admin)."""
    if user_id == admin_id:
        raise HTTPException(status_code=400, detail="Không thể tự thay đổi quyền của chính mình.")
    if req.role is not None and req.role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="Role không hợp lệ. Chỉ chấp nhận 'user' hoặc 'admin'.")
    ok = update_user_status(user_id, role=req.role, is_premium=req.is_premium)
    if not ok:
        raise HTTPException(status_code=404, detail="Không tìm thấy user.")
    return {"message": "Cập nhật thành công."}

@app.delete("/api/admin/users/{user_id}", tags=["Admin"])
def admin_remove_user(user_id: int, admin_id: int = Depends(require_admin)):
    """Xóa tài khoản người dùng và lịch sử của họ (Chỉ Admin)."""
    if user_id == admin_id:
        raise HTTPException(status_code=400, detail="Không thể xóa chính tài khoản Admin đang đăng nhập.")
    ok = admin_delete_user(user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Không tìm thấy user.")
    return {"message": "Đã xóa tài khoản thành công."}


# ── Thông tin cá nhân & đổi mật khẩu ─────────────────────────────────────────
@app.get("/api/me", tags=["Profile"])
def get_my_profile(user_id: int = Depends(get_current_user_id)):
    """Lấy thông tin tài khoản đang đăng nhập."""
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Tài khoản không tồn tại.")
    return {
        "id"        : user["id"],
        "username"  : user["username"],
        "role"      : user["role"],
        "is_premium": bool(user.get("is_premium", 0)),
    }

class ChangePasswordReq(BaseModel):
    current_password: str
    new_password    : str

@app.post("/api/me/change-password", tags=["Profile"])
def change_my_password(req: ChangePasswordReq, user_id: int = Depends(get_current_user_id)):
    """Đổi mật khẩu cho tài khoản đang đăng nhập."""
    current_hash = get_user_password_hash(user_id)
    if not current_hash:
        raise HTTPException(status_code=404, detail="Tài khoản không tồn tại.")
    if not verify_password(req.current_password, current_hash):
        raise HTTPException(status_code=400, detail="Mật khẩu hiện tại không đúng.")
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="Mật khẩu mới phải có ít nhất 8 ký tự.")
    hashed = get_password_hash(req.new_password)
    update_user_password(user_id, hashed)
    return {"message": "Đổi mật khẩu thành công!"}


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


# ── Cấu hình validate ảnh đầu vào ──────────────────────────────────────────────
MIN_SIZE      = 256          # px — cạnh tối thiểu
MAX_SIZE      = 20_000       # px — cạnh tối đa
MAX_RATIO     = 4.0          # tỉ lệ w/h hoặc h/w tối đa
OPTIMAL_MIN   = 512          # kích thước lý tưởng tối thiểu

def _validate_image(image_bytes: bytes) -> tuple[tuple[int,int], list[str]]:
    """
    Kiểm tra ảnh đầu vào.
    Returns: (width, height), danh sách cảnh báo (có thể rỗng).
    Raises HTTPException nếu ảnh hoàn toàn không phù hợp.
    """
    try:
        img = PILImage.open(io.BytesIO(image_bytes))
        w, h = img.size
    except Exception:
        raise HTTPException(status_code=400, detail="Không thể đọc file ảnh. Hãy đảm bảo file là PNG, JPEG hoặc TIFF hợp lệ.")

    warnings: list[str] = []

    # (1) Kích thước quá nhỏ — từ chối hẻ
    if w < MIN_SIZE or h < MIN_SIZE:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Ảnh quá nhỏ ({w}×{h} px). "
                f"Kích thước tối thiểu là {MIN_SIZE}×{MIN_SIZE} px. "
                "Những ảnh nhỏ như ảnh chụp màn hình, thumbnail, icon... "
                "không phù hợp với mô hình phân đoạn vệ tinh."
            ),
        )

    # (2) Kích thước quá lớn — cảnh báo (vẫn xử lý được do sliding window)
    if w > MAX_SIZE or h > MAX_SIZE:
        warnings.append(
            f"Ảnh rất lớn ({w}×{h} px) — thời gian xử lý có thể lâu hơn."
        )

    # (3) Tỉ lệ khung hình lệch quá — cảnh báo
    ratio = max(w, h) / min(w, h)
    if ratio > MAX_RATIO:
        warnings.append(
            f"Ảnh có tỉ lệ khung hình bất thường ({w}×{h}, ratio {ratio:.1f}:1). "
            "Ảnh vệ tinh thường có tỉ lệ gần vuông — kết quả có thể không chính xác."
        )

    # (4) Ảnh nhỏ hơn kích thước lý tưởng
    if w < OPTIMAL_MIN or h < OPTIMAL_MIN:
        warnings.append(
            f"Ảnh khá nhỏ ({w}×{h} px). Đề đạt kết quả tốt nhất, nên dùng ảnh ≥ {OPTIMAL_MIN}×{OPTIMAL_MIN} px."
        )

    # (5) Ảnh grayscale — mô hình train trên RGB
    if img.mode not in ("RGB", "RGBA"):
        warnings.append(
            f"Ảnh đang ở chế độ màu {img.mode} (không phải RGB). "
            "Mô hình được huấn luyện trên ảnh RGB — kết quả có thể không chính xác."
        )

    return (w, h), warnings


# ── Predict endpoint ──────────────────────────────────────────────────────────
@app.post("/api/predict", tags=["Inference"])
async def predict_segmentation(image: UploadFile = File(...), user_id: int = Depends(get_current_user_id)):
    """
    Nhận ảnh vệ tinh, trả về mask, overlay và stats. (Yêu cầu đăng nhập)
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File upload phải là ảnh (PNG / JPEG / TIFF).")

    try:
        image_bytes = await image.read()

        # ── Validate ảnh: kích thước, tỉ lệ, mode màu ────────────────────────
        _size, input_warnings = _validate_image(image_bytes)
        # ─────────────────────────────────────────────────────────────────────

        result = predict(image_bytes)

        # ── Lưu lịch sử ──────────────────────────────────────────────────────
        try:
            filename = image.filename or "unknown.png"
            save_record(
                user_id       = user_id,
                filename      = filename,
                original_bytes= image_bytes,
                mask_bytes    = result["mask_image"],
                overlay_bytes = result["overlay_image"],
                class_stats   = result["class_stats"],
            )
        except Exception as db_err:
            logger.warning(f"Không thể lưu lịch sử: {db_err}")
        # ─────────────────────────────────────────────────────────────────────

        return JSONResponse({
            "mask_image"   : base64.b64encode(result["mask_image"]).decode(),
            "overlay_image": base64.b64encode(result["overlay_image"]).decode(),
            "raw_mask"     : base64.b64encode(result["raw_mask"]).decode(),
            "class_stats"  : result["class_stats"],
            "warnings"     : input_warnings,   # [] nếu ảnh hợp lệ
        })

    except RuntimeError as e:
        logger.error(f"Model error: {e}")
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng. Kiểm tra volume mount của last_generator.pth.")

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý ảnh: {str(e)}")


# ── History endpoints ─────────────────────────────────────────────────────────
@app.get("/api/history", tags=["History"])
def get_history(user_id: int = Depends(get_current_user_id)):
    """Trả về danh sách lịch sử của user (mới nhất trước)."""
    try:
        return JSONResponse(list_records(user_id))
    except Exception as e:
        logger.error(f"History list error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Không thể tải lịch sử.")


@app.get("/api/history/{record_id}", tags=["History"])
def get_history_detail(record_id: int, user_id: int = Depends(get_current_user_id)):
    """Trả về 1 bản ghi đầy đủ của user."""
    record = get_record(record_id, user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi.")
    return JSONResponse(record)


@app.delete("/api/history/{record_id}", tags=["History"])
def delete_history(record_id: int, user_id: int = Depends(get_current_user_id)):
    """Xóa 1 bản ghi khỏi lịch sử của user."""
    deleted = delete_record(record_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi.")
    return JSONResponse({"message": "Đã xóa.", "id": record_id})
