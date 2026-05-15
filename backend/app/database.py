"""
database.py — PostgreSQL helper cho lịch sử phân đoạn.

Kết nối qua DATABASE_URL (env var), ví dụ:
  postgresql://postgres:password@localhost:5432/gan_segmentation

Giới hạn: tối đa 50 bản ghi gần nhất mỗi user (FIFO — xóa cũ nhất khi vượt quá).
"""

import psycopg2
import psycopg2.extras
import json
import base64
import io
import os
from datetime import datetime
from PIL import Image

# ── Cấu hình kết nối ──────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/gan_segmentation"
)
MAX_HISTORY = 50
THUMB_SIZE  = (220, 220)   # kích thước thumbnail dùng cho danh sách history


# ─────────────────────────────────────────────
# Khởi tạo bảng
# ─────────────────────────────────────────────
def init_db() -> None:
    with _conn() as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL PRIMARY KEY,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL DEFAULT 'user',
                is_premium    INTEGER NOT NULL DEFAULT 0,
                created_at    TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id             SERIAL PRIMARY KEY,
                user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                filename       TEXT    NOT NULL,
                created_at     TEXT    NOT NULL,
                original_thumb TEXT    NOT NULL,
                original_image TEXT    NOT NULL DEFAULT '',
                mask_image     TEXT    NOT NULL,
                overlay_image  TEXT    NOT NULL,
                class_stats    TEXT    NOT NULL
            )
        """)
        # Migration: thêm cột original_image nếu bảng cũ chưa có
        cur.execute("""
            ALTER TABLE history
            ADD COLUMN IF NOT EXISTS original_image TEXT NOT NULL DEFAULT ''
        """)
        con.commit()

        # Tạo admin mặc định nếu chưa có
        from app.auth import get_password_hash
        cur.execute("SELECT id FROM users WHERE username = 'admin'")
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO users (username, password_hash, role, is_premium, created_at) VALUES (%s, %s, %s, %s, %s)",
                ("admin", get_password_hash("admin123"), "admin", 1, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            con.commit()


# ─────────────────────────────────────────────
# Thao tác User
# ─────────────────────────────────────────────
def create_user(username: str, password_hash: str, role: str = "user") -> int | None:
    try:
        with _conn() as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (%s, %s, %s, %s) RETURNING id",
                (username, password_hash, role, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            new_id = cur.fetchone()[0]
            con.commit()
            return new_id
    except psycopg2.errors.UniqueViolation:
        return None  # Username đã tồn tại


def get_user_by_username(username: str) -> dict | None:
    with _conn() as con:
        cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, username, password_hash, role, is_premium FROM users WHERE username = %s",
            (username,)
        )
        row = cur.fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with _conn() as con:
        cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, username, role, is_premium FROM users WHERE id = %s",
            (user_id,)
        )
        row = cur.fetchone()
    return dict(row) if row else None


# ─────────────────────────────────────────────
# Admin: Quản lý tài khoản
# ─────────────────────────────────────────────
def list_all_users() -> list[dict]:
    """Liệt kê tất cả người dùng (dùng cho Admin)."""
    with _conn() as con:
        cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, username, role, is_premium, created_at FROM users ORDER BY id ASC"
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def update_user_status(user_id: int, role: str | None = None, is_premium: bool | None = None) -> bool:
    """Cập nhật role và/hoặc is_premium cho 1 user."""
    fields, values = [], []
    if role is not None:
        fields.append("role = %s")
        values.append(role)
    if is_premium is not None:
        fields.append("is_premium = %s")
        values.append(1 if is_premium else 0)
    if not fields:
        return False
    values.append(user_id)
    with _conn() as con:
        cur = con.cursor()
        cur.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = %s", values)
        affected = cur.rowcount
        con.commit()
    return affected > 0


def admin_delete_user(user_id: int) -> bool:
    """Xóa tài khoản người dùng (history tự xóa do ON DELETE CASCADE)."""
    with _conn() as con:
        cur = con.cursor()
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        affected = cur.rowcount
        con.commit()
    return affected > 0


# ─────────────────────────────────────────────
# Insert & prune
# ─────────────────────────────────────────────
def save_record(
    user_id: int,
    filename: str,
    original_bytes: bytes,
    mask_bytes: bytes,
    overlay_bytes: bytes,
    class_stats: list,
) -> int:
    thumb_b64    = _make_thumb_b64(original_bytes)                   # thumbnail nhỏ cho danh sách
    original_b64 = base64.b64encode(original_bytes).decode()         # ảnh gốc đầy đủ cho re-segment
    mask_b64     = base64.b64encode(mask_bytes).decode()
    overlay_b64  = base64.b64encode(overlay_bytes).decode()
    stats_json   = json.dumps(class_stats, ensure_ascii=False)
    created_at   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            """INSERT INTO history
                   (user_id, filename, created_at, original_thumb, original_image,
                    mask_image, overlay_image, class_stats)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (user_id, filename, created_at, thumb_b64, original_b64,
             mask_b64, overlay_b64, stats_json),
        )
        new_id = cur.fetchone()[0]
        # Giữ tối đa MAX_HISTORY bản ghi MỖI USER
        cur.execute(
            """DELETE FROM history WHERE user_id = %s AND id NOT IN (
                   SELECT id FROM history WHERE user_id = %s ORDER BY id DESC LIMIT %s
               )""",
            (user_id, user_id, MAX_HISTORY),
        )
        con.commit()
    return new_id


# ─────────────────────────────────────────────
# Queries
# ─────────────────────────────────────────────
def list_records(user_id: int) -> list[dict]:
    """Trả về tất cả records của 1 user, chỉ kèm thumbnail (không có mask/overlay đầy đủ)."""
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            """SELECT id, filename, created_at, original_thumb, class_stats
               FROM history WHERE user_id = %s ORDER BY id DESC""",
            (user_id,)
        )
        rows = cur.fetchall()
    return [
        {
            "id":             r[0],
            "filename":       r[1],
            "created_at":     r[2],
            "original_thumb": r[3],
            "class_stats":    json.loads(r[4]),
        }
        for r in rows
    ]


def get_record(record_id: int, user_id: int) -> dict | None:
    """Trả về 1 record đầy đủ (kèm ảnh gốc, mask + overlay) nếu thuộc về user."""
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            """SELECT id, filename, created_at, original_thumb, original_image,
                      mask_image, overlay_image, class_stats
               FROM history WHERE id = %s AND user_id = %s""",
            (record_id, user_id),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "id":             row[0],
        "filename":       row[1],
        "created_at":     row[2],
        "original_thumb": row[3],
        "original_image": row[4],   # ảnh gốc đầy đủ dùng cho re-segment
        "mask_image":     row[5],
        "overlay_image":  row[6],
        "class_stats":    json.loads(row[7]),
    }


def delete_record(record_id: int, user_id: int) -> bool:
    """Xóa 1 record của user. Trả về True nếu tìm thấy và xóa được."""
    with _conn() as con:
        cur = con.cursor()
        cur.execute("DELETE FROM history WHERE id = %s AND user_id = %s", (record_id, user_id))
        affected = cur.rowcount
        con.commit()
    return affected > 0


# ─────────────────────────────────────────────
# Đổi mật khẩu (dùng trực tiếp trong main.py)
# ─────────────────────────────────────────────
def get_user_password_hash(user_id: int) -> str | None:
    """Lấy password_hash của user theo id."""
    with _conn() as con:
        cur = con.cursor()
        cur.execute("SELECT password_hash FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
    return row[0] if row else None


def update_user_password(user_id: int, new_hash: str) -> bool:
    """Cập nhật password_hash cho user."""
    with _conn() as con:
        cur = con.cursor()
        cur.execute("UPDATE users SET password_hash = %s WHERE id = %s", (new_hash, user_id))
        affected = cur.rowcount
        con.commit()
    return affected > 0


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _conn() -> psycopg2.extensions.connection:
    con = psycopg2.connect(DATABASE_URL)
    return con


def _make_thumb_b64(image_bytes: bytes) -> str:
    """Tạo thumbnail PNG (THUMB_SIZE) từ raw image bytes, trả về base64."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail(THUMB_SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
