# 🛰️ GAN Satellite Segmentation

Ứng dụng phân đoạn ảnh vệ tinh sử dụng mô hình **DeepLabV3+ ResNet50** (Generator từ kiến trúc GAN) được huấn luyện trên dataset **LoveDA**. Hệ thống hỗ trợ phân đoạn 7 lớp đất đai, với giao diện web đầy đủ tính năng và hệ thống xác thực người dùng.

---

## 📸 Demo

| Ảnh vệ tinh đầu vào | Mask phân đoạn | Overlay |
|---|---|---|
| ![input](training_history.png) | *(mask)* | *(overlay)* |

---

## 🗂️ Cấu trúc dự án

```
GAN-segmentation/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py             # API endpoints
│   │   ├── model.py            # Load model & inference
│   │   ├── database.py         # PostgreSQL (psycopg2)
│   │   ├── auth.py             # JWT authentication
│   │   └── utils.py            # Tiện ích
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                   # React + Vite frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx   # Trang phân đoạn chính
│   │   │   ├── Admin.jsx       # Trang quản trị
│   │   │   ├── Login.jsx       # Đăng nhập
│   │   │   └── Register.jsx    # Đăng ký
│   │   ├── components/         # Các component dùng chung
│   │   ├── App.jsx
│   │   └── AuthContext.jsx     # Quản lý trạng thái đăng nhập
│   ├── Dockerfile
│   └── package.json
├── models/                     # Thư mục chứa model weights
├── segGAN.ipynb                # Notebook huấn luyện GAN gốc
├── segGAN_improved.py          # Script huấn luyện cải tiến (SSH-friendly)
├── preprocess.ipynb            # Notebook tiền xử lý LoveDA
├── unet.ipynb                  # Thử nghiệm với UNet
├── run_eval.py                 # Script đánh giá mô hình
├── eval_results.json           # Kết quả đánh giá
├── docker-compose.yml          # Orchestration: DB + Backend + Frontend
└── .gitignore
```

---

## 🧠 Kiến trúc mô hình

| Thành phần | Kiến trúc |
|---|---|
| **Generator** | DeepLabV3+ với backbone ResNet50 |
| **Discriminator** | PatchGAN (70×70) |
| **Loss** | BCE (Adversarial) + Cross-Entropy (Segmentation) |
| **Dataset** | LoveDA (Urban + Rural) |
| **Input size** | 512×512 px |

### 📊 Kết quả đánh giá (trên tập Val)

| Class | IoU | Precision | Recall | F1 |
|---|---|---|---|---|
| Background | 0.5384 | 0.6583 | 0.7473 | 0.7000 |
| **Building** | **0.6397** | 0.7044 | 0.8745 | 0.7803 |
| Road | 0.5435 | 0.6925 | 0.7164 | 0.7042 |
| **Water** | **0.6832** | 0.7918 | 0.8328 | 0.8118 |
| Barren | 0.3145 | 0.5338 | 0.4336 | 0.4785 |
| Forest | 0.4343 | 0.5720 | 0.6434 | 0.6056 |
| Agricultural | 0.5543 | 0.8484 | 0.6152 | 0.7132 |
| **Mean** | **0.5297** | **0.6859** | **0.6948** | **0.6848** |

**Pixel Accuracy:** 70.7%

---

## 🚀 Hướng dẫn chạy (Docker Compose)

### Yêu cầu

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (đã cài và đang chạy)
- Model weights: `last_generator.pth` (đặt ở thư mục gốc)

### Các bước chạy

```bash
# 1. Clone repo
git clone https://github.com/kamusarj/GAN-segmentation.git
cd GAN-segmentation

# 2. Đặt file model weight vào thư mục gốc
# (Tải last_generator.pth từ nguồn của bạn)

# 3. Khởi động toàn bộ hệ thống
docker compose up --build -d

# 4. Truy cập ứng dụng
# Frontend: http://localhost:3000
# Backend API docs: http://localhost:8000/docs
```

### Dừng hệ thống

```bash
docker compose down
```

---

## 🔌 API Endpoints

Base URL: `http://localhost:8000`

| Method | Endpoint | Mô tả | Auth |
|---|---|---|---|
| `GET` | `/health` | Kiểm tra trạng thái server | — |
| `POST` | `/api/auth/register` | Đăng ký tài khoản mới | — |
| `POST` | `/api/auth/login` | Đăng nhập, nhận JWT token | — |
| `GET` | `/api/me` | Lấy thông tin cá nhân | JWT |
| `POST` | `/api/me/change-password` | Đổi mật khẩu | JWT |
| `POST` | `/api/predict` | Phân đoạn ảnh vệ tinh | JWT |
| `GET` | `/api/classes` | Danh sách 7 class LoveDA | — |
| `GET` | `/api/history` | Lịch sử phân đoạn của user | JWT |
| `GET` | `/api/history/{id}` | Chi tiết 1 bản ghi lịch sử | JWT |
| `DELETE` | `/api/history/{id}` | Xóa 1 bản ghi lịch sử | JWT |
| `GET` | `/api/models` | Danh sách model weights | — |
| `POST` | `/api/models/switch` | Đổi model đang dùng | Admin |
| `GET` | `/api/admin/users` | Danh sách tất cả user | Admin |
| `PATCH` | `/api/admin/users/{id}` | Cập nhật role/premium | Admin |
| `DELETE` | `/api/admin/users/{id}` | Xóa tài khoản user | Admin |

---

## 🖥️ Tính năng Frontend

- **Dashboard:** Upload ảnh vệ tinh → hiển thị mask, overlay, thống kê từng lớp
- **Lịch sử:** Xem lại các lần phân đoạn trước, re-segment, điều chỉnh opacity/màu
- **Bản đồ:** Chọn vùng ảnh từ bản đồ tương tác (MapSelector)
- **Kiểm soát nâng cao:** Unsharp mask preprocessing, color adjustment (AdvancedControls)
- **Admin Panel:** Quản lý user, chuyển đổi model weights
- **Xác thực:** JWT, phân quyền User / Admin / Premium

---

## 🏷️ 7 Lớp phân đoạn LoveDA

| ID | Class | Màu |
|---|---|---|
| 0 | Background | ⬛ |
| 1 | Building | 🟥 |
| 2 | Road | ⬜ |
| 3 | Water | 🟦 |
| 4 | Barren | 🟫 |
| 5 | Forest | 🟩 |
| 6 | Agricultural | 🟨 |

---

## ⚙️ Phát triển cục bộ (không dùng Docker)

### Backend

```bash
cd backend
pip install -r requirements.txt

# Cần PostgreSQL đang chạy
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/gan_segmentation

uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## 🏋️ Huấn luyện lại mô hình

```bash
# Cài đặt môi trường train (cần GPU)
pip install torch torchvision segmentation-models-pytorch albumentations matplotlib

# Chạy script cải tiến (tương thích SSH, không cần display)
python segGAN_improved.py

# Đánh giá mô hình sau khi train
python run_eval.py
```

> **Lưu ý:** Dataset LoveDA (`Train/`, `Val/`, `Test/`) không được push lên GitHub do kích thước lớn. Tải từ [LoveDA Official](https://github.com/Junjue-Wang/LoveDA).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Model** | PyTorch, segmentation-models-pytorch |
| **Backend** | FastAPI, Uvicorn |
| **Database** | PostgreSQL (psycopg2) |
| **Auth** | JWT (PyJWT), bcrypt (passlib) |
| **Frontend** | React 18, Vite, TailwindCSS |
| **Container** | Docker, Docker Compose |

---

## 📄 License

MIT License — © 2025 Bui Hoang Linh
