"""
preprocess.py — Tiền xử lý và Visualize dữ liệu LoveDA
=======================================================
Gồm 2 phần:

  PHẦN 1 — Crop dataset (từ preprocess.ipynb):
    Cắt ảnh gốc LoveDA (Train & Val) thành các patch 512×512
    và lưu vào thư mục LoveDA_patch/.

  PHẦN 2 — Visualize dữ liệu gốc (TRƯỚC tiền xử lý):
    1. visualize_raw_samples()       — Ảnh mẫu + mask + overlay + lưới patch
    2. visualize_class_distribution() — Phân bố lớp Rural vs Urban
    3. visualize_image_sizes()        — Histogram kích thước ảnh gốc
"""

# ── Imports ────────────────────────────────────────────────────────
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════════════════════════
# PHẦN 1 — Crop dataset
# ══════════════════════════════════════════════════════════════════

PATCH_SIZE = 512


def crop_dataset(input_root, output_root):
    """
    Cắt tất cả ảnh và mask trong input_root thành các patch 512×512
    (non-overlapping, bỏ qua patch biên không đủ kích thước).

    Cấu trúc input_root:
        input_root/
          Rural/
            images_png/  *.png
            masks_png/   *.png
          Urban/
            images_png/  *.png
            masks_png/   *.png

    Cấu trúc output_root:
        output_root/
          images/  <name>_<idx>.png
          masks/   <name>_<idx>.png
    """
    areas    = ["Rural", "Urban"]
    img_out  = os.path.join(output_root, "images")
    mask_out = os.path.join(output_root, "masks")

    os.makedirs(img_out,  exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    total_patches = 0

    for area in areas:
        img_dir  = os.path.join(input_root, area, "images_png")
        mask_dir = os.path.join(input_root, area, "masks_png")

        if not os.path.isdir(img_dir):
            print(f"[WARN] Không tìm thấy: {img_dir}")
            continue

        names = sorted(os.listdir(img_dir))
        print(f"[{area}] Xử lý {len(names)} ảnh...")

        for name in names:
            img_path  = os.path.join(img_dir,  name)
            mask_path = os.path.join(mask_dir, name)

            image = cv2.imread(img_path)
            mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"  [WARN] Không đọc được: {name}")
                continue

            h, w = image.shape[:2]
            idx  = 0

            for y in range(0, h, PATCH_SIZE):
                for x in range(0, w, PATCH_SIZE):
                    img_patch  = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                    mask_patch = mask [y:y + PATCH_SIZE, x:x + PATCH_SIZE]

                    # Bỏ qua patch biên không đủ kích thước
                    if img_patch.shape[0] != PATCH_SIZE:
                        continue
                    if img_patch.shape[1] != PATCH_SIZE:
                        continue

                    out_name = name.replace(".png", f"_{idx}.png")
                    cv2.imwrite(os.path.join(img_out,  out_name), img_patch)
                    cv2.imwrite(os.path.join(mask_out, out_name), mask_patch)
                    idx += 1

            total_patches += idx

    print(f"\n✅ Hoàn tất! Tổng số patch: {total_patches}")
    print(f"   Lưu tại: {os.path.abspath(output_root)}")


# ══════════════════════════════════════════════════════════════════
# PHẦN 2 — Visualize dữ liệu gốc (TRƯỚC tiền xử lý)
# ══════════════════════════════════════════════════════════════════

# Nhãn trong file mask LoveDA gốc: 1-7 (không phải 0-6 như khi train)
# 0 và 255 = vùng ignore → map về 0 khi visualize
RAW_LABEL_NAMES = {
    1: "Background",
    2: "Building",
    3: "Road",
    4: "Water",
    5: "Barren",
    6: "Forest",
    7: "Agricultural",
}

# Color map LoveDA official (dùng lại, shift 1-7 → 0-6)
_BASE_COLORS = np.array([
    [255, 255, 255],   # 0: Background
    [255,   0,   0],   # 1: Building
    [255, 255,   0],   # 2: Road
    [  0,   0, 255],   # 3: Water
    [159, 129, 183],   # 4: Barren
    [  0, 255,   0],   # 5: Forest
    [255, 195, 128],   # 6: Agricultural
], dtype=np.uint8)

RAW_COLOR_MAP = np.zeros((8, 3), dtype=np.uint8)
RAW_COLOR_MAP[0] = [200, 200, 200]          # ignore → xám nhạt
for _i in range(1, 8):
    RAW_COLOR_MAP[_i] = _BASE_COLORS[_i - 1]


# ── Helper ─────────────────────────────────────────────────────────
def _read_raw_pair(img_path, mask_path):
    """Đọc ảnh RGB và mask gốc (nhãn 1-7; 0/255 → ignore)."""
    img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    mask[mask == 255] = 0
    mask = np.clip(mask, 0, 7)
    return img, mask


def _count_class_pixels(mask_dir, max_files=None):
    """
    Đếm pixel mỗi lớp (0-7) trong toàn bộ thư mục mask.
    Index 0 = ignore (bỏ qua khi vẽ).
    """
    counts = np.zeros(8, dtype=np.int64)
    files  = sorted(os.listdir(mask_dir))
    if max_files:
        files = files[:max_files]
    for fn in files:
        mask = cv2.imread(os.path.join(mask_dir, fn),
                          cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        mask[mask == 255] = 0
        mask = np.clip(mask, 0, 7)
        for c in range(8):
            counts[c] += (mask == c).sum()
    return counts


# ── Hàm 1: Hiển thị ảnh mẫu từ dataset gốc ───────────────────────
def visualize_raw_samples(raw_root, n_per_area=2, patch_size=512,
                          save_path=None):
    """
    Hiển thị n_per_area mẫu từ mỗi khu vực (Rural, Urban).
    Mỗi hàng: [Ảnh gốc + lưới patch] [Mask] [Overlay α=0.5] [Bar chart lớp]

    Args:
        raw_root   : thư mục chứa Rural/ và Urban/ (vd: "./Train/Train")
        n_per_area : số ảnh lấy mỗi khu vực (mặc định 2)
        patch_size : kích thước ô lưới vàng overlay (mặc định 512)
        save_path  : lưu figure nếu truyền vào
    """
    samples = []

    for area in ["Rural", "Urban"]:
        img_dir  = os.path.join(raw_root, area, "images_png")
        mask_dir = os.path.join(raw_root, area, "masks_png")
        if not os.path.isdir(img_dir):
            print(f"[WARN] Không tìm thấy: {img_dir}")
            continue
        for name in sorted(os.listdir(img_dir))[:n_per_area]:
            img, mask = _read_raw_pair(
                os.path.join(img_dir,  name),
                os.path.join(mask_dir, name),
            )
            samples.append((area, img, mask, name))

    n_rows = len(samples)
    if n_rows == 0:
        print("Không tìm thấy ảnh nào. Kiểm tra lại raw_root.")
        return

    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 5.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        f"Ảnh gốc (lưới vàng = patch {patch_size}×{patch_size})",
        "Mask Ground Truth (nhãn 1-7)",
        "Overlay (α = 0.5)",
        "Phân bố lớp (%)",
    ]
    for col, t in enumerate(col_titles):
        axes[0, col].set_title(t, fontsize=11, fontweight="bold", pad=8)

    for row, (area, img, mask, name) in enumerate(samples):
        H, W    = img.shape[:2]
        colored = RAW_COLOR_MAP[mask]
        overlay = (img * 0.5 + colored * 0.5).astype(np.uint8)

        total_valid  = max((mask > 0).sum(), 1)
        class_ratios = [(mask == c).sum() / total_valid * 100
                        for c in range(1, 8)]

        # Cột 0: ảnh gốc + lưới
        axes[row, 0].imshow(img)
        for y_line in range(0, H, patch_size):
            axes[row, 0].axhline(y_line, color="yellow", lw=0.8, alpha=0.7)
        for x_line in range(0, W, patch_size):
            axes[row, 0].axvline(x_line, color="yellow", lw=0.8, alpha=0.7)
        axes[row, 0].set_ylabel(
            f"[{area}]\n{name}\n{W}×{H} px",
            fontsize=9, rotation=0, labelpad=130, va="center",
        )
        axes[row, 0].axis("off")

        # Cột 1: mask tô màu
        axes[row, 1].imshow(colored)
        axes[row, 1].axis("off")

        # Cột 2: overlay
        axes[row, 2].imshow(overlay)
        axes[row, 2].axis("off")

        # Cột 3: bar chart ngang
        ax_b = axes[row, 3]
        bar_colors = [RAW_COLOR_MAP[c] / 255.0 for c in range(1, 8)]
        bars = ax_b.barh(range(7), class_ratios,
                         color=bar_colors, edgecolor="grey", lw=0.5)
        ax_b.set_yticks(range(7))
        ax_b.set_yticklabels(list(RAW_LABEL_NAMES.values()), fontsize=9)
        ax_b.set_xlabel("Tỉ lệ (%)", fontsize=9)
        ax_b.invert_yaxis()
        ax_b.grid(axis="x", alpha=0.3)
        for bar, r in zip(bars, class_ratios):
            if r > 1.5:
                ax_b.text(bar.get_width() + 0.3,
                          bar.get_y() + bar.get_height() / 2,
                          f"{r:.1f}%", va="center", fontsize=8)

    legend_patches = [
        mpatches.Patch(color=RAW_COLOR_MAP[c] / 255.0,
                       label=f"{c}: {RAW_LABEL_NAMES[c]}")
        for c in range(1, 8)
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=7,
               fontsize=9, framealpha=0.9, edgecolor="#ccc",
               bbox_to_anchor=(0.45, -0.02))

    plt.suptitle(
        f"Dữ liệu LoveDA gốc — TRƯỚC khi tiền xử lý\n"
        f"(lưới vàng = vùng crop patch {patch_size}×{patch_size})",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Đã lưu → {save_path}")
    plt.show()


# ── Hàm 2: So sánh phân bố lớp Rural vs Urban ────────────────────
def visualize_class_distribution(raw_root, max_files_per_area=None,
                                 save_path=None):
    """
    Tính tổng pixel mỗi lớp trong Rural và Urban (riêng & tổng hợp),
    vẽ 3 biểu đồ cột + đường so sánh.

    Args:
        raw_root           : thư mục chứa Rural/ và Urban/
        max_files_per_area : giới hạn số file đọc (None = đọc hết)
        save_path          : lưu figure nếu truyền vào
    """
    print("Đang đếm pixel theo lớp (có thể mất vài phút)...")
    counts = {}
    for area in ["Rural", "Urban"]:
        mask_dir = os.path.join(raw_root, area, "masks_png")
        if os.path.isdir(mask_dir):
            counts[area] = _count_class_pixels(mask_dir, max_files_per_area)
            print(f"  {area}: xong ({len(os.listdir(mask_dir))} file)")
        else:
            counts[area] = np.zeros(8, dtype=np.int64)
            print(f"  [WARN] Không tìm thấy {mask_dir}")

    labels     = list(RAW_LABEL_NAMES.values())
    bar_colors = [RAW_COLOR_MAP[c] / 255.0 for c in range(1, 8)]
    x = np.arange(7)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "Phân bố lớp địa thực vật — Dữ liệu LoveDA gốc\n"
        "(tính trên tổng số pixel, không kể vùng ignore)",
        fontsize=13, fontweight="bold",
    )

    def _draw(ax, title, cnt):
        total  = max(cnt[1:].sum(), 1)
        ratios = cnt[1:] / total * 100
        bars   = ax.bar(x, ratios, color=bar_colors,
                        edgecolor="grey", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
        ax.set_ylabel("Tỉ lệ (%)")
        ax.set_ylim(0, min(ratios.max() * 1.25 + 1, 100))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, r in zip(bars, ratios):
            if r > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.4,
                        f"{r:.1f}%", ha="center", va="bottom", fontsize=8)

    _draw(axes[0], "Rural", counts.get("Rural", np.zeros(8)))
    _draw(axes[1], "Urban", counts.get("Urban", np.zeros(8)))

    combined = sum(counts.values())
    _draw(axes[2], "Rural + Urban (tổng hợp)", combined)

    # Đường so sánh trên panel tổng hợp
    r_tot   = max(counts.get("Rural", np.zeros(8))[1:].sum(), 1)
    u_tot   = max(counts.get("Urban", np.zeros(8))[1:].sum(), 1)
    r_ratio = counts.get("Rural", np.zeros(8))[1:] / r_tot * 100
    u_ratio = counts.get("Urban", np.zeros(8))[1:] / u_tot * 100
    axes[2].plot(x, r_ratio, "o--", color="saddlebrown",
                 lw=1.5, ms=5, label="Rural (%)")
    axes[2].plot(x, u_ratio, "s--", color="steelblue",
                 lw=1.5, ms=5, label="Urban (%)")
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Đã lưu → {save_path}")
    plt.show()


# ── Hàm 3: Histogram kích thước ảnh gốc ─────────────────────────
def visualize_image_sizes(raw_root, save_path=None):
    """
    Đọc kích thước (W, H) của tất cả ảnh gốc trong Rural + Urban,
    vẽ scatter plot và histogram phân bố kích thước.

    Args:
        raw_root  : thư mục chứa Rural/ và Urban/
        save_path : lưu figure nếu truyền vào
    """
    widths, heights, areas_list = [], [], []

    for area in ["Rural", "Urban"]:
        img_dir = os.path.join(raw_root, area, "images_png")
        if not os.path.isdir(img_dir):
            continue
        for fn in sorted(os.listdir(img_dir)):
            img = cv2.imread(os.path.join(img_dir, fn))
            if img is None:
                continue
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            areas_list.append(area)

    if not widths:
        print("Không đọc được ảnh nào. Kiểm tra lại raw_root.")
        return

    widths    = np.array(widths)
    heights   = np.array(heights)
    areas_arr = np.array(areas_list)

    colors_area = {"Rural": "saddlebrown", "Urban": "steelblue"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Phân bố kích thước ảnh gốc LoveDA — TRƯỚC tiền xử lý",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: Scatter W vs H
    for area in ["Rural", "Urban"]:
        m = areas_arr == area
        axes[0].scatter(widths[m], heights[m],
                        c=colors_area[area], alpha=0.7, s=40,
                        label=area, edgecolors="grey", lw=0.3)
    axes[0].axvline(PATCH_SIZE, color="red", lw=1, ls="--",
                    label=f"{PATCH_SIZE} px")
    axes[0].axhline(PATCH_SIZE, color="red", lw=1, ls="--")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Height (px)")
    axes[0].set_title("Scatter: Width vs Height")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Panel 2: Histogram Width
    for area in ["Rural", "Urban"]:
        m = areas_arr == area
        axes[1].hist(widths[m], bins=20, alpha=0.6,
                     color=colors_area[area], edgecolor="grey",
                     lw=0.5, label=area)
    axes[1].set_xlabel("Width (px)")
    axes[1].set_ylabel("Số ảnh")
    axes[1].set_title("Histogram chiều rộng (Width)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # Panel 3: Histogram Height
    for area in ["Rural", "Urban"]:
        m = areas_arr == area
        axes[2].hist(heights[m], bins=20, alpha=0.6,
                     color=colors_area[area], edgecolor="grey",
                     lw=0.5, label=area)
    axes[2].set_xlabel("Height (px)")
    axes[2].set_ylabel("Số ảnh")
    axes[2].set_title("Histogram chiều cao (Height)")
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    # In thống kê console
    print(f"\n{'='*52}")
    print(f"  Tổng số ảnh gốc : {len(widths)}")
    print(f"  Width  — min={widths.min():4d}  max={widths.max():4d}  "
          f"mean={widths.mean():.0f}  unique={np.unique(widths).tolist()}")
    print(f"  Height — min={heights.min():4d}  max={heights.max():4d}  "
          f"mean={heights.mean():.0f}  unique={np.unique(heights).tolist()}")
    print(f"{'='*52}")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Đã lưu → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════
# CHẠY
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    RAW_TRAIN = "./Train/Train"   # thư mục gốc chứa Rural/ và Urban/

    # ── Phần 2: Visualize dữ liệu gốc TRƯỚC khi crop ──────────────
    print("\n[1/3] Ảnh mẫu + lưới patch...")
    visualize_raw_samples(
        RAW_TRAIN,
        n_per_area=2,
        patch_size=PATCH_SIZE,
        save_path="./predictions/raw_samples.png",
    )

    print("\n[2/3] Phân bố lớp Rural vs Urban...")
    visualize_class_distribution(
        RAW_TRAIN,
        max_files_per_area=None,   # None = đọc toàn bộ
        save_path="./predictions/class_distribution_raw.png",
    )

    print("\n[3/3] Histogram kích thước ảnh gốc...")
    visualize_image_sizes(
        RAW_TRAIN,
        save_path="./predictions/image_sizes.png",
    )

    # ── Phần 1: Crop dataset (uncomment khi cần chạy lại) ─────────
    # print("\nCrop Train...")
    # crop_dataset("./Train/Train", "./LoveDA_patch/Train")
    # print("\nCrop Val...")
    # crop_dataset("./Val/Val", "./LoveDA_patch/Val")
