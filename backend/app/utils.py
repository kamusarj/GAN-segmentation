import io
from PIL import Image, ImageOps
import numpy as np

def mock_predict(image_bytes: bytes) -> bytes:
    """
    Mock functionality: Takes an image, and simply returns a dummy segmentation mask.
    Chưa sử dụng model thật, chúng ta mock tạm một bộ lọc ảnh để test pipeline.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Tạo mask mock: chuyển ảnh sang xám rồi đảo màu 
    img_gray = img.convert("L")
    mask = ImageOps.invert(img_gray)
    
    # Trộn màu mock để trông giống ảnh segmentation
    mask_np = np.array(mask)
    colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    
    # Tô màu bừa dựa vào threshold sáng tối
    colored_mask[mask_np > 128] = [219, 39, 119]  # Pink for class A
    colored_mask[mask_np <= 128] = [16, 185, 129] # Green for class B

    colored_img = Image.fromarray(colored_mask)
    
    output_buffer = io.BytesIO()
    colored_img.save(output_buffer, format="JPEG")
    return output_buffer.getvalue()
