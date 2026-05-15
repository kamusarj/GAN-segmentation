import React, { useRef, useState } from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { Loader2, Maximize } from 'lucide-react';

// TileLayer sử dụng Google Maps Satellite (miễn phí, không cần API key)
const GOOGLE_SATELLITE = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}";

// Kích thước khung ngắm hiển thị (px trên màn hình)
const FRAME_SIZE = 256;

// Kích thước ảnh đầu ra gửi lên server — luôn cố định 512×512
const OUTPUT_SIZE = 512;

// ── Unsharp Mask bằng Canvas API ────────────────────────────────────────────
// Áp dụng kernel làm sắc nét cạnh lên ImageData của canvas.
// Kernel 3×3 (Laplacian-based):  [0,-1,0 / -1,5,-1 / 0,-1,0]
function _applyUnsharpMask(canvas) {
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  const src = ctx.getImageData(0, 0, width, height);
  const dst = ctx.createImageData(width, height);
  const s = src.data;
  const d = dst.data;

  const kernel = [0, -1, 0, -1, 5, -1, 0, -1, 0];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;

      // Pixel biên — copy nguyên
      if (x === 0 || x === width - 1 || y === 0 || y === height - 1) {
        d[i] = s[i]; d[i+1] = s[i+1]; d[i+2] = s[i+2]; d[i+3] = s[i+3];
        continue;
      }

      // Tích chập với kernel 3×3
      for (let c = 0; c < 3; c++) {
        let sum = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const ni = ((y + ky) * width + (x + kx)) * 4 + c;
            sum += s[ni] * kernel[(ky + 1) * 3 + (kx + 1)];
          }
        }
        d[i + c] = Math.min(255, Math.max(0, sum));
      }
      d[i + 3] = s[i + 3]; // alpha không đổi
    }
  }

  ctx.putImageData(dst, 0, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
export default function MapSelector({ onCapture }) {
  const mapRef  = useRef(null);
  const wrapRef = useRef(null);
  const [capturing, setCapturing] = useState(false);

  const handleCapture = async () => {
    if (!mapRef.current) return;
    setCapturing(true);

    try {
      const mapElement = mapRef.current.getContainer();

      // 1. Tạm ẩn Leaflet controls (nút zoom, attribution...)
      const controls = mapElement.querySelectorAll(
        '.leaflet-control-container, .leaflet-control'
      );
      controls.forEach(el => { el.style.visibility = 'hidden'; });

      // 2. Chụp toàn bộ map container bằng html2canvas
      const html2canvas = (await import('html2canvas')).default;
      const fullCanvas = await html2canvas(mapElement, {
        useCORS      : true,
        allowTaint   : false,
        backgroundColor: '#000',
        logging      : false,
      });

      // 3. Hiện lại controls
      controls.forEach(el => { el.style.visibility = ''; });

      // 4. Tính vùng crop tại tâm — theo kích thước HIỂN THỊ (FRAME_SIZE)
      //    html2canvas có thể scale theo devicePixelRatio, cần tính tỉ lệ
      const { width: domW, height: domH } = mapElement.getBoundingClientRect();
      const scaleX = fullCanvas.width  / domW;
      const scaleY = fullCanvas.height / domH;
      const cropW  = Math.round(FRAME_SIZE * scaleX);
      const cropH  = Math.round(FRAME_SIZE * scaleY);
      const sx     = Math.round((fullCanvas.width  - cropW) / 2);
      const sy     = Math.round((fullCanvas.height - cropH) / 2);

      // 5. Scale vùng crop lên OUTPUT_SIZE × OUTPUT_SIZE (512×512)
      const outCanvas = document.createElement('canvas');
      outCanvas.width  = OUTPUT_SIZE;
      outCanvas.height = OUTPUT_SIZE;
      const ctx = outCanvas.getContext('2d');
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(fullCanvas, sx, sy, cropW, cropH, 0, 0, OUTPUT_SIZE, OUTPUT_SIZE);

      // 6. Áp dụng Unsharp Mask làm sắc nét cạnh
      _applyUnsharpMask(outCanvas);

      // 7. Chuyển thành File và gửi lên parent
      outCanvas.toBlob(blob => {
        if (blob) {
          const file = new File([blob], 'map_snapshot.png', { type: 'image/png' });
          onCapture(file, URL.createObjectURL(blob));
        }
        setCapturing(false);
      }, 'image/png');

    } catch (err) {
      console.error('Lỗi khi chụp bản đồ:', err);
      alert('Không thể chụp bản đồ. Có thể do lỗi CORS từ server ảnh vệ tinh.');
      setCapturing(false);
    }
  };

  return (
    <div ref={wrapRef} className="relative w-full h-[500px] bg-slate-100 dark:bg-slate-800 rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-inner">
      <MapContainer
        center={[21.0285, 105.8542]} // Hà Nội
        zoom={15}
        style={{ width: '100%', height: '100%' }}
        ref={mapRef}
        attributionControl={false}
        zoomControl={true}
      >
        <TileLayer
          url={GOOGLE_SATELLITE}
          crossOrigin="anonymous"
          maxZoom={19}
        />
      </MapContainer>

      {/* Màn mờ + khung ngắm — pointer-events: none để không chặn tương tác bản đồ */}
      <div className="absolute inset-0 pointer-events-none flex items-center justify-center z-[400]">
        {/* Bóng mờ xung quanh khung */}
        <div
          className="w-64 h-64 border-2 border-white/80 shadow-[0_0_0_9999px_rgba(0,0,0,0.40)]"
          style={{ boxSizing: 'border-box' }}
        >
          {/* Nhãn kích thước output */}
          <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-[10px] text-white/80 bg-black/60 px-2 py-0.5 rounded-full whitespace-nowrap font-mono">
            512 × 512 px
          </div>

          {/* Dấu thập tâm */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-px h-6 bg-white/60 absolute" />
            <div className="h-px w-6 bg-white/60 absolute" />
            <div className="w-2 h-2 rounded-full bg-white/80 absolute" />
          </div>
          {/* 4 góc nhỏ */}
          <div className="absolute top-0 left-0  w-4 h-4 border-t-2 border-l-2 border-white" />
          <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-white" />
          <div className="absolute bottom-0 left-0  w-4 h-4 border-b-2 border-l-2 border-white" />
          <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-white" />
        </div>
      </div>

      {/* Nút chụp */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-[400]">
        <button
          onClick={handleCapture}
          disabled={capturing}
          className="px-6 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold rounded-xl shadow-xl flex items-center gap-2 disabled:opacity-50 transition-all"
        >
          {capturing ? <Loader2 className="w-5 h-5 animate-spin" /> : <Maximize className="w-5 h-5" />}
          {capturing ? 'Đang chụp & làm sắc nét...' : 'Chụp vùng này & Phân đoạn'}
        </button>
      </div>

      {/* Badge nguồn */}
      <div className="absolute top-2 left-2 z-[400] text-xs text-white/70 bg-black/50 px-2 py-1 rounded pointer-events-none">
        Google Maps Satellite
      </div>

      {/* Hướng dẫn */}
      <div className="absolute top-2 right-2 z-[400] text-xs text-white/70 bg-black/50 px-2 py-1 rounded pointer-events-none">
        Di chuyển bản đồ vào khung để chụp
      </div>
    </div>
  );
}
