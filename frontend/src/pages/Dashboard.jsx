import React, { useState, useEffect, useRef, useCallback, useContext } from 'react';
import axios from 'axios';
import {
  UploadCloud, Image as ImageIcon, CheckCircle,
  RefreshCcw, Loader2, Download, ZoomIn, X, Clock, Sun, Moon, Map, UserCircle2
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { AuthContext } from '../AuthContext';
import { useTheme } from '../App';
import HistoryPanel from '../HistoryPanel';
import MapSelector from '../components/MapSelector';
import AdvancedControls from '../components/AdvancedControls';
import ProfileModal from '../components/ProfileModal';

// ── Lightbox với scroll zoom + kéo ──────────────────────────────────────────
function DashboardLightbox({ src, onClose }) {
  const [scale, setScale] = React.useState(1);
  const [pos,   setPos]   = React.useState({ x: 0, y: 0 });
  const [drag,  setDrag]  = React.useState(false);
  const [start, setStart] = React.useState({ x: 0, y: 0 });

  React.useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const handleWheel = e => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.15 : -0.15;
    setScale(s => Math.min(Math.max(s + delta, 0.2), 8));
  };

  return (
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center select-none"
      style={{ cursor: scale > 1 ? (drag ? 'grabbing' : 'grab') : 'default', background: 'rgba(0,0,0,0.92)', backdropFilter: 'blur(8px)' }}
      onWheel={handleWheel}
      onMouseDown={e => { if (scale <= 1) return; setDrag(true); setStart({ x: e.clientX - pos.x, y: e.clientY - pos.y }); }}
      onMouseMove={e => { if (!drag) return; setPos({ x: e.clientX - start.x, y: e.clientY - start.y }); }}
      onMouseUp={() => setDrag(false)}
      onMouseLeave={() => setDrag(false)}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      {/* Toolbar */}
      <div className="absolute top-4 right-4 flex gap-2 z-10">
        <button onClick={() => setScale(s => Math.max(s - 0.25, 0.2))} className="p-2 rounded-xl bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-300 hover:text-white transition-all" title="Thu nhỏ">
          <ZoomIn className="w-4 h-4 rotate-180" style={{ transform: 'scaleY(-1)' }} />
        </button>
        <button onClick={() => setScale(s => Math.min(s + 0.25, 8))} className="p-2 rounded-xl bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-300 hover:text-white transition-all" title="Phóng to">
          <ZoomIn className="w-4 h-4" />
        </button>
        <button onClick={() => { setScale(1); setPos({ x: 0, y: 0 }); }} className="p-2 rounded-xl bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-300 hover:text-white transition-all" title="Reset">
          <RefreshCcw className="w-4 h-4" />
        </button>
        <button onClick={onClose} className="p-2 rounded-xl bg-red-900/60 hover:bg-red-700 border border-red-700 text-red-300 hover:text-white transition-all" title="Đóng">
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Scale badge */}
      <div className="absolute top-4 left-4 z-10 px-3 py-1 rounded-full bg-slate-900/70 text-xs text-slate-300 font-mono">
        {Math.round(scale * 100)}%
      </div>

      {/* Image */}
      <img
        src={src}
        alt="Kết quả phân đoạn"
        draggable={false}
        style={{
          transform: `translate(${pos.x}px, ${pos.y}px) scale(${scale})`,
          transition: drag ? 'none' : 'transform 0.15s ease',
          maxWidth: '90vw',
          maxHeight: '90vh',
          objectFit: 'contain',
          borderRadius: 12,
          boxShadow: '0 25px 60px rgba(0,0,0,0.7)',
          userSelect: 'none',
        }}
        onClick={e => e.stopPropagation()}
      />

      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-xs text-slate-500 bg-slate-900/70 px-4 py-2 rounded-full whitespace-nowrap">
        Scroll để zoom · Kéo để di chuyển · Esc để đóng
      </div>
    </div>
  );
}


const DEFAULT_CLASSES = [
  { id: 0, name: "Background" },
  { id: 1, name: "Building" },
  { id: 2, name: "Road" },
  { id: 3, name: "Water" },
  { id: 4, name: "Barren" },
  { id: 5, name: "Forest" },
  { id: 6, name: "Agricultural" }
];

const DEFAULT_COLORS = {
  0: '#ffffff', // Background (white or transparent)
  1: '#ff0000', // Building (red)
  2: '#ffff00', // Road (yellow)
  3: '#0000ff', // Water (blue)
  4: '#9f81b7', // Barren (purple)
  5: '#00ff00', // Forest (green)
  6: '#ffc380', // Agricultural (orange)
};

function hexToRgb(hex) {
  var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : {r: 0, g: 0, b: 0};
}

export default function Dashboard() {
  const { user, logout } = useContext(AuthContext);
  const navigate = useNavigate();
  const [dark, toggleTheme] = useTheme();
  
  // Tabs
  const [view, setView] = useState('segment');
  const [inputTab, setInputTab] = useState('upload');
  const [showProfile, setShowProfile] = useState(false);

  // Người dùng có thể dùng bản đồ nếu là Admin hoặc có quyền Premium
  const canUseMap = user?.role === 'admin' || user?.is_premium === true;

  // Lightbox xem ảnh kết quả to
  const [lightboxSrc, setLightboxSrc] = useState(null);

  // Data State
  const [selectedFile, setFile] = useState(null);
  const [previewUrl, setPreview] = useState(null);
  
  // API Result State
  const [isLoading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [warnings, setWarnings] = useState([]);
  const [rawMaskUrl, setRawMaskUrl] = useState(null);
  const [classStats, setClassStats] = useState([]);
  
  // Advanced Control State
  const [opacity, setOpacity] = useState(50);
  const [classColors, setClassColors] = useState(DEFAULT_COLORS);
  const [activeClasses, setActiveClasses] = useState([0, 1, 2, 3, 4, 5, 6]);
  
  // Canvas Refs
  const canvasRef = useRef(null);
  const rawImageRef = useRef(null);

  const handleFile = e => {
    if (e.target.files?.[0]) {
      const f = e.target.files[0];
      if (!f.type.startsWith('image/')) {
        setError(`File "${f.name}" không phải là ảnh. Vui lòng chọn file PNG, JPG hoặc TIFF.`);
        e.target.value = '';
        return;
      }
      setFile(f); setPreview(URL.createObjectURL(f)); resetResult();
    }
  };

  const handleMapCapture = (file, url) => {
    setFile(file);
    setPreview(url);
    resetResult();
    handlePredict(file);
  };

  const resetResult = () => {
    setRawMaskUrl(null);
    setClassStats([]);
    setError(null);
    setWarnings([]);
  };

  const resetAll = () => {
    setFile(null);
    setPreview(null);
    resetResult();
  };

  const handlePredict = async (fileToUpload = selectedFile) => {
    if (!fileToUpload) return;
    setLoading(true); setError(null); setWarnings([]);
    const fd = new FormData();
    fd.append('image', fileToUpload);
    try {
      const { data } = await axios.post('http://localhost:8000/api/predict', fd);
      setRawMaskUrl(`data:image/png;base64,${data.raw_mask}`);
      setClassStats(data.class_stats);
      setWarnings(data.warnings || []);
    } catch (err) {
      if (err.response?.status === 401) {
        logout();
      } else {
        setError(err.response?.data?.detail || 'Đã có lỗi xảy ra khi kết nối Backend.');
      }
    } finally { 
      setLoading(false); 
    }
  };

  // Tối ưu hóa: Cache lại dữ liệu pixel để không phải load Image trên mỗi frame
  const rawMaskDataRef = useRef(null);
  const maskDimensionsRef = useRef({ w: 0, h: 0 });
  const imageDataRef = useRef(null);

  // 1. Chỉ load Image 1 lần khi rawMaskUrl thay đổi
  useEffect(() => {
    if (!rawMaskUrl) return;
    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.src = rawMaskUrl;
    img.onload = () => {
      const cvs = document.createElement('canvas');
      cvs.width = img.width;
      cvs.height = img.height;
      const ctx = cvs.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(img, 0, 0);
      const imgData = ctx.getImageData(0, 0, cvs.width, cvs.height);
      
      // Trích xuất classId của từng pixel vào mảng 1D (cực nhanh)
      const len = imgData.data.length / 4;
      const classIds = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        classIds[i] = imgData.data[i * 4];
      }
      
      rawMaskDataRef.current = classIds;
      maskDimensionsRef.current = { w: img.width, h: img.height };
      imageDataRef.current = new ImageData(img.width, img.height);
      
      drawColoredMask(); // Kích hoạt vẽ lần đầu
    };
  }, [rawMaskUrl]);

  // 2. Hàm vẽ mask siêu tốc khi người dùng kéo color picker
  const drawColoredMask = useCallback(() => {
    if (!rawMaskDataRef.current || !canvasRef.current || !imageDataRef.current) return;
    const classIds = rawMaskDataRef.current;
    const { w, h } = maskDimensionsRef.current;
    
    if (canvasRef.current.width !== w) canvasRef.current.width = w;
    if (canvasRef.current.height !== h) canvasRef.current.height = h;

    const imgData = imageDataRef.current;
    const data = imgData.data;
    
    // Tính toán trước bảng màu (precomputed lookup table) để không phải parse Hex liên tục
    const colorMap = {};
    for (let i = 0; i <= 6; i++) {
      if (activeClasses.includes(i)) {
        const rgb = hexToRgb(classColors[i]);
        colorMap[i] = [rgb.r, rgb.g, rgb.b, 255];
      } else {
        colorMap[i] = [0, 0, 0, 0];
      }
    }

    // Gán màu trực tiếp từ lookup table (siêu nhanh, ~1ms cho 262k pixels)
    for (let i = 0; i < classIds.length; i++) {
      const cId = classIds[i];
      const color = colorMap[cId] || [0, 0, 0, 0];
      const idx = i * 4;
      data[idx]     = color[0];
      data[idx + 1] = color[1];
      data[idx + 2] = color[2];
      data[idx + 3] = color[3];
    }
    
    canvasRef.current.getContext('2d').putImageData(imgData, 0, 0);
  }, [classColors, activeClasses]);

  // 3. Tự động vẽ lại mask chỉ khi màu hoặc toggle bị thay đổi
  useEffect(() => {
    drawColoredMask();
  }, [drawColoredMask]);

  // 4. Vẽ lại mask khi người dùng quay lại tab 'segment' sau khi đã phân đoạn
  //    (vì canvas bị unmount khi chuyển sang tab lịch sử)
  useEffect(() => {
    if (view === 'segment' && rawMaskDataRef.current) {
      // Dùng setTimeout để đảm bảo canvas đã được render vào DOM
      const timer = setTimeout(() => {
        drawColoredMask();
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [view, drawColoredMask]);

  const toggleActiveClass = (id) => {
    setActiveClasses(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };

  const changeClassColor = (id, color) => {
    setClassColors(prev => ({ ...prev, [id]: color }));
  };

  const handleDownload = () => {
    // Merge original image and canvas
    if (!previewUrl || !canvasRef.current) return;
    const origImg = rawImageRef.current;
    
    const mergeCvs = document.createElement('canvas');
    mergeCvs.width = origImg.naturalWidth;
    mergeCvs.height = origImg.naturalHeight;
    const mergeCtx = mergeCvs.getContext('2d');
    
    // Draw original
    mergeCtx.drawImage(origImg, 0, 0);
    // Draw mask with opacity
    mergeCtx.globalAlpha = opacity / 100;
    mergeCtx.drawImage(canvasRef.current, 0, 0, origImg.naturalWidth, origImg.naturalHeight);
    
    const a = document.createElement('a');
    a.href = mergeCvs.toDataURL('image/png');
    a.download = 'segmentation_result.png';
    a.click();
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 flex flex-col font-sans transition-colors duration-300">
      <div className="h-1 bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-500"/>

      <header className="container mx-auto px-6 py-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-pink-500 to-purple-600 flex items-center justify-center shadow-lg shadow-pink-500/20">
            <ImageIcon className="w-5 h-5 text-white"/>
          </div>
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-pink-500 to-indigo-500">
            GAN Segmentation
          </h1>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-1 shadow-sm">
            <button onClick={() => setView('segment')}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-all flex items-center gap-2 ${view==='segment' ? 'bg-gradient-to-r from-pink-600 to-indigo-600 text-white shadow-lg' : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'}`}>
              <ImageIcon className="w-4 h-4"/> Phân đoạn
            </button>
            <button onClick={() => setView('history')}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-all flex items-center gap-2 ${view==='history' ? 'bg-gradient-to-r from-pink-600 to-indigo-600 text-white shadow-lg' : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'}`}>
              <Clock className="w-4 h-4"/> Lịch sử
            </button>
          </div>

          {user?.role === 'admin' && (
            <button onClick={() => navigate('/admin')} className="px-4 py-2 rounded-xl text-sm font-medium bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors">
              Admin Panel
            </button>
          )}

          <button onClick={toggleTheme} className="p-2.5 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-all shadow-sm">
            {dark ? <Sun className="w-4 h-4"/> : <Moon className="w-4 h-4"/>}
          </button>
          
          <button onClick={logout} className="px-4 py-2 text-sm font-medium text-slate-500 hover:text-red-500 transition-colors">
            Đăng xuất
          </button>

          {/* Avatar button mở ProfileModal */}
          <button
            onClick={() => setShowProfile(true)}
            title="Thông tin tài khoản"
            className="flex items-center gap-2 pl-1 pr-3 py-1 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 hover:border-indigo-400 dark:hover:border-indigo-500 shadow-sm hover:shadow-indigo-200/50 dark:hover:shadow-indigo-900/50 transition-all group"
          >
            {/* Icon avatar */}
            <div className={`w-7 h-7 rounded-xl flex items-center justify-center shrink-0 text-white text-xs font-bold shadow-inner ${
              user?.role === 'admin'
                ? 'bg-gradient-to-br from-red-500 to-orange-500'
                : 'bg-gradient-to-br from-indigo-500 to-purple-600'
            }`}>
              <UserCircle2 className="w-4 h-4" />
            </div>
            {/* Tên + role */}
            <div className="flex flex-col items-start leading-none">
              <span className="text-xs font-semibold text-slate-800 dark:text-slate-200 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                {user?.username}
              </span>
              <span className={`text-[10px] font-medium mt-0.5 ${
                user?.role === 'admin' ? 'text-red-500' : user?.is_premium ? 'text-amber-500' : 'text-slate-400'
              }`}>
                {user?.role === 'admin' ? '👑 Admin' : user?.is_premium ? '⭐ Premium' : 'User'}
              </span>
            </div>
          </button>
        </div>
      </header>

      {showProfile && <ProfileModal onClose={() => setShowProfile(false)} />}

      <main className="flex-1 container mx-auto px-6 py-6 flex flex-col items-center gap-8">
        
        {view === 'history' && (
          <div className="w-full max-w-6xl">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100">Lịch sử phân đoạn</h2>
              <p className="text-slate-500 dark:text-slate-500 text-sm mt-1">Lịch sử được lưu tự động cho tài khoản của bạn</p>
            </div>
            <div className="rounded-3xl bg-white dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800 p-6 md:p-8 shadow-xl">
              <HistoryPanel />
            </div>
          </div>
        )}

        {view === 'segment' && (
          <div className="w-full max-w-6xl grid grid-cols-1 xl:grid-cols-3 gap-6">
            
            {/* Lớp bên trái: Ảnh và Map */}
            <div className="xl:col-span-2 space-y-6">
              <div className="rounded-3xl bg-white dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800 p-6 shadow-xl">
                
                {/* Tab Input */}
                {!previewUrl && (
                  <div className="flex gap-2 mb-6">
                    <button onClick={() => setInputTab('upload')} className={`flex-1 py-3 rounded-xl flex items-center justify-center gap-2 font-medium border-2 transition-colors ${inputTab === 'upload' ? 'border-indigo-500 text-indigo-600 bg-indigo-50 dark:bg-indigo-900/20' : 'border-slate-200 dark:border-slate-700 text-slate-500'}`}>
                      <UploadCloud className="w-5 h-5"/> Tải ảnh lên
                    </button>
                    {canUseMap ? (
                      <button onClick={() => setInputTab('map')} className={`flex-1 py-3 rounded-xl flex items-center justify-center gap-2 font-medium border-2 transition-colors ${inputTab === 'map' ? 'border-emerald-500 text-emerald-600 bg-emerald-50 dark:bg-emerald-900/20' : 'border-slate-200 dark:border-slate-700 text-slate-500'}`}>
                        <Map className="w-5 h-5"/> Chọn từ bản đồ
                      </button>
                    ) : (
                      <div className="flex-1 py-3 rounded-xl flex flex-col items-center justify-center gap-0.5 border-2 border-dashed border-slate-200 dark:border-slate-700 text-slate-400 cursor-not-allowed select-none">
                        <div className="flex items-center gap-2 text-sm font-medium">
                          <Map className="w-4 h-4"/> Bản đồ vệ tinh
                        </div>
                        <p className="text-xs">🔒 Yêu cầu quyền Premium</p>
                      </div>
                    )}
                  </div>
                )}

                {/* Khu vực Upload/Map */}
                {!previewUrl ? (
                  inputTab === 'upload' ? (
                    <div className="space-y-3">
                      {/* Cảnh báo chỉ dùng ảnh vệ tinh */}
                      <div className="flex items-start gap-3 px-4 py-3 rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700/50 text-amber-700 dark:text-amber-400">
                        <span className="text-lg shrink-0 mt-0.5">🛰️</span>
                        <div className="text-sm leading-relaxed">
                          <span className="font-semibold">Lưu ý:</span> Mô hình được huấn luyện trên ảnh vệ tinh (LoveDA dataset). Vui lòng chỉ tải lên <span className="font-semibold">ảnh vệ tinh</span> để có kết quả phân đoạn chính xác nhất.
                        </div>
                      </div>
                      <label className={`relative flex flex-col items-center justify-center w-full h-72 rounded-2xl border-2 border-dashed bg-slate-50 dark:bg-slate-900 cursor-pointer transition-all duration-300 ${error ? 'border-red-400' : 'border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800'}`}>
                        <div className="flex flex-col items-center justify-center pt-5 pb-6">
                          <UploadCloud className={`w-10 h-10 mb-4 ${error ? 'text-red-400' : 'text-indigo-500'}`}/>
                          <p className="mb-2 text-lg font-semibold text-slate-700 dark:text-slate-300">Nhấn để tải ảnh lên</p>
                          <p className="text-sm text-slate-500 dark:text-slate-400">Hỗ trợ: PNG, JPG, JPEG, TIFF (Tối đa 10MB)</p>
                        </div>
                        <input type="file" className="hidden" accept="image/*" onChange={handleFile}/>
                      </label>
                    </div>
                  ) : (
                    <MapSelector onCapture={handleMapCapture} />
                  )
                ) : (
                  // Preview và Overlay
                  <div className="space-y-4">
                    <div className="relative rounded-2xl overflow-hidden bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 flex items-center justify-center aspect-square">
                      {/* Ảnh gốc nằm dưới cùng */}
                      <img 
                        ref={rawImageRef}
                        src={previewUrl} 
                        alt="Original" 
                        className="absolute inset-0 w-full h-full object-contain" 
                      />
                      
                      {/* Canvas Mask nằm đè lên trên với opacity */}
                      <canvas 
                        ref={canvasRef}
                        className="absolute inset-0 w-full h-full object-contain pointer-events-none mix-blend-normal"
                        style={{ opacity: opacity / 100 }}
                      />

                      {/* Nút zoom - chỉ hiện khi đã có kết quả */}
                      {rawMaskUrl && !isLoading && (
                        <button
                          onClick={() => {
                            const merge = document.createElement('canvas');
                            const ri = rawImageRef.current;
                            merge.width  = ri.naturalWidth;
                            merge.height = ri.naturalHeight;
                            const ctx = merge.getContext('2d');
                            ctx.drawImage(ri, 0, 0);
                            ctx.globalAlpha = opacity / 100;
                            ctx.drawImage(canvasRef.current, 0, 0, merge.width, merge.height);
                            setLightboxSrc(merge.toDataURL('image/png'));
                          }}
                          className="absolute top-2 right-2 z-10 p-2 rounded-xl bg-white/85 dark:bg-black/60 hover:bg-indigo-600/90 border border-slate-200 dark:border-slate-700 hover:border-indigo-500 text-slate-600 dark:text-slate-300 hover:text-white transition-all shadow-md"
                          title="Xem ảnh kết quả cỡ lớn"
                        >
                          <ZoomIn className="w-5 h-5" />
                        </button>
                      )}

                      {isLoading && (
                        <div className="absolute inset-0 bg-white/50 dark:bg-black/50 backdrop-blur-sm flex flex-col items-center justify-center z-10">
                          <Loader2 className="w-10 h-10 text-indigo-600 animate-spin mb-4" />
                          <span className="font-medium text-slate-800 dark:text-white">AI đang phân tích...</span>
                        </div>
                      )}
                    </div>

                    <div className="flex justify-between items-center">
                      <button onClick={resetAll} className="px-5 py-2.5 rounded-xl bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 font-medium hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors flex items-center gap-2">
                        <RefreshCcw className="w-4 h-4"/> Thử ảnh khác
                      </button>
                      
                      {!rawMaskUrl ? (
                        <button onClick={() => handlePredict()} disabled={isLoading} className="px-6 py-2.5 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-700 transition-colors flex items-center gap-2">
                          <CheckCircle className="w-4 h-4"/> Phân đoạn ngay
                        </button>
                      ) : (
                        <button onClick={handleDownload} className="px-6 py-2.5 rounded-xl bg-emerald-600 text-white font-medium hover:bg-emerald-700 transition-colors flex items-center gap-2">
                          <Download className="w-4 h-4"/> Tải ảnh kết quả
                        </button>
                      )}
                    </div>
                  </div>
                )}
                
                {error && <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-xl text-sm border border-red-200">⚠️ {error}</div>}
              </div>
            </div>

            {/* Lớp bên phải: Advanced Controls */}
            <div className="xl:col-span-1">
              <AdvancedControls 
                classes={DEFAULT_CLASSES}
                activeClasses={activeClasses}
                toggleActiveClass={toggleActiveClass}
                classColors={classColors}
                changeClassColor={changeClassColor}
                opacity={opacity}
                setOpacity={setOpacity}
                classStats={classStats}
              />
            </div>
          </div>
        )}
      </main>

      {/* Lightbox xem kết quả phân đoạn — có scroll zoom + kéo */}
      {lightboxSrc && (
        <DashboardLightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />
      )}
    </div>
  );
}
