import React, { useState } from 'react';
import axios from 'axios';
import {
  UploadCloud, Image as ImageIcon, CheckCircle,
  RefreshCcw, Loader2, BarChart2, Layers
} from 'lucide-react';

// ── LoveDA color palette (phải khớp với backend) ──────────────────────────
const CLASS_COLORS = [
  '#ffffff', // Background
  '#ff0000', // Building
  '#ffff00', // Road
  '#0000ff', // Water
  '#9f81b7', // Barren
  '#00ff00', // Forest
  '#ffc380', // Agricultural
];

function App() {
  const [selectedFile, setSelectedFile]   = useState(null);
  const [previewUrl, setPreviewUrl]       = useState(null);
  const [result, setResult]               = useState(null);   // { maskUrl, overlayUrl, classStats }
  const [activeTab, setActiveTab]         = useState('mask'); // 'mask' | 'overlay'
  const [isLoading, setIsLoading]         = useState(false);
  const [error, setError]                 = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await axios.post('http://localhost:8000/api/predict', formData);
      const data = response.data;

      setResult({
        maskUrl    : `data:image/png;base64,${data.mask_image}`,
        overlayUrl : `data:image/png;base64,${data.overlay_image}`,
        classStats : data.class_stats,
      });
      setActiveTab('overlay');
    } catch (err) {
      console.error('Prediction error:', err);
      const msg = err.response?.data?.detail || 'Đã có lỗi xảy ra khi kết nối Backend.';
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  const resetAll = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-pink-500 selection:text-white">
      {/* Top accent bar */}
      <div className="h-1 bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-500"></div>

      <header className="container mx-auto px-6 py-8 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-pink-500 to-purple-600 flex items-center justify-center">
            <ImageIcon className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
            GAN Segmentation
          </h1>
        </div>
        <span className="text-xs text-slate-500 border border-slate-700 rounded-full px-3 py-1">
          DeepLabV3+ ResNet50
        </span>
      </header>

      <main className="flex-1 container mx-auto px-6 py-8 flex flex-col items-center gap-10">

        {/* ── Hero text ── */}
        <div className="text-center max-w-2xl">
          <h2 className="text-4xl md:text-5xl font-extrabold mb-4 tracking-tight">
            Phân đoạn ảnh vệ tinh <br />
          </h2>
          <p className="text-slate-400 text-lg">
            Tải lên hình ảnh vệ tinh, hệ thống AI sẽ tự động nhận diện 7 lớp địa thực vật:
            Nền, Tòa nhà, Đường, Nước, Đất trống, Rừng, Canh tác.
          </p>
        </div>

        {/* ── Main card ── */}
        <div className="w-full max-w-6xl rounded-3xl bg-slate-900/50 border border-slate-800 backdrop-blur-xl p-6 md:p-10 shadow-2xl">

          {/* Upload zone */}
          {!previewUrl ? (
            <div className="relative group">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-pink-500 to-indigo-500 rounded-3xl blur opacity-20 group-hover:opacity-40 transition duration-500"></div>
              <label
                htmlFor="file-upload"
                className="relative flex flex-col items-center justify-center w-full h-80 rounded-2xl border-2 border-dashed border-slate-700 bg-slate-900 hover:bg-slate-800/80 cursor-pointer transition-all duration-300"
              >
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <div className="p-4 bg-slate-800 rounded-full mb-4 group-hover:scale-110 transition-transform duration-300">
                    <UploadCloud className="w-8 h-8 text-indigo-400" />
                  </div>
                  <p className="mb-2 text-lg font-semibold text-slate-300">Nhấn để tải ảnh lên</p>
                  <p className="text-sm text-slate-500">Hỗ trợ PNG, JPG, TIFF (Tối đa 20MB)</p>
                </div>
                <input id="file-upload" type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
              </label>
            </div>
          ) : (
            <div className="space-y-8">

              {/* ── Image row ── */}
              <div className="flex flex-col lg:flex-row gap-6 items-stretch">

                {/* Original */}
                <div className="flex-1 bg-slate-900 rounded-2xl overflow-hidden border border-slate-800 relative">
                  <div className="absolute top-3 left-3 bg-black/60 backdrop-blur-md px-3 py-1 rounded-lg text-xs font-medium z-10">
                    Ảnh gốc
                  </div>
                  <img src={previewUrl} alt="Original" className="w-full h-full object-cover aspect-square" />
                </div>

                {/* Controls */}
                <div className="flex lg:flex-col justify-center items-center gap-4">
                  {!result ? (
                    <button
                      id="btn-predict"
                      onClick={handlePredict}
                      disabled={isLoading}
                      className="px-8 py-4 bg-gradient-to-r from-pink-500 to-indigo-600 hover:from-pink-400 hover:to-indigo-500 text-white rounded-2xl font-bold shadow-lg shadow-indigo-500/25 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-3 transform hover:-translate-y-1"
                    >
                      {isLoading ? (
                        <><Loader2 className="w-5 h-5 animate-spin" /> Đang xử lý…</>
                      ) : (
                        <><CheckCircle className="w-5 h-5" /> Phân đoạn ngay</>
                      )}
                    </button>
                  ) : (
                    <button
                      id="btn-reset"
                      onClick={resetAll}
                      className="px-6 py-4 bg-slate-800 hover:bg-slate-700 text-white rounded-2xl font-semibold border border-slate-700 transition-colors flex items-center gap-3"
                    >
                      <RefreshCcw className="w-5 h-5" /> Thử ảnh khác
                    </button>
                  )}
                </div>

                {/* Result panel */}
                <div className="flex-1 bg-slate-900 rounded-2xl overflow-hidden border border-slate-800 relative flex flex-col aspect-square">
                  {result ? (
                    <>
                      {/* Tab switcher */}
                      <div className="absolute top-3 left-3 z-10 flex gap-1 bg-black/60 backdrop-blur rounded-lg p-1">
                        <button
                          id="tab-overlay"
                          onClick={() => setActiveTab('overlay')}
                          className={`px-3 py-1 rounded-md text-xs font-medium transition-colors flex items-center gap-1 ${activeTab === 'overlay' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}
                        >
                          <Layers className="w-3 h-3" /> Overlay
                        </button>
                        <button
                          id="tab-mask"
                          onClick={() => setActiveTab('mask')}
                          className={`px-3 py-1 rounded-md text-xs font-medium transition-colors flex items-center gap-1 ${activeTab === 'mask' ? 'bg-pink-600 text-white' : 'text-slate-400 hover:text-white'}`}
                        >
                          <ImageIcon className="w-3 h-3" /> Mask
                        </button>
                      </div>
                      <img
                        src={activeTab === 'mask' ? result.maskUrl : result.overlayUrl}
                        alt={activeTab}
                        className="w-full h-full object-cover"
                      />
                    </>
                  ) : (
                    <div className="flex-1 flex items-center justify-center text-center p-6">
                      {isLoading ? (
                        <div className="flex flex-col items-center gap-4">
                          <div className="relative">
                            <div className="w-16 h-16 border-4 border-slate-700 rounded-full"></div>
                            <div className="w-16 h-16 border-4 border-indigo-500 rounded-full border-t-transparent animate-spin absolute top-0 left-0"></div>
                          </div>
                          <p className="text-indigo-400 font-medium animate-pulse">AI đang phân tích ảnh…</p>
                        </div>
                      ) : (
                        <>
                          <div className="w-16 h-16 rounded-full bg-slate-800/50 flex items-center justify-center mx-auto mb-4 border border-slate-700">
                            <ImageIcon className="w-6 h-6 text-slate-500" />
                          </div>
                          <p className="text-slate-500 font-medium">Bản đồ phân đoạn sẽ hiển thị tại đây</p>
                        </>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* ── Error banner ── */}
              {error && (
                <div className="bg-red-900/40 border border-red-700 text-red-300 rounded-xl px-5 py-4 text-sm">
                  ⚠️ {error}
                </div>
              )}

              {/* ── Class stats ── */}
              {result?.classStats && (
                <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6">
                  <h3 className="text-sm font-semibold text-slate-400 mb-4 flex items-center gap-2">
                    <BarChart2 className="w-4 h-4" /> Phân bố lớp địa thực vật
                  </h3>
                  <div className="space-y-3">
                    {result.classStats
                      .slice()
                      .sort((a, b) => b.percent - a.percent)
                      .map((cls) => (
                        <div key={cls.id} className="flex items-center gap-3">
                          <span
                            className="w-3 h-3 rounded-full shrink-0 border border-white/20"
                            style={{ backgroundColor: CLASS_COLORS[cls.id] }}
                          />
                          <span className="text-xs text-slate-400 w-24 shrink-0">{cls.name}</span>
                          <div className="flex-1 bg-slate-800 rounded-full h-2 overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-700"
                              style={{
                                width: `${cls.percent}%`,
                                backgroundColor: CLASS_COLORS[cls.id],
                              }}
                            />
                          </div>
                          <span className="text-xs text-slate-300 w-12 text-right shrink-0">
                            {cls.percent.toFixed(1)}%
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              )}

            </div>
          )}
        </div>
      </main>

      <footer className="py-6 text-center text-xs text-slate-600">
        GAN Satellite Segmentation · DeepLabV3+ ResNet50 · LoveDA Dataset
      </footer>
    </div>
  );
}

export default App;
