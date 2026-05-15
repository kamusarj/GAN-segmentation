import React, { useState, useEffect, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import axios from 'axios';
import { Clock, Trash2, Download, ZoomIn, X, ZoomOut, Maximize2, Layers, Image as ImageIcon, RefreshCcw, Loader2, Eye, EyeOff, Sliders, Palette } from 'lucide-react';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
ChartJS.register(ArcElement, Tooltip, Legend);

const API = 'http://localhost:8000';
const CLASS_COLORS = ['#ffffff','#ff0000','#ffff00','#0000ff','#9f81b7','#00ff00','#ffc380'];

const DEFAULT_CLASSES = [
  {id:0,name:'Background'},{id:1,name:'Building'},{id:2,name:'Road'},
  {id:3,name:'Water'},{id:4,name:'Barren'},{id:5,name:'Forest'},{id:6,name:'Agricultural'}
];
const DEFAULT_COLORS = {0:'#ffffff',1:'#ff0000',2:'#ffff00',3:'#0000ff',4:'#9f81b7',5:'#00ff00',6:'#ffc380'};
function hexToRgb(hex){const r=/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);return r?{r:parseInt(r[1],16),g:parseInt(r[2],16),b:parseInt(r[3],16)}:{r:0,g:0,b:0};}

function downloadB64(b64, filename) {
  const a = document.createElement('a');
  a.href = `data:image/png;base64,${b64}`;
  a.download = filename; a.click();
}

// ── Mini Lightbox ──────────────────────────────────────────────────────────
function Lightbox({ src, onClose }) {
  const [scale, setScale] = useState(1);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const [drag, setDrag] = useState(false);
  const [start, setStart] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const h = e => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);

  // createPortal → render thẳng vào document.body, thoát khỏi
  // stacking context của backdrop-filter trong parent modal
  return createPortal(
    <div className="fixed inset-0 z-[9999] flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.92)', backdropFilter: 'blur(8px)' }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="absolute top-4 right-4 flex gap-2 z-10">
        <button onClick={() => setScale(s=>Math.max(s-0.25,0.25))} className="p-2 rounded-xl bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-300 hover:text-white transition-all"><ZoomOut className="w-4 h-4"/></button>
        <button onClick={() => setScale(s=>Math.min(s+0.25,5))} className="p-2 rounded-xl bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-300 hover:text-white transition-all"><ZoomIn className="w-4 h-4"/></button>
        <button onClick={() => { setScale(1); setPos({x:0,y:0}); }} className="p-2 rounded-xl bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-300 hover:text-white transition-all"><Maximize2 className="w-4 h-4"/></button>
        <button onClick={onClose} className="p-2 rounded-xl bg-red-900/60 hover:bg-red-700 border border-red-700 text-red-300 hover:text-white transition-all"><X className="w-4 h-4"/></button>
      </div>
      <div className="w-full h-full flex items-center justify-center overflow-hidden"
        style={{ cursor: scale>1?(drag?'grabbing':'grab'):'default' }}
        onWheel={e=>{e.preventDefault();setScale(s=>Math.min(Math.max(s+(e.deltaY<0?0.15:-0.15),0.25),5));}}
        onMouseDown={e=>{if(scale<=1)return;setDrag(true);setStart({x:e.clientX-pos.x,y:e.clientY-pos.y});}}
        onMouseMove={e=>{if(!drag)return;setPos({x:e.clientX-start.x,y:e.clientY-start.y});}}
        onMouseUp={()=>setDrag(false)} onMouseLeave={()=>setDrag(false)}>
        <img src={src} draggable={false} alt="detail"
          style={{transform:`translate(${pos.x}px,${pos.y}px) scale(${scale})`,transition:drag?'none':'transform 0.15s',maxWidth:'90vw',maxHeight:'90vh',objectFit:'contain',borderRadius:12,boxShadow:'0 25px 60px rgba(0,0,0,0.7)',userSelect:'none'}}/>
      </div>
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-xs text-slate-500 bg-slate-900/70 px-4 py-2 rounded-full">
        Scroll để zoom · Kéo để di chuyển · Esc để đóng
      </div>
    </div>,
    document.body
  );
}

// ── Detail Modal ───────────────────────────────────────────────────────────
function DetailModal({ recordId, onClose }) {
  const [data, setData]       = useState(null);
  const [tab, setTab]         = useState('overlay');
  const [lb, setLb]           = useState(null);
  const [opacity, setOpacity] = useState(60);
  const [reSegLoading, setReSegLoading]         = useState(false);
  const [reSegRawMaskUrl, setReSegRawMaskUrl]   = useState(null);
  const [reSegClassStats, setReSegClassStats]   = useState([]);
  const [reSegActiveClasses, setReSegActiveClasses] = useState([0,1,2,3,4,5,6]);
  const [reSegClassColors, setReSegClassColors] = useState({...DEFAULT_COLORS});

  const canvasRef          = useRef(null);
  const offMaskCanvasRef   = useRef(document.createElement('canvas'));
  const reSegRawDataRef    = useRef(null);
  const reSegDimRef        = useRef({w:0,h:0});
  const reSegImgDataRef    = useRef(null);
  // Cache ảnh để blendCanvas không phải load lại mỗi lần kéo slider
  const origImgRef         = useRef(null);
  const overlayImgRef      = useRef(null);

  // Reset khi đổi record
  useEffect(() => {
    setData(null); setTab('overlay'); setLb(null);
    setReSegRawMaskUrl(null); setReSegClassStats([]);
    setReSegActiveClasses([0,1,2,3,4,5,6]); setReSegClassColors({...DEFAULT_COLORS});
    reSegRawDataRef.current = null;
    origImgRef.current = null; overlayImgRef.current = null;
    const token = localStorage.getItem('token');
    axios.get(`${API}/api/history/${recordId}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    }).then(r=>setData(r.data)).catch(()=>onClose());
  }, [recordId]);

  // Pre-load ảnh gốc và overlay 1 lần khi data thay đổi
  useEffect(() => {
    if (!data) return;
    const orig = new Image();
    orig.onload = () => { origImgRef.current = orig; };
    orig.src = `data:image/png;base64,${data.original_image || data.original_thumb}`;
    const ov = new Image();
    ov.onload = () => { overlayImgRef.current = ov; blendCanvas(); };
    ov.src = `data:image/png;base64,${data.overlay_image}`;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  // Blend canvas: đồng bộ (không load lại ảnh), dùng ảnh đã cache trong ref
  const blendCanvas = useCallback(() => {
    if (!origImgRef.current || tab !== 'overlay' || !canvasRef.current) return;
    const orig = origImgRef.current;
    const cvs = canvasRef.current;
    cvs.width = orig.naturalWidth || 512; cvs.height = orig.naturalHeight || 512;
    const ctx = cvs.getContext('2d');
    ctx.clearRect(0, 0, cvs.width, cvs.height);
    ctx.drawImage(orig, 0, 0, cvs.width, cvs.height);
    ctx.globalAlpha = opacity / 100;
    if (reSegRawDataRef.current) {
      ctx.drawImage(offMaskCanvasRef.current, 0, 0, cvs.width, cvs.height);
    } else if (overlayImgRef.current) {
      ctx.drawImage(overlayImgRef.current, 0, 0, cvs.width, cvs.height);
    }
    ctx.globalAlpha = 1;
  }, [opacity, tab]);

  // Vẽ colored mask vào off-screen canvas, rồi blend
  const drawReSegMask = useCallback(() => {
    if (!reSegRawDataRef.current) return;
    const {w, h} = reSegDimRef.current;
    const cvs = offMaskCanvasRef.current;
    cvs.width = w; cvs.height = h;
    const imgData = reSegImgDataRef.current;
    const d = imgData.data;
    const ids = reSegRawDataRef.current;
    const cm = {};
    for (let i=0;i<=6;i++) {
      if (reSegActiveClasses.includes(i)) { const rgb=hexToRgb(reSegClassColors[i]); cm[i]=[rgb.r,rgb.g,rgb.b,255]; }
      else cm[i]=[0,0,0,0];
    }
    for (let i=0;i<ids.length;i++) { const c=cm[ids[i]]||[0,0,0,0]; const x=i*4; d[x]=c[0];d[x+1]=c[1];d[x+2]=c[2];d[x+3]=c[3]; }
    cvs.getContext('2d').putImageData(imgData, 0, 0);
    blendCanvas();
  }, [reSegActiveClasses, reSegClassColors, blendCanvas]);

  // Load raw mask pixels khi re-seg xong
  useEffect(() => {
    if (!reSegRawMaskUrl) return;
    const img = new Image();
    img.onload = () => {
      const tmp = document.createElement('canvas');
      tmp.width=img.width; tmp.height=img.height;
      const ctx = tmp.getContext('2d',{willReadFrequently:true});
      ctx.drawImage(img,0,0);
      const imgData = ctx.getImageData(0,0,img.width,img.height);
      const ids = new Uint8Array(imgData.data.length/4);
      for(let i=0;i<ids.length;i++) ids[i]=imgData.data[i*4];
      reSegRawDataRef.current=ids;
      reSegDimRef.current={w:img.width,h:img.height};
      reSegImgDataRef.current=new ImageData(img.width,img.height);
      drawReSegMask();
    };
    img.src = reSegRawMaskUrl;
  }, [reSegRawMaskUrl, drawReSegMask]);

  // Redraw khi màu / active classes thay đổi
  useEffect(() => { if(reSegRawDataRef.current) drawReSegMask(); }, [drawReSegMask]);

  // Redraw khi opacity hoặc tab thay đổi (đồng bộ, không load ảnh)
  useEffect(() => { blendCanvas(); }, [blendCanvas]);

  // ── Re-segment ───────────────────────────────────────────────────────────
  const handleReSegment = async () => {
    if (!data) return;
    setReSegLoading(true);
    try {
      const src = data.original_image || data.original_thumb;
      const res  = await fetch(`data:image/png;base64,${src}`);
      const blob = await res.blob();
      const file = new File([blob], data.filename, { type: 'image/png' });
      const fd = new FormData(); fd.append('image', file);
      const token = localStorage.getItem('token');
      const { data: result } = await axios.post(`${API}/api/predict`, fd, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      setReSegActiveClasses([0,1,2,3,4,5,6]);
      setReSegClassColors({...DEFAULT_COLORS});
      setReSegClassStats(result.class_stats);
      setReSegRawMaskUrl(`data:image/png;base64,${result.raw_mask}`);
      setTab('overlay');
    } catch (err) {
      alert(err.response?.data?.detail || 'Lỗi khi phân đoạn lại.');
    } finally { setReSegLoading(false); }
  };

  if (!data) return (
    <div className="fixed inset-0 z-40 flex items-center justify-center" style={{background:'rgba(0,0,0,0.7)'}}>
      <Loader2 className="w-8 h-8 text-indigo-500 animate-spin"/>
    </div>
  );

  const maskSrc    = `data:image/png;base64,${data.mask_image}`;
  const overlaySrc = canvasRef.current?.toDataURL('image/png') || `data:image/png;base64,${data.overlay_image}`;
  const displayStats = reSegRawMaskUrl ? reSegClassStats : data.class_stats;

  const handleDownload = () => {
    if (tab === 'mask') { downloadB64(data.mask_image, `mask_${data.filename}`); return; }
    if (!canvasRef.current) return;
    const a = document.createElement('a');
    a.href = canvasRef.current.toDataURL('image/png');
    a.download = `overlay_${opacity}pct_${data.filename}`;
    a.click();
  };

  return (
    <>
      <div className="fixed inset-0 z-40 flex items-center justify-center p-4"
        style={{background:'rgba(0,0,0,0.6)', backdropFilter:'blur(6px)'}}
        onClick={e=>{if(e.target===e.currentTarget) onClose();}}>
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-3xl w-full max-w-4xl max-h-[92vh] overflow-y-auto shadow-2xl">

          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 dark:border-slate-800">
            <div>
              <p className="font-semibold text-slate-800 dark:text-slate-200 truncate max-w-xs">{data.filename}</p>
              <p className="text-xs text-slate-400 flex items-center gap-1 mt-0.5"><Clock className="w-3 h-3"/>{data.created_at}</p>
            </div>
            <div className="flex gap-2">
              {/* Nút Phân đoạn lại */}
              <button
                onClick={handleReSegment}
                disabled={reSegLoading}
                className="px-3 py-2 text-xs bg-indigo-50 dark:bg-indigo-900/30 hover:bg-indigo-100 dark:hover:bg-indigo-800/50 border border-indigo-200 dark:border-indigo-700 text-indigo-700 dark:text-indigo-300 rounded-xl transition-all flex items-center gap-1.5 disabled:opacity-50"
                title="Phân đoạn lại ảnh này"
              >
                {reSegLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin"/> : <RefreshCcw className="w-3.5 h-3.5"/>}
                {reSegLoading ? 'Đang xử lý...' : 'Phân đoạn lại'}
              </button>
              <button onClick={handleDownload}
                className="px-3 py-2 text-xs bg-emerald-50 dark:bg-emerald-700/60 hover:bg-emerald-100 dark:hover:bg-emerald-600 border border-emerald-200 dark:border-emerald-700 text-emerald-700 dark:text-emerald-300 rounded-xl transition-all flex items-center gap-1.5">
                <Download className="w-3.5 h-3.5"/> Tải xuống
              </button>
              <button onClick={onClose}
                className="p-2 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 rounded-xl transition-all">
                <X className="w-4 h-4"/>
              </button>
            </div>
          </div>

          <div className="p-6 flex flex-col lg:flex-row gap-6">
            {/* Image panel */}
            <div className="flex-1">
              {/* Tab bar */}
              <div className="flex gap-1 mb-3 bg-slate-100 dark:bg-slate-800 rounded-xl p-1 w-fit">
                {['overlay','mask'].map(t=>(
                  <button key={t} onClick={()=>setTab(t)}
                    className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors flex items-center gap-1 ${tab===t?(t==='overlay'?'bg-indigo-600':'bg-pink-600')+' text-white':'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-white'}`}>
                    {t==='overlay'?<Layers className="w-3 h-3"/>:<ImageIcon className="w-3 h-3"/>} {t==='overlay'?'Overlay':'Mask'}
                  </button>
                ))}
              </div>

              {/* Image area */}
              <div className="relative rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 aspect-square bg-slate-100 dark:bg-slate-800">
                {tab === 'overlay' ? (
                  <canvas
                    ref={canvasRef}
                    className="w-full h-full object-cover cursor-zoom-in"
                    onClick={() => setLb(canvasRef.current?.toDataURL('image/png') || overlaySrc)}
                  />
                ) : (
                  <img src={maskSrc} alt="mask" className="w-full h-full object-cover cursor-zoom-in" onClick={()=>setLb(maskSrc)}/>
                )}
                <button
                  onClick={() => tab==='overlay'
                    ? setLb(canvasRef.current?.toDataURL('image/png') || overlaySrc)
                    : setLb(maskSrc)
                  }
                  className="absolute top-2 right-2 p-1.5 rounded-lg bg-white/80 dark:bg-black/60 hover:bg-indigo-600/80 border border-slate-200 dark:border-slate-700 hover:border-indigo-500 text-slate-600 dark:text-slate-300 hover:text-white transition-all">
                  <ZoomIn className="w-4 h-4"/>
                </button>
                {reSegRawMaskUrl && tab==='overlay' && (
                  <div className="absolute top-2 left-2 px-2 py-1 rounded-lg bg-indigo-600/90 text-white text-xs font-semibold">
                    ✨ Kết quả mới
                  </div>
                )}
              </div>

              {/* Opacity slider */}
              {tab === 'overlay' && (
                <div className="mt-4 px-1">
                  <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-2">
                    <span className="flex items-center gap-1"><Sliders className="w-3 h-3"/>Độ đậm Overlay</span>
                    <span className="font-semibold text-indigo-500">{opacity}%</span>
                  </div>
                  <input type="range" min="0" max="100" value={opacity}
                    onChange={e => setOpacity(Number(e.target.value))}
                    onWheel={e => { e.stopPropagation(); e.currentTarget.blur(); }}
                    className="w-full accent-indigo-500 cursor-pointer"/>
                  <div className="flex justify-between text-xs text-slate-400 mt-1">
                    <span>Ảnh gốc</span><span>Overlay thuần</span>
                  </div>
                </div>
              )}

              {/* Class toggle + color (chỉ sau khi re-seg) */}
              {reSegRawMaskUrl && (
                <div className="mt-4 space-y-3">
                  <p className="text-xs font-semibold text-slate-600 dark:text-slate-300 flex items-center gap-1"><Layers className="w-3 h-3"/>Bật/tắt &amp; Đổi màu phân lớp</p>
                  <div className="grid grid-cols-2 gap-1.5">
                    {DEFAULT_CLASSES.map(cls => {
                      const isOn = reSegActiveClasses.includes(cls.id);
                      return (
                        <div key={cls.id} className="flex items-center justify-between p-1.5 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
                          <button onClick={()=>setReSegActiveClasses(p=>isOn?p.filter(x=>x!==cls.id):[...p,cls.id])}
                            className={`p-1 rounded-md ${isOn?'bg-indigo-100 dark:bg-indigo-900/50 text-indigo-600':'bg-slate-100 dark:bg-slate-700 text-slate-400'}`}>
                            {isOn?<Eye className="w-3 h-3"/>:<EyeOff className="w-3 h-3"/>}
                          </button>
                          <span className={`text-xs flex-1 px-1 truncate ${isOn?'text-slate-700 dark:text-slate-300':'text-slate-400 line-through'}`}>{cls.name}</span>
                          <label className="w-5 h-5 rounded-full border border-slate-300 dark:border-slate-600 cursor-pointer shrink-0" style={{backgroundColor:reSegClassColors[cls.id]}}>
                            <input type="color" value={reSegClassColors[cls.id]}
                              onChange={e=>setReSegClassColors(p=>({...p,[cls.id]:e.target.value}))}
                              className="opacity-0 w-0 h-0 absolute"/>
                          </label>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            {/* Stats sidebar */}
            <div className="lg:w-56 flex-shrink-0">
              {reSegRawMaskUrl ? (
                <>
                  <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-3 flex items-center gap-1"><Palette className="w-3 h-3"/>Phân bố địa thực vật</p>
                  <div className="h-44">
                    <Pie
                      data={{
                        labels: reSegClassStats.map(c=>c.name),
                        datasets:[{data:reSegClassStats.map(c=>c.percent),backgroundColor:reSegClassStats.map(c=>reSegClassColors[c.id]),borderWidth:1}]
                      }}
                      options={{plugins:{legend:{position:'bottom',labels:{color:'#94a3b8',boxWidth:10,padding:8}},tooltip:{callbacks:{label:ctx=>` ${ctx.raw.toFixed(1)}%`}}},maintainAspectRatio:false}}
                    />
                  </div>
                </>
              ) : (
                <>
                  <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-3">Phân bố lớp</p>
                  <div className="space-y-2.5">
                    {[...displayStats].sort((a,b)=>b.percent-a.percent).map(cls=>(
                      <div key={cls.id} className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full shrink-0 border border-black/10 dark:border-white/20" style={{backgroundColor:CLASS_COLORS[cls.id]}}/>
                        <span className="text-xs text-slate-500 dark:text-slate-400 w-20 shrink-0 truncate">{cls.name}</span>
                        <div className="flex-1 bg-slate-100 dark:bg-slate-800 rounded-full h-1.5 overflow-hidden">
                          <div className="h-full rounded-full" style={{width:`${cls.percent}%`,backgroundColor:CLASS_COLORS[cls.id]}}/>
                        </div>
                        <span className="text-xs text-slate-600 dark:text-slate-300 w-10 text-right shrink-0">{cls.percent.toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
      {lb && <Lightbox src={lb} onClose={()=>setLb(null)}/>}
    </>
  );
}

// ── History Panel ──────────────────────────────────────────────────────────
export default function HistoryPanel() {
  const [items, setItems]       = useState([]);
  const [loading, setLoading]   = useState(true);
  const [detailId, setDetailId] = useState(null);
  const [deleting, setDeleting] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    const token = localStorage.getItem('token');
    axios.get(`${API}/api/history`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    })
      .then(r => setItems(r.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleDelete = async (id, e) => {
    e.stopPropagation();
    if (!window.confirm('Xóa bản ghi này khỏi lịch sử?')) return;
    setDeleting(id);
    try {
      await axios.delete(`${API}/api/history/${id}`);
      setItems(prev => prev.filter(i => i.id !== id));
    } catch { alert('Không thể xóa.'); }
    finally { setDeleting(null); }
  };

  const topClass = stats => stats?.length ? [...stats].sort((a,b)=>b.percent-a.percent)[0] : null;

  if (loading) return (
    <div className="flex flex-col items-center justify-center py-32 gap-4">
      <Loader2 className="w-10 h-10 text-indigo-500 animate-spin"/>
      <p className="text-slate-400">Đang tải lịch sử…</p>
    </div>
  );

  if (!items.length) return (
    <div className="flex flex-col items-center justify-center py-32 gap-4 text-center">
      <div className="w-20 h-20 rounded-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center border border-slate-200 dark:border-slate-700">
        <Clock className="w-8 h-8 text-slate-300 dark:text-slate-600"/>
      </div>
      <p className="text-slate-500 dark:text-slate-400 font-medium">Chưa có lịch sử phân đoạn</p>
      <p className="text-slate-400 dark:text-slate-600 text-sm">Hãy phân đoạn một ảnh để bắt đầu lưu lịch sử</p>
    </div>
  );

  return (
    <>
      <div className="flex items-center justify-between mb-6">
        <p className="text-slate-400 dark:text-slate-500 text-sm">{items.length} / 50 bản ghi gần nhất</p>
        <button onClick={load} className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors">
          <RefreshCcw className="w-3.5 h-3.5"/> Làm mới
        </button>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
        {items.map(item => {
          const top = topClass(item.class_stats);
          return (
            <div key={item.id} onClick={() => setDetailId(item.id)}
              className="group relative bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:border-indigo-400 dark:hover:border-indigo-500/60 rounded-2xl overflow-hidden cursor-pointer transition-all duration-200 hover:shadow-lg hover:shadow-indigo-100 dark:hover:shadow-indigo-900/20 hover:-translate-y-0.5">
              {/* Thumbnail */}
              <div className="aspect-square overflow-hidden">
                <img src={`data:image/png;base64,${item.original_thumb}`} alt={item.filename}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"/>
              </div>

              {/* Hover overlay */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex flex-col justify-end p-2.5">
                <button onClick={e=>handleDelete(item.id,e)} disabled={deleting===item.id}
                  className="absolute top-2 right-2 p-1.5 bg-red-900/70 hover:bg-red-600 border border-red-800 hover:border-red-500 text-red-300 hover:text-white rounded-lg transition-all">
                  {deleting===item.id ? <Loader2 className="w-3.5 h-3.5 animate-spin"/> : <Trash2 className="w-3.5 h-3.5"/>}
                </button>
                <ZoomIn className="w-4 h-4 text-white/70 mb-1 mx-auto"/>
                <p className="text-white text-xs font-medium text-center">Xem chi tiết</p>
              </div>

              {/* Card footer */}
              <div className="p-2.5">
                <p className="text-xs text-slate-700 dark:text-slate-300 font-medium truncate" title={item.filename}>{item.filename}</p>
                <p className="text-xs text-slate-400 dark:text-slate-600 mt-0.5">{item.created_at}</p>
                {top && (
                  <div className="flex items-center gap-1.5 mt-1.5">
                    <span className="w-2 h-2 rounded-full border border-black/10 dark:border-white/20" style={{backgroundColor:CLASS_COLORS[top.id]}}/>
                    <span className="text-xs text-slate-400 dark:text-slate-500 truncate">{top.name} {top.percent.toFixed(0)}%</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {detailId && <DetailModal recordId={detailId} onClose={()=>setDetailId(null)}/>}
    </>
  );
}
