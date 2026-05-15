import React from 'react';
import { Layers, Sliders, Palette, Eye, EyeOff } from 'lucide-react';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

export default function AdvancedControls({
  classes,
  activeClasses,
  toggleActiveClass,
  classColors,
  changeClassColor,
  opacity,
  setOpacity,
  classStats
}) {
  // Chuẩn bị dữ liệu cho biểu đồ Pie
  const chartData = {
    labels: classStats?.map(c => c.name) || [],
    datasets: [
      {
        data: classStats?.map(c => c.percent) || [],
        backgroundColor: classStats?.map(c => classColors[c.id]) || [],
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    plugins: {
      legend: { position: 'bottom', labels: { color: '#94a3b8', boxWidth: 12, padding: 15 } },
      tooltip: { callbacks: { label: (ctx) => ` ${ctx.raw.toFixed(2)}%` } }
    },
    maintainAspectRatio: false,
  };

  return (
    <div className="bg-slate-50 dark:bg-slate-900 rounded-2xl p-6 border border-slate-200 dark:border-slate-800 space-y-6">
      
      {/* 1. Thanh trượt Opacity */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-2 mb-3">
          <Sliders className="w-4 h-4" /> Độ đậm nhạt (Opacity)
        </h3>
        <input 
          type="range" 
          min="0" max="100" 
          value={opacity} 
          onChange={(e) => setOpacity(Number(e.target.value))}
          className="w-full accent-indigo-500"
        />
        <div className="flex justify-between text-xs text-slate-500 mt-1">
          <span>0% (Chỉ ảnh gốc)</span>
          <span>{opacity}%</span>
          <span>100% (Chỉ Mask)</span>
        </div>
      </div>

      <hr className="border-slate-200 dark:border-slate-800" />

      {/* 2. Điều khiển Class: Bật/tắt và Đổi màu */}
      <div>
        <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-2 mb-4">
          <Layers className="w-4 h-4" /> Bật/tắt & Đổi màu phân lớp
        </h3>
        <div className="grid grid-cols-2 gap-2">
          {classes.map(cls => {
            const isActive = activeClasses.includes(cls.id);
            return (
              <div key={cls.id} className="flex items-center justify-between p-2 rounded-xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 min-w-0">
                <div className="flex items-center gap-1.5 flex-1 min-w-0 pr-1">
                  <button 
                    onClick={() => toggleActiveClass(cls.id)}
                    className={`shrink-0 p-1.5 rounded-lg transition-colors ${isActive ? 'bg-indigo-100 dark:bg-indigo-900/50 text-indigo-600 dark:text-indigo-400' : 'bg-slate-100 dark:bg-slate-700 text-slate-400'}`}
                  >
                    {isActive ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                  </button>
                  <span className={`text-xs md:text-sm truncate block ${isActive ? 'text-slate-700 dark:text-slate-300' : 'text-slate-400 dark:text-slate-500 line-through'}`} title={cls.name}>
                    {cls.name}
                  </span>
                </div>
                <label 
                  className="w-6 h-6 rounded-full border border-slate-300 dark:border-slate-600 shrink-0 cursor-pointer shadow-sm transition-transform hover:scale-110" 
                  style={{ backgroundColor: classColors[cls.id] }} 
                  title={`Đổi màu ${cls.name}`}
                >
                  <input 
                    type="color" 
                    value={classColors[cls.id]}
                    onChange={(e) => changeClassColor(cls.id, e.target.value)}
                    className="opacity-0 w-0 h-0 absolute"
                  />
                </label>
              </div>
            );
          })}
        </div>
      </div>

      <hr className="border-slate-200 dark:border-slate-800" />

      {/* 3. Biểu đồ Metrics trực quan */}
      {classStats && classStats.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-2 mb-4">
            <Palette className="w-4 h-4" /> Phân bố địa thực vật
          </h3>
          <div className="h-48 flex justify-center">
            <Pie data={chartData} options={chartOptions} />
          </div>
        </div>
      )}
    </div>
  );
}
