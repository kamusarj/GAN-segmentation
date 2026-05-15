import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Settings, Save, Loader2, ArrowLeft, Cpu, Users } from 'lucide-react';
import { Link } from 'react-router-dom';
import AdminUserManagement from '../components/AdminUserManagement';

export default function Admin() {
  const [tab, setTab] = useState('model'); // 'model' | 'users'

  // ── Model state ──────────────────────────────────────────────────────────────
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  useEffect(() => { fetchModels(); }, []);

  const fetchModels = async () => {
    try {
      const res = await axios.get('http://localhost:8000/api/models');
      setModels(res.data.models);
      // Ưu tiên model đang active trên backend; fallback về model đầu tiên
      const active = res.data.active_model;
      if (active && res.data.models.includes(active)) {
        setSelectedModel(active);
      } else if (res.data.models.length > 0) {
        setSelectedModel(res.data.models[0]);
      }
    } catch {
      setError('Lỗi lấy danh sách model');
    }
  };

  const handleSwitchModel = async () => {
    setLoading(true); setMessage(''); setError('');
    try {
      const res = await axios.post('http://localhost:8000/api/models/switch', { model_name: selectedModel });
      setMessage(res.data.message);
    } catch (err) {
      setError(err.response?.data?.detail || 'Lỗi đổi model');
    } finally { setLoading(false); }
  };

  const TABS = [
    { key: 'model', label: 'Quản lý Model', icon: Cpu },
    { key: 'users', label: 'Quản lý Người dùng', icon: Users },
  ];

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 p-6">
      <div className="max-w-4xl mx-auto">
        <Link to="/" className="inline-flex items-center gap-2 text-indigo-500 hover:text-indigo-600 mb-6 font-medium">
          <ArrowLeft className="w-4 h-4" /> Về Dashboard
        </Link>

        <div className="bg-white dark:bg-slate-900 rounded-3xl p-8 border border-slate-200 dark:border-slate-800 shadow-xl">
          {/* Header */}
          <div className="flex items-center gap-4 mb-8">
            <div className="p-3 bg-red-100 dark:bg-red-900/30 text-red-500 rounded-2xl">
              <Settings className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Admin Control Panel</h1>
              <p className="text-slate-500">Quản lý hệ thống và người dùng</p>
            </div>
          </div>

          {/* Tab navigation */}
          <div className="flex gap-2 mb-8 bg-slate-100 dark:bg-slate-800 rounded-2xl p-1">
            {TABS.map(t => {
              const Icon = t.icon;
              return (
                <button
                  key={t.key}
                  onClick={() => setTab(t.key)}
                  className={`flex-1 py-2.5 rounded-xl flex items-center justify-center gap-2 text-sm font-semibold transition-all ${tab === t.key ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-sm' : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'}`}
                >
                  <Icon className="w-4 h-4" /> {t.label}
                </button>
              );
            })}
          </div>

          {/* Tab: Model Management */}
          {tab === 'model' && (
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                  Chọn Model phân đoạn:
                </label>
                <select
                  value={selectedModel}
                  onChange={e => setSelectedModel(e.target.value)}
                  className="w-full bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl px-4 py-3 text-slate-900 dark:text-white outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  {models.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>

              <button
                onClick={handleSwitchModel}
                disabled={loading}
                className="px-6 py-3 bg-gradient-to-r from-red-500 to-orange-500 hover:from-red-400 hover:to-orange-400 text-white font-bold rounded-xl flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Save className="w-5 h-5" />}
                Lưu thay đổi model
              </button>

              {message && <div className="p-4 bg-emerald-50 dark:bg-emerald-900/30 text-emerald-600 border border-emerald-200 dark:border-emerald-800 rounded-xl">{message}</div>}
              {error   && <div className="p-4 bg-red-50    dark:bg-red-900/30    text-red-600    border border-red-200    dark:border-red-800    rounded-xl">{error}</div>}
            </div>
          )}

          {/* Tab: User Management */}
          {tab === 'users' && <AdminUserManagement />}
        </div>
      </div>
    </div>
  );
}
