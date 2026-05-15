import React, { useState, useContext, useMemo } from 'react';
import { createPortal } from 'react-dom';
import axios from 'axios';
import {
  X, User, Crown, Star, Lock, Eye, EyeOff,
  CheckCircle, XCircle, KeyRound, ShieldCheck, Loader2
} from 'lucide-react';
import { AuthContext } from '../AuthContext';

const API = 'http://localhost:8000';

// Quy tắc mật khẩu (giống Register)
const RULES = [
  { id: 'len',     label: 'Ít nhất 8 ký tự',              test: p => p.length >= 8 },
  { id: 'upper',   label: 'Có ít nhất 1 chữ HOA (A-Z)',    test: p => /[A-Z]/.test(p) },
  { id: 'lower',   label: 'Có ít nhất 1 chữ thường (a-z)', test: p => /[a-z]/.test(p) },
  { id: 'digit',   label: 'Có ít nhất 1 chữ số (0-9)',     test: p => /[0-9]/.test(p) },
  { id: 'special', label: 'Có ít nhất 1 ký tự đặc biệt',  test: p => /[^A-Za-z0-9]/.test(p) },
];

function getStrength(password) {
  const passed = RULES.filter(r => r.test(password)).length;
  if (!password) return null;
  if (passed <= 2) return { level: 'Yếu',      score: 1, color: 'bg-red-500',    text: 'text-red-500' };
  if (passed === 3) return { level: 'Trung bình', score: 2, color: 'bg-yellow-400', text: 'text-yellow-500' };
  if (passed === 4) return { level: 'Khá mạnh',  score: 3, color: 'bg-blue-500',  text: 'text-blue-500' };
  return { level: 'Mạnh',     score: 4, color: 'bg-emerald-500', text: 'text-emerald-500' };
}

const ROLE_BADGE = {
  admin: { label: 'Admin',   style: 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400',       icon: Crown },
  user:  { label: 'User',    style: 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400',  icon: User },
};

export default function ProfileModal({ onClose }) {
  const { user } = useContext(AuthContext);
  const [tab, setTab] = useState('info'); // 'info' | 'password'

  // ── Change password state ───────────────────────────────────────────────────
  const [currentPw, setCurrentPw] = useState('');
  const [newPw,     setNewPw]     = useState('');
  const [showCurr,  setShowCurr]  = useState(false);
  const [showNew,   setShowNew]   = useState(false);
  const [pwLoading, setPwLoading] = useState(false);
  const [pwMsg,     setPwMsg]     = useState(null); // {text, ok}

  const strength = useMemo(() => getStrength(newPw), [newPw]);
  const allRulesPassed = RULES.every(r => r.test(newPw));

  const handleChangePw = async (e) => {
    e.preventDefault();
    if (!allRulesPassed) {
      setPwMsg({ text: 'Mật khẩu mới chưa đủ mạnh.', ok: false });
      return;
    }
    setPwLoading(true);
    setPwMsg(null);
    try {
      const res = await axios.post(`${API}/api/me/change-password`, {
        current_password: currentPw,
        new_password:     newPw,
      });
      setPwMsg({ text: res.data.message, ok: true });
      setCurrentPw('');
      setNewPw('');
    } catch (err) {
      setPwMsg({ text: err.response?.data?.detail || 'Đổi mật khẩu thất bại.', ok: false });
    } finally {
      setPwLoading(false);
    }
  };

  const RoleConf = ROLE_BADGE[user?.role] || ROLE_BADGE.user;
  const RoleIcon = RoleConf.icon;
  const isPremium = user?.role === 'admin' || user?.is_premium;

  return createPortal(
    // Backdrop
    <div
      className="fixed inset-0 z-[9998] flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.60)', backdropFilter: 'blur(6px)' }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl w-full max-w-md shadow-2xl">
        
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-slate-100 dark:border-slate-800">
          <h2 className="text-lg font-bold text-slate-900 dark:text-white">Tài khoản của tôi</h2>
          <button onClick={onClose} className="p-2 rounded-xl text-slate-400 hover:text-slate-700 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab switcher */}
        <div className="flex gap-1 mx-6 mt-5 bg-slate-100 dark:bg-slate-800 rounded-xl p-1">
          {[
            { key: 'info',     label: 'Thông tin',     icon: User },
            { key: 'password', label: 'Đổi mật khẩu',  icon: KeyRound },
          ].map(t => {
            const Icon = t.icon;
            return (
              <button
                key={t.key}
                onClick={() => { setTab(t.key); setPwMsg(null); }}
                className={`flex-1 py-2 rounded-lg text-sm font-semibold flex items-center justify-center gap-2 transition-all ${tab === t.key ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm' : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'}`}
              >
                <Icon className="w-4 h-4" /> {t.label}
              </button>
            );
          })}
        </div>

        <div className="px-6 py-6">

          {/* ── Tab: Thông tin tài khoản ─────────────────────────────────── */}
          {tab === 'info' && (
            <div className="space-y-4">
              {/* Avatar & name */}
              <div className="flex items-center gap-4 p-4 bg-slate-50 dark:bg-slate-800 rounded-2xl">
                <div className={`w-14 h-14 rounded-2xl flex items-center justify-center text-2xl font-bold text-white shrink-0 ${user?.role === 'admin' ? 'bg-gradient-to-br from-red-500 to-orange-500' : 'bg-gradient-to-br from-indigo-500 to-purple-600'}`}>
                  {user?.username?.[0]?.toUpperCase()}
                </div>
                <div>
                  <p className="text-lg font-bold text-slate-900 dark:text-white">{user?.username}</p>
                  <p className="text-xs text-slate-400">@{user?.username}</p>
                </div>
              </div>

              {/* Role */}
              <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-800 rounded-2xl">
                <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                  <ShieldCheck className="w-4 h-4" /> Vai trò
                </div>
                <span className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold ${RoleConf.style}`}>
                  <RoleIcon className="w-3 h-3" /> {RoleConf.label}
                </span>
              </div>

              {/* Premium */}
              <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-800 rounded-2xl">
                <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                  <Star className="w-4 h-4" /> Quyền Premium
                </div>
                {isPremium ? (
                  <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400">
                    <Star className="w-3 h-3 fill-amber-400" />
                    {user?.role === 'admin' ? 'Bao gồm (Admin)' : 'Đã kích hoạt'}
                  </span>
                ) : (
                  <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-slate-200 dark:bg-slate-700 text-slate-500 dark:text-slate-400">
                    🔒 Chưa kích hoạt
                  </span>
                )}
              </div>

              {/* Tính năng có được */}
              <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-2xl">
                <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-2">Tính năng hiện có:</p>
                <div className="space-y-1.5">
                  {[
                    { label: 'Phân đoạn ảnh tải lên',  ok: true },
                    { label: 'Lưu lịch sử phân đoạn',  ok: true },
                    { label: 'Bản đồ vệ tinh',          ok: isPremium },
                    { label: 'Admin Control Panel',     ok: user?.role === 'admin' },
                  ].map(f => (
                    <div key={f.label} className="flex items-center gap-2">
                      {f.ok
                        ? <CheckCircle className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                        : <XCircle    className="w-3.5 h-3.5 text-slate-400 shrink-0" />}
                      <span className={`text-xs ${f.ok ? 'text-slate-700 dark:text-slate-300' : 'text-slate-400 dark:text-slate-600'}`}>
                        {f.label}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ── Tab: Đổi mật khẩu ────────────────────────────────────────── */}
          {tab === 'password' && (
            <form onSubmit={handleChangePw} className="space-y-4">
              {pwMsg && (
                <div className={`p-3 rounded-xl text-sm text-center border ${pwMsg.ok ? 'bg-emerald-50 dark:bg-emerald-900/20 text-emerald-600 border-emerald-200 dark:border-emerald-800' : 'bg-red-50 dark:bg-red-900/20 text-red-600 border-red-200 dark:border-red-800'}`}>
                  {pwMsg.text}
                </div>
              )}

              {/* Mật khẩu hiện tại */}
              <div>
                <label className="block text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1.5 pl-1">Mật khẩu hiện tại</label>
                <div className="relative">
                  <Lock className="w-4 h-4 absolute left-3 top-3.5 text-slate-400" />
                  <input
                    type={showCurr ? 'text' : 'password'}
                    value={currentPw}
                    onChange={e => setCurrentPw(e.target.value)}
                    placeholder="Nhập mật khẩu hiện tại"
                    className="w-full pl-9 pr-10 py-3 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white text-sm transition-all"
                    required
                  />
                  <button type="button" onClick={() => setShowCurr(v => !v)} className="absolute right-3 top-3.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200">
                    {showCurr ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              {/* Mật khẩu mới */}
              <div>
                <label className="block text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1.5 pl-1">Mật khẩu mới</label>
                <div className="relative">
                  <Lock className="w-4 h-4 absolute left-3 top-3.5 text-slate-400" />
                  <input
                    type={showNew ? 'text' : 'password'}
                    value={newPw}
                    onChange={e => setNewPw(e.target.value)}
                    placeholder="Nhập mật khẩu mới"
                    className="w-full pl-9 pr-10 py-3 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white text-sm transition-all"
                    required
                  />
                  <button type="button" onClick={() => setShowNew(v => !v)} className="absolute right-3 top-3.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200">
                    {showNew ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>

                {/* Strength meter */}
                {strength && (
                  <div className="mt-3 space-y-2">
                    <div className="flex items-center gap-3">
                      <div className="flex-1 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                        <div className={`h-full rounded-full transition-all duration-500 ${strength.color}`} style={{ width: `${(strength.score / 4) * 100}%` }} />
                      </div>
                      <span className={`text-xs font-bold shrink-0 ${strength.text}`}>{strength.level}</span>
                    </div>
                    <div className="bg-slate-50 dark:bg-slate-800/60 rounded-xl p-3 space-y-1.5">
                      {RULES.map(rule => {
                        const ok = rule.test(newPw);
                        return (
                          <div key={rule.id} className="flex items-center gap-2">
                            {ok ? <CheckCircle className="w-3 h-3 text-emerald-500 shrink-0" /> : <XCircle className="w-3 h-3 text-slate-400 dark:text-slate-600 shrink-0" />}
                            <span className={`text-xs ${ok ? 'text-emerald-600 dark:text-emerald-400 line-through decoration-emerald-400/60' : 'text-slate-500 dark:text-slate-400'}`}>{rule.label}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

              <button
                type="submit"
                disabled={pwLoading || !allRulesPassed || !currentPw}
                className="w-full py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:opacity-90 text-white font-bold text-sm flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed transition-opacity"
              >
                {pwLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <KeyRound className="w-4 h-4" />}
                Xác nhận đổi mật khẩu
              </button>
            </form>
          )}
        </div>
      </div>
    </div>,
    document.body
  );
}
