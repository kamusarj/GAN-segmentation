import React, { useState, useMemo } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import { Loader2, Lock, User, UserPlus, Eye, EyeOff, CheckCircle, XCircle } from 'lucide-react';

// ── Kiểm tra độ mạnh mật khẩu ────────────────────────────────────────────────
const RULES = [
  { id: 'len',     label: 'Ít nhất 8 ký tự',             test: p => p.length >= 8 },
  { id: 'upper',   label: 'Có ít nhất 1 chữ HOA (A-Z)',   test: p => /[A-Z]/.test(p) },
  { id: 'lower',   label: 'Có ít nhất 1 chữ thường (a-z)',test: p => /[a-z]/.test(p) },
  { id: 'digit',   label: 'Có ít nhất 1 chữ số (0-9)',    test: p => /[0-9]/.test(p) },
  { id: 'special', label: 'Có ít nhất 1 ký tự đặc biệt (!@#$...)', test: p => /[^A-Za-z0-9]/.test(p) },
];

function getStrength(password) {
  const passed = RULES.filter(r => r.test(password)).length;
  if (password.length === 0) return null;
  if (passed <= 2) return { level: 'Yếu',    score: 1, color: 'bg-red-500',    textColor: 'text-red-500' };
  if (passed === 3) return { level: 'Trung bình', score: 2, color: 'bg-yellow-400', textColor: 'text-yellow-500' };
  if (passed === 4) return { level: 'Khá mạnh', score: 3, color: 'bg-blue-500',  textColor: 'text-blue-500' };
  return { level: 'Mạnh',   score: 4, color: 'bg-emerald-500', textColor: 'text-emerald-500' };
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function Register() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const strength = useMemo(() => getStrength(password), [password]);
  const allPassed = RULES.every(r => r.test(password));

  const handleRegister = async (e) => {
    e.preventDefault();
    setError('');
    if (!allPassed) {
      setError('Mật khẩu chưa đủ mạnh. Vui lòng thỏa mãn tất cả yêu cầu bên dưới.');
      return;
    }
    setLoading(true);
    try {
      await axios.post('http://localhost:8000/api/auth/register', { username, password });
      navigate('/login');
    } catch (err) {
      setError(err.response?.data?.detail || 'Lỗi đăng ký');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-950 px-4 py-8">
      <div className="max-w-md w-full bg-white dark:bg-slate-900 rounded-3xl shadow-xl border border-slate-200 dark:border-slate-800 p-8 space-y-6">
        
        {/* Header */}
        <div className="text-center">
          <div className="w-16 h-16 bg-gradient-to-br from-indigo-500 to-emerald-500 rounded-2xl mx-auto flex items-center justify-center mb-4 shadow-lg shadow-indigo-200 dark:shadow-indigo-900/30">
            <UserPlus className="text-white w-8 h-8" />
          </div>
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white">Tạo tài khoản mới</h2>
          <p className="text-slate-500 text-sm mt-1">Điền thông tin để bắt đầu sử dụng</p>
        </div>

        {/* Error */}
        {error && (
          <div className="p-3 bg-red-50 text-red-600 rounded-xl text-sm text-center border border-red-200 dark:bg-red-900/20 dark:border-red-800 dark:text-red-400">
            {error}
          </div>
        )}

        <form onSubmit={handleRegister} className="space-y-5">
          
          {/* Username */}
          <div>
            <label className="block text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1.5 pl-1">Tên đăng nhập</label>
            <div className="relative">
              <User className="w-5 h-5 absolute left-3 top-3 text-slate-400" />
              <input
                type="text"
                placeholder="vd: nguyen_van_a"
                value={username}
                onChange={e => setUsername(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white transition-all"
                required
                minLength={3}
                maxLength={20}
                pattern="^[a-zA-Z0-9_]+$"
                title="Từ 3-20 ký tự, chỉ gồm chữ cái, số hoặc dấu gạch dưới (_)"
              />
            </div>
            <p className="text-xs text-slate-400 dark:text-slate-500 pl-1 mt-1.5">3–20 ký tự, chỉ chữ cái, số và dấu gạch dưới (_)</p>
          </div>

          {/* Password */}
          <div>
            <label className="block text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1.5 pl-1">Mật khẩu</label>
            <div className="relative">
              <Lock className="w-5 h-5 absolute left-3 top-3 text-slate-400" />
              <input
                type={showPassword ? 'text' : 'password'}
                placeholder="Nhập mật khẩu mạnh..."
                value={password}
                onChange={e => setPassword(e.target.value)}
                className="w-full pl-10 pr-12 py-3 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white transition-all"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-3 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
              >
                {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
              </button>
            </div>

            {/* Strength Meter */}
            {strength && (
              <div className="mt-3 space-y-2">
                {/* Bar + Label */}
                <div className="flex items-center gap-3">
                  <div className="flex-1 h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${strength.color}`}
                      style={{ width: `${(strength.score / 4) * 100}%` }}
                    />
                  </div>
                  <span className={`text-xs font-bold shrink-0 ${strength.textColor}`}>
                    {strength.level}
                  </span>
                </div>

                {/* Checklist */}
                <div className="bg-slate-50 dark:bg-slate-800/60 rounded-xl p-3 space-y-1.5">
                  {RULES.map(rule => {
                    const ok = rule.test(password);
                    return (
                      <div key={rule.id} className="flex items-center gap-2">
                        {ok
                          ? <CheckCircle className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                          : <XCircle    className="w-3.5 h-3.5 text-slate-400 dark:text-slate-600 shrink-0" />
                        }
                        <span className={`text-xs ${ok ? 'text-emerald-600 dark:text-emerald-400 line-through decoration-emerald-400/60' : 'text-slate-500 dark:text-slate-400'}`}>
                          {rule.label}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={loading || !allPassed}
            className="w-full bg-gradient-to-r from-indigo-600 to-emerald-600 hover:opacity-90 text-white font-bold py-3 rounded-xl flex items-center justify-center gap-2 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <UserPlus className="w-5 h-5" />}
            Đăng ký
          </button>
        </form>

        <p className="text-center text-slate-500 text-sm">
          Đã có tài khoản?{' '}
          <Link to="/login" className="text-indigo-500 font-semibold hover:underline">Đăng nhập</Link>
        </p>
      </div>
    </div>
  );
}
