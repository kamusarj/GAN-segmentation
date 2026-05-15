import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Users, Shield, Star, Trash2, Loader2, RefreshCcw, Crown, User as UserIcon } from 'lucide-react';

const API = 'http://localhost:8000';

const ROLE_CONFIG = {
  admin: { label: 'Admin', color: 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400', icon: Crown },
  user: { label: 'User', color: 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400', icon: UserIcon },
};

export default function AdminUserManagement() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState(null); // user_id đang được cập nhật
  const [deleting, setDeleting] = useState(null);
  const [toast, setToast] = useState(null);

  const showToast = (msg, type = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${API}/api/admin/users`);
      setUsers(res.data);
    } catch {
      showToast('Không thể tải danh sách người dùng.', 'error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleUpdate = async (userId, patch) => {
    setUpdating(userId);
    try {
      await axios.patch(`${API}/api/admin/users/${userId}`, patch);
      setUsers(prev => prev.map(u =>
        u.id === userId ? { ...u, ...patch } : u
      ));
      showToast('Cập nhật thành công!');
    } catch (err) {
      showToast(err.response?.data?.detail || 'Cập nhật thất bại.', 'error');
    } finally {
      setUpdating(null);
    }
  };

  const handleDelete = async (userId, username) => {
    if (!window.confirm(`Xóa tài khoản "${username}"?\nThao tác này sẽ xóa toàn bộ lịch sử phân đoạn của tài khoản này.`)) return;
    setDeleting(userId);
    try {
      await axios.delete(`${API}/api/admin/users/${userId}`);
      setUsers(prev => prev.filter(u => u.id !== userId));
      showToast(`Đã xóa tài khoản "${username}".`);
    } catch (err) {
      showToast(err.response?.data?.detail || 'Không thể xóa tài khoản.', 'error');
    } finally {
      setDeleting(null);
    }
  };

  return (
    <div className="space-y-4">
      {/* Toast */}
      {toast && (
        <div className={`fixed top-6 right-6 z-50 px-5 py-3 rounded-xl shadow-2xl text-sm font-medium transition-all duration-300 ${toast.type === 'success' ? 'bg-emerald-600 text-white' : 'bg-red-600 text-white'}`}>
          {toast.msg}
        </div>
      )}

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-sm">
          <Users className="w-4 h-4" /> {users.length} tài khoản
        </div>
        <button onClick={load} className="flex items-center gap-1.5 text-sm text-indigo-500 hover:text-indigo-600 font-medium">
          <RefreshCcw className="w-4 h-4" /> Làm mới
        </button>
      </div>

      {loading ? (
        <div className="flex justify-center py-16">
          <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
        </div>
      ) : (
        <div className="overflow-x-auto rounded-2xl border border-slate-200 dark:border-slate-700">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 dark:bg-slate-800/60">
              <tr>
                <th className="px-4 py-3 text-left font-semibold text-slate-600 dark:text-slate-300">Người dùng</th>
                <th className="px-4 py-3 text-left font-semibold text-slate-600 dark:text-slate-300">Ngày tạo</th>
                <th className="px-4 py-3 text-center font-semibold text-slate-600 dark:text-slate-300">Vai trò</th>
                <th className="px-4 py-3 text-center font-semibold text-slate-600 dark:text-slate-300">Quyền Premium</th>
                <th className="px-4 py-3 text-center font-semibold text-slate-600 dark:text-slate-300">Hành động</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {users.map(u => {
                const roleConf = ROLE_CONFIG[u.role] || ROLE_CONFIG.user;
                const RoleIcon = roleConf.icon;
                const isUpdating = updating === u.id;
                const isDeleting = deleting === u.id;
                const isAdmin = u.role === 'admin';
                // Admin mặc định luôn có quyền premium
                const isPremium = isAdmin || Boolean(u.is_premium);

                return (
                  <tr key={u.id} className="bg-white dark:bg-slate-900 hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors">
                    {/* Tên & ID */}
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${u.role === 'admin' ? 'bg-red-100 dark:bg-red-900/50' : 'bg-indigo-100 dark:bg-indigo-900/50'}`}>
                          <RoleIcon className={`w-4 h-4 ${u.role === 'admin' ? 'text-red-600 dark:text-red-400' : 'text-indigo-600 dark:text-indigo-400'}`} />
                        </div>
                        <div>
                          <p className="font-semibold text-slate-800 dark:text-slate-200">{u.username}</p>
                          <p className="text-xs text-slate-400">ID #{u.id}</p>
                        </div>
                      </div>
                    </td>

                    {/* Ngày tạo */}
                    <td className="px-4 py-3 text-slate-500 dark:text-slate-400 text-xs">{u.created_at}</td>

                    {/* Role toggle */}
                    <td className="px-4 py-3 text-center">
                      {isUpdating ? (
                        <Loader2 className="w-5 h-5 animate-spin mx-auto text-indigo-500" />
                      ) : (
                        <button
                          onClick={() => handleUpdate(u.id, { role: u.role === 'admin' ? 'user' : 'admin' })}
                          className={`px-3 py-1 rounded-full text-xs font-semibold flex items-center gap-1 mx-auto transition-all hover:scale-105 ${roleConf.color}`}
                          title={u.role === 'admin' ? 'Hạ xuống User' : 'Nâng lên Admin'}
                        >
                          <RoleIcon className="w-3 h-3" /> {roleConf.label}
                        </button>
                      )}
                    </td>

                    {/* Premium toggle */}
                    <td className="px-4 py-3 text-center">
                      {isUpdating ? (
                        <Loader2 className="w-5 h-5 animate-spin mx-auto text-indigo-500" />
                      ) : isAdmin ? (
                        // Admin luôn được bao gồm quyền Premium — hiển thị cố định, không cần gạt
                        <div className="flex flex-col items-center gap-0.5">
                          <span className="inline-flex h-6 w-11 items-center rounded-full bg-amber-400 mx-auto opacity-70 cursor-default">
                            <span className="inline-block w-4 h-4 translate-x-6 rounded-full bg-white shadow-md" />
                          </span>
                          <span className="text-xs text-amber-500 font-medium">Bao gồm</span>
                        </div>
                      ) : (
                        <>
                          <button
                            onClick={() => handleUpdate(u.id, { is_premium: !isPremium })}
                            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 focus:outline-none mx-auto ${isPremium ? 'bg-amber-400' : 'bg-slate-300 dark:bg-slate-600'}`}
                            title={isPremium ? 'Thu hồi quyền Premium' : 'Cấp quyền Premium (mở khóa Bản đồ)'}
                          >
                            <span className={`inline-block w-4 h-4 transform rounded-full bg-white shadow-md transition-transform duration-200 ${isPremium ? 'translate-x-6' : 'translate-x-1'}`} />
                          </button>
                          {isPremium && (
                            <div className="flex items-center justify-center gap-1 mt-1 text-xs text-amber-500">
                              <Star className="w-3 h-3 fill-amber-400" /> Đã cấp
                            </div>
                          )}
                        </>
                      )}
                    </td>

                    {/* Xóa */}
                    <td className="px-4 py-3 text-center">
                      <button
                        onClick={() => handleDelete(u.id, u.username)}
                        disabled={isDeleting}
                        className="p-2 rounded-xl bg-red-50 dark:bg-red-900/20 text-red-500 hover:bg-red-100 dark:hover:bg-red-900/40 transition-colors disabled:opacity-50"
                        title="Xóa tài khoản"
                      >
                        {isDeleting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <p className="text-xs text-slate-400 dark:text-slate-600 text-center">
        💡 Nhấn vào nhãn <strong>Vai trò</strong> để nâng/hạ quyền. Nhấn nút gạt <Star className="inline w-3 h-3 fill-amber-400 text-amber-400" /> để cấp/thu hồi quyền Premium (mở khóa tính năng Bản đồ vệ tinh).
      </p>
    </div>
  );
}
