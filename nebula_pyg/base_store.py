# nebula_pyg/core/base_store.py
from __future__ import annotations
import os
import threading
from abc import ABC
from typing import Callable, Optional

class NebulaStoreBase(ABC):
    def __init__(
        self,
        pool_factory: Callable[[], "ConnectionPool"],
        sclient_factory: Optional[Callable[[], "GraphStorageClient"]],
        space: str,
        username: str = "root",
        password: str = "nebula",
        missing_value = 0,
    ):
        self._pool_factory = pool_factory
        self._sclient_factory = sclient_factory
        self.space = space
        self.username = username
        self.password = password

        # 懒创建：不要在 __init__ 里建连接
        self._pool = None
        self._sclient = None
        self._sess_local = threading.local()
        self._pid = os.getpid()

        # 公共缓存/配置
        self._numeric_cols_by_tag: dict[str, list[str]] = {}
        self._x_cols: dict[str, list[str]] = {}
        self._reserved_cols = {"x", "y", "label", "category", "target"}
        self._missing_value = missing_value

    # ---------- 生命周期与（重）建 ----------
    def _ensure_pool(self):
        """确保当前进程内已创建 pool/sclient；若进程变更则重建。"""
        cur_pid = os.getpid()
        if (self._pool is None) or (self._pid != cur_pid):
            # 旧进程的资源作废
            self._release_session_only()
            self._pool = self._pool_factory()
            self._sclient = self._sclient_factory() if self._sclient_factory else None
            self._sess_local = threading.local()  # 线程隔离重置
            self._pid = cur_pid
        return self._pool

    def _ensure_session(self):
        """线程本地的 session，懒创建；进程切换时自动失效重建。"""
        self._ensure_pool()
        sess = getattr(self._sess_local, "sess", None)
        if sess is None:
            sess = self._pool.get_session(self.username, self.password)
            self._sess_local.sess = sess
        return sess

    def _ensure_session_fast(self):
        # 99% 的请求会命中这里：已有 sess，且 pid 未变
        sess = getattr(self._sess_local, "sess", None)
        if sess is not None and self._pid == os.getpid() and self._pool is not None:
            return sess
        # 否则再走完整的 _ensure_session（含进程切换处理/懒创建）
        return self._ensure_session()

    # ---------- 执行 ----------
    def _execute(self, ngql: str, check: bool = True):
        """
        统一执行入口：自动 USE <space>; 失败自动重建一次；可选错误检查。
        返回 nebula3.ResultSet
        """
        sess = self._ensure_session_fast()
        try:
            result = sess.execute(f'USE {self.space}; {ngql}')
        except Exception:
            # session 失效，重建后重试一次
            self._rebuild_session()
            sess = self._ensure_session_fast()
            result = sess.execute(f'USE {self.space}; {ngql}')

        if check and not result.is_succeeded():
            raise RuntimeError(
                "Nebula NGQL failed.\n"
                f"NGQL: {ngql}\n"
                f"error_msg: {result.error_msg()}\n"
                f"latency(ms): {result.latency()}"
            )
        return result

    # ---------- 清理 ----------
    def _release_session_only(self):
        sess = getattr(self._sess_local, "sess", None)
        if sess is not None:
            try:
                sess.release()
            except Exception:
                pass
            finally:
                self._sess_local.sess = None

    def _rebuild_session(self):
        """仅重建当前线程的 session（pool 保持）。"""
        self._release_session_only()

    def close(self):
        """手动关闭（训练结束时可调用）。"""
        self._release_session_only()
        self._pool = None
        self._sclient = None

    # ---------- 进程间序列化 ----------
    def __getstate__(self):
        d = self.__dict__.copy()
        # 不把活连接/会话带进子进程
        d["_pool"] = None
        d["_sclient"] = None
        d["_sess_local"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._sess_local = threading.local()
        self._pid = os.getpid()

    # ---------- 公共工具（可在子类里用） ----------
    def _normalize_scalar(self, v):
        if v is None:
            return self._missing_value
        if isinstance(v, bool):
            return int(v)
        return v

    def _ensure_x_cols(self, tag: str) -> list[str]:
        if tag in self._x_cols:
            return self._x_cols[tag]
        cols = self._numeric_cols_by_tag.get(tag) or self._collect_numeric_cols(tag)
        x_cols = sorted([c for c in cols if c not in self._reserved_cols])
        self._x_cols[tag] = x_cols
        return x_cols

    def _collect_numeric_cols(self, tag: str) -> list[str]:
        """建议子类覆写：默认依赖 self._sclient._meta_cache。"""
        raise NotImplementedError("Subclasses should implement _collect_numeric_cols(tag).")

    # 便捷访问（给子类用）
    @property
    def pool(self):
        self._ensure_pool()
        return self._pool

    @property
    def sclient(self):
        self._ensure_pool()
        return self._sclient
