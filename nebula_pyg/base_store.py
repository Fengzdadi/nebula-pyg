# nebula_pyg/core/base_store.py
from __future__ import annotations
import os
import threading
from abc import ABC
from typing import Callable, Optional
from nebula_pyg.utils.vid_codec import VidCodec

class NebulaStoreBase(ABC):
    """
    Base class for NebulaGraph-based stores with process/thread-safe connection
    and session lifecycle management.

    This class handles:
      - Lazy initialization of the graph query connection pool and storage client.
      - Automatic session re-creation when processes or threads change.
      - Thread-local session storage for concurrent operations.
      - Utilities for executing NGQL with automatic space selection.
      - Optional collection and caching of numeric property columns.

    Args:
        pool_factory: Callable returning a new ConnectionPool instance.
        sclient_factory: Callable returning a new GraphStorageClient instance (optional).
        space: Name of the NebulaGraph space to operate on.
        username: NebulaGraph username (default: "root").
        password: NebulaGraph password (default: "nebula").
        missing_value: Default value to replace None for scalar normalization.
    """
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

        # Lazy creation:
        self._pool = None
        self._sclient = None
        self._sess_local = threading.local()
        self._pid = os.getpid()

        # Public configuration
        self._numeric_cols_by_tag: dict[str, list[str]] = {}
        self._x_cols: dict[str, list[str]] = {}
        self._reserved_cols = {"x", "y", "label", "category", "target"}
        self._missing_value = missing_value
        self._vid_codec = None

    # ---------- Lifecycle management ----------
    def _ensure_pool(self):
        """
        Ensure the connection pool and storage client are initialized in the current process.

        If the process ID has changed (e.g., after forking) or no pool exists, both
        the pool and storage client are re-created, and thread-local sessions reset.

        Returns:
            ConnectionPool: The active pool instance.
        """
        cur_pid = os.getpid()
        if (self._pool is None) or (self._pid != cur_pid):
           # Invalidate old process resources
            self._release_session_only()
            self._pool = self._pool_factory()
            self._sclient = self._sclient_factory() if self._sclient_factory else None
            self._sess_local = threading.local()  # Reset thread isolation
            self._pid = cur_pid
            self._vid_codec = None
        return self._pool

    def _ensure_session(self):
        """
        Ensure a thread-local session exists for the current thread.

        Creates a new session if one is not available, or reinitializes it after
        a process change.

        Returns:
            Session: Active NebulaGraph session.
        """
        self._ensure_pool()
        sess = getattr(self._sess_local, "sess", None)
        if sess is None:
            sess = self._pool.get_session(self.username, self.password)
            self._sess_local.sess = sess
        return sess

    def _ensure_session_fast(self):
        """
        Fast path for retrieving the current thread's session.

        Returns the existing session if:
          - It exists in thread-local storage.
          - The process ID hasn't changed.
          - The pool is still available.

        Otherwise, falls back to `_ensure_session()` for full checks.

        Returns:
            Session: Active NebulaGraph session.
        """
        sess = getattr(self._sess_local, "sess", None)
        if sess is not None and self._pid == os.getpid() and self._pool is not None:
            return sess
        return self._ensure_session()
    
    def _ensure_vid_codec(self) -> VidCodec:
        """Detect VID type once per (process, store), cache result."""
        if self._vid_codec is None:
            sess = self._ensure_session_fast()
            self._vid_codec = VidCodec.detect(sess, self.space)
        return self._vid_codec

    def _vid_literal(self, v) -> str:
        """Format VID literal for NGQL based on space VID type."""
        return self._ensure_vid_codec().literal(v)

    # ---------- Execution  ----------
    def _execute(self, ngql: str, check: bool = True):
        """
        Execute an NGQL statement in the current space, with automatic session handling.

        Automatically prepends `USE <space>;` to the query. If the session fails,
        it will be rebuilt once before retrying.

        Args:
            ngql: NGQL query string.
            check: Whether to raise an exception if execution fails.

        Returns:
            nebula3.ResultSet: The execution result.

        Raises:
            RuntimeError: If `check` is True and execution fails.
        """
        sess = self._ensure_session_fast()
        try:
            result = sess.execute(f'USE {self.space}; {ngql}')
        except Exception:
            # Session failed, rebuild and retry
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

    # ---------- Cleanup ----------
    def _release_session_only(self):
        """
        Release only the current thread's session.

        Leaves the connection pool intact.
        """
        sess = getattr(self._sess_local, "sess", None)
        if sess is not None:
            try:
                sess.release()
            except Exception:
                pass
            finally:
                self._sess_local.sess = None

    def _rebuild_session(self):
        """
        Rebuild the current thread's session while keeping the pool.
        """
        self._release_session_only()

    def close(self):
        """
        Manually close the store (e.g., at the end of training).

        Releases the current session and drops the pool and storage client references.
        """
        self._release_session_only()
        self._pool = None
        self._sclient = None

    # ---------- Process serialization ----------
    def __getstate__(self):
        """
        Remove live connections and sessions when pickling.

        This avoids carrying over active resources into child processes.
        """
        d = self.__dict__.copy()
        d["_pool"] = None
        d["_sclient"] = None
        d["_sess_local"] = None
        d["_vid_codec"] = None
        return d

    def __setstate__(self, d):
        """
        Restore state after unpickling, reinitializing thread-local storage.
        """
        self.__dict__.update(d)
        self._sess_local = threading.local()
        self._pid = os.getpid()
        self._vid_codec = None

    # ---------- Utility methods ----------
    # TODO: The methods in FeatureStore and GraphStore have not been replaced yet.
    #       Consider abstracting from it/don't abstract from it.
    def _normalize_scalar(self, v):
        """
        Normalize a scalar value.

        Replaces None with `self._missing_value`, converts bool to int, leaves others unchanged.
        """
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
        """
        Collect numeric columns for a given tag.

        Default implementation depends on `self._sclient._meta_cache`.
        Subclasses should override this method to customize numeric column collection.
        """
        raise NotImplementedError("Subclasses should implement _collect_numeric_cols(tag).")

    # Properties 
    @property
    def pool(self):
        self._ensure_pool()
        return self._pool

    @property
    def sclient(self):
        self._ensure_pool()
        return self._sclient
