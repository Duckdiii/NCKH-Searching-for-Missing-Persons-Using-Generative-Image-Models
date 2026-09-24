"""T01 — Kết nối DB Supabase (psycopg3) + pool cho worker FastAPI.

Quy ước:
- Đọc DATABASE_URL phía backend (có thể nạp từ .env local qua python-dotenv).
- Mọi lỗi cấu hình/kết nối đều KHÔNG in DSN/mật khẩu.
- Chuẩn hóa URL: bỏ tham số chỉ dành cho pgbouncer/JDBC (pgbouncer,
  prepare_threshold, ...), giữ tham số libpq hợp lệ, ép SSL.
- Pool đơn giản, thread-safe: mỗi request / thread job_runner mượn riêng
  một connection qua ``connection()`` rồi trả lại; không chia sẻ connection
  đang dùng dở giữa các thread.
"""

from __future__ import annotations

import os
import queue
import threading
from contextlib import contextmanager
from typing import Iterator, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

try:  # python-dotenv là dependency khai báo trong requirements.txt
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # thiếu package / thiếu file .env đều không được crash import
    pass

# Tham số libpq được giữ lại khi chuẩn hóa URL. Mọi tham số khác
# (đặc biệt các cờ pgbouncer/JDBC) bị loại để tránh lỗi
# "invalid connection option" từ libpq.
_ALLOWED_QUERY_PARAMS = frozenset(
    {
        "sslmode",
        "sslrootcert",
        "sslcert",
        "sslkey",
        "connect_timeout",
        "application_name",
        "options",
        "target_session_attrs",
        "host",
        "port",
        "dbname",
        "user",
    }
)

# Tham số đã biết là không thuộc libpq — phải bỏ, không được forward.
_DROPPED_QUERY_PARAMS = frozenset(
    {
        "pgbouncer",
        "preparethreshold",
        "prepare_threshold",
        "poolmode",
        "pool_mode",
    }
)


class DatabaseConfigError(RuntimeError):
    """Lỗi cấu hình DB — message không bao giờ chứa DSN/secret."""


def _redacted_host(url: str) -> str:
    """Trích host để log mà không lộ user/password/query."""
    try:
        parts = urlsplit(url)
        return parts.hostname or "unknown-host"
    except Exception:
        return "unknown-host"


def get_database_url(*, env_var: str = "DATABASE_URL") -> str:
    """Đọc URL kết nối từ môi trường.

    Raises:
        DatabaseConfigError: khi biến môi trường thiếu/rỗng. Message chỉ nêu
            tên biến, không in giá trị.
    """
    url = os.environ.get(env_var, "").strip().strip("'\"")
    if not url:
        raise DatabaseConfigError(
            f"Thiếu cấu hình database: biến môi trường {env_var} chưa được đặt. "
            "Copy .env.example thành .env và điền giá trị (không commit .env)."
        )
    scheme = url.split("://", 1)[0].lower()
    if scheme not in ("postgres", "postgresql"):
        raise DatabaseConfigError(
            f"Biến {env_var} phải bắt đầu bằng postgres:// hoặc postgresql:// "
            f"(host={_redacted_host(url)})."
        )
    return url


def get_test_database_url() -> str:
    """URL cho test — tách khỏi DB chứa dữ liệu thật (T01).

    Từ chối khi DATABASE_URL_TEST trùng DATABASE_URL production để fixture
    dọn DB không bao giờ chạy nhầm lên dữ liệu thật.
    """
    test_url = os.environ.get("DATABASE_URL_TEST", "").strip().strip("'\"")
    if not test_url:
        raise DatabaseConfigError(
            "Thiếu cấu hình database test: biến DATABASE_URL_TEST chưa được đặt."
        )
    prod_url = os.environ.get("DATABASE_URL", "").strip()
    if prod_url and test_url == prod_url:
        raise DatabaseConfigError(
            "DATABASE_URL_TEST trùng DATABASE_URL production — từ chối để tránh "
            "xóa nhầm dữ liệu thật. Hãy tạo database test riêng."
        )
    return test_url


def normalize_database_url(url: str) -> str:
    """Chuẩn hóa URL cho kết nối pooler Supabase hiện có.

    - Bỏ cờ pgbouncer/JDBC mà libpq không hiểu (``pgbouncer``,
      ``prepare_threshold``/``preparethreshold``, ``pool_mode``...).
    - Giữ nguyên các tham số libpq hợp lệ khác.
    - Ép ``sslmode`` (mặc định ``require``) vì Supabase bắt buộc SSL.
    """
    parts = urlsplit(url)
    kept: list[tuple[str, str]] = []
    for key, value in parse_qsl(parts.query, keep_blank_values=False):
        lowered = key.lower()
        if lowered in _DROPPED_QUERY_PARAMS:
            continue
        if lowered not in _ALLOWED_QUERY_PARAMS:
            continue
        kept.append((key, value))
    query_dict = dict(kept)
    lowered_keys = {k.lower() for k, _ in kept}
    if "sslmode" not in lowered_keys:
        query_dict["sslmode"] = "require"
    if "connect_timeout" not in lowered_keys:
        # Tránh treo lâu khi DB không reachable: fail nhanh để endpoint
        # fallback về hành vi legacy (RAM + filesystem).
        query_dict["connect_timeout"] = "10"
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query_dict), parts.fragment)
    )


class DatabasePool:
    """Pool connection tối giản, thread-safe trên psycopg3.

    Dùng hàng đợi để cấp phát; mỗi ``connection()`` mượn đúng 1 conn
    độc lập — caller phải dùng xong trong khối with để trả về pool.
    """

    def __init__(self, dsn: str, max_size: int = 10) -> None:
        if max_size < 1:
            raise ValueError("max_size phải >= 1")
        self._dsn = dsn
        self._max_size = max_size
        self._available: queue.Queue = queue.Queue()
        self._created = 0
        # RLock vì getconn() giữ guard trong lúc gọi _new_connection_and_count()
        # (cũng acquire guard) — Lock thường sẽ deadlock chính thread mình.
        self._guard = threading.RLock()
        self._closed = False

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def created_count(self) -> int:
        with self._guard:
            return self._created

    def _new_connection(self):  # type: ignore[no-untyped-def]
        import psycopg

        return psycopg.connect(self._dsn, autocommit=False)

    def getconn(self, timeout: float = 30.0):
        with self._guard:
            if self._closed:
                raise DatabaseConfigError("Connection pool đã đóng.")
            if self._created < self._max_size:
                try:
                    return self._new_connection_and_count()
                except Exception:
                    raise
        try:
            return self._available.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(
                "Hết connection trong pool (tăng DATABASE_POOL_SIZE hoặc "
                "kiểm tra connection leak)."
            ) from exc

    def _new_connection_and_count(self):  # type: ignore[no-untyped-def]
        conn = self._new_connection()
        with self._guard:
            self._created += 1
        return conn

    def putconn(self, conn) -> None:  # type: ignore[no-untyped-def]
        if conn is None:
            return
        with self._guard:
            closed = self._closed
        if closed:
            try:
                conn.close()
            except Exception:
                pass
            return
        try:
            if conn.closed:
                with self._guard:
                    self._created = max(0, self._created - 1)
                return
            conn.rollback()
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            with self._guard:
                self._created = max(0, self._created - 1)
            return
        self._available.put(conn)

    def closeall(self) -> None:
        with self._guard:
            self._closed = True
        while True:
            try:
                conn = self._available.get_nowait()
            except queue.Empty:
                break
            try:
                conn.close()
            except Exception:
                pass
        with self._guard:
            self._created = 0

    @contextmanager
    def connection(self) -> Iterator:
        conn = self.getconn()
        try:
            yield conn
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        else:
            try:
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
        finally:
            self.putconn(conn)


_POOL: Optional[DatabasePool] = None
_POOL_LOCK = threading.Lock()


def init_pool(*, dsn: Optional[str] = None, max_size: Optional[int] = None) -> DatabasePool:
    """Khởi tạo pool toàn cục. Gọi 1 lần khi app startup."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            return _POOL
        raw = dsn or get_database_url()
        size = max_size or int(os.environ.get("DATABASE_POOL_SIZE", "10"))
        _POOL = DatabasePool(normalize_database_url(raw), max_size=max(1, size))
        return _POOL


def get_pool() -> DatabasePool:
    """Lấy pool toàn cục (tự init lười nếu chưa có)."""
    pool = _POOL
    if pool is not None:
        return pool
    return init_pool()


def close_pool() -> None:
    """Đóng pool toàn cục. Gọi khi app shutdown để không rò connection."""
    global _POOL
    with _POOL_LOCK:
        pool, _POOL = _POOL, None
    if pool is not None:
        pool.closeall()
