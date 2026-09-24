"""T03 — Lớp lưu file thống nhất (MediaStorage).

- ``storage_key`` luôn tương đối (``reference/crops/<uuid>.png``...), không
  phụ thuộc đường dẫn máy, không chứa ``..`` hay đường dẫn tuyệt đối.
- Mặc định ``STORAGE_BACKEND=local`` ghi dưới ``MEDIA_ROOT`` (``outputs/media``).
  Supabase DB không đồng nghĩa file đã nằm trên Storage — backend tạo bucket
  private riêng qua cấu hình triển khai; ``STORAGE_BACKEND=supabase`` hiện báo
  lỗi rõ cho đến khi infra cung cấp bucket + key (frontend chỉ nhận URL có
  hạn hoặc qua endpoint có quyền, key dịch vụ không bao giờ ra frontend).
- Thứ tự ghi: file thành công trước, metadata DB sau; DB lỗi → caller dọn
  file mồ côi bằng ``delete_quiet`` (có khoảng chờ/retention do caller cấu hình).
- Upload stream có giới hạn dung lượng, không đọc toàn bộ video lớn rồi mới
  kiểm tra. Kiểm tra decode ảnh; ``cv2.imwrite`` phải trả True trước khi tạo asset.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import tempfile
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator, Optional

# Prefix chuẩn theo docs/face-media-database.md (T03).
PREFIX_REFERENCE_ORIGINALS = "reference/originals"
PREFIX_REFERENCE_CROPS = "reference/crops"
PREFIX_REFERENCE_GENERATED = "reference/generated"
PREFIX_SEARCH_ORIGINALS = "search/originals"
PREFIX_SEARCH_FRAMES = "search/frames"
PREFIX_SEARCH_CROPS = "search/crops"

_MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/x-matroska": ".mkv",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/webm": ".webm",
}
_EXT_TO_MIME = {ext: mime for mime, ext in _MIME_TO_EXT.items()}


def _default_root() -> Path:
    return Path(os.environ.get("MEDIA_ROOT", os.path.join("outputs", "media")))


def _default_max_bytes() -> int:
    try:
        return int(os.environ.get("MEDIA_MAX_BYTES", str(200 * 1024 * 1024)))
    except ValueError:
        return 200 * 1024 * 1024


def validate_storage_key(key: str) -> str:
    """Chặn path traversal và tên file do người dùng điều khiển."""
    if not key or not key.strip():
        raise ValueError("storage_key rỗng.")
    normalized = key.replace("\\", "/").strip()
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"storage_key không hợp lệ: {key!r}.")
    if not (1 <= len(pure.parts) <= 8):
        raise ValueError(f"storage_key có độ sâu bất thường: {key!r}.")
    return "/".join(pure.parts)


def build_key(prefix: str, asset_id: str, mime_type: str) -> str:
    ext = _MIME_TO_EXT.get(mime_type.lower())
    if ext is None:
        guessed = mimetypes.guess_extension(mime_type.split(";")[0].strip())
        ext = guessed or ".bin"
    validate_storage_key(f"{prefix}/x")
    try:
        uuid.UUID(str(asset_id))
    except ValueError as exc:
        raise ValueError(f"asset_id phải là UUID: {asset_id!r}.") from exc
    return f"{prefix}/{asset_id}{ext}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sniff_mime(data: bytes, fallback: str = "application/octet-stream") -> str:
    if data[:8].startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] in (b"RIFF",) and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:8] == b"ftyp":
        return "video/mp4"
    return fallback


class MediaStorage(ABC):
    @abstractmethod
    def put_bytes(self, data: bytes, key: str, *, mime_type: str) -> str:
        """Ghi bytes, không ghi đè key đã có. Trả về key đã chuẩn hóa."""

    @abstractmethod
    def put_stream(
        self, stream: BinaryIO, key: str, *, mime_type: str,
        max_bytes: Optional[int] = None,
    ) -> tuple[str, int, str]:
        """Ghi stream theo chunk (giới hạn dung lượng). Trả (key, size, sha256)."""

    @abstractmethod
    def open(self, key: str) -> BinaryIO:
        """Mở file để đọc binary. Caller đóng sau khi dùng."""

    @abstractmethod
    def get_access_url(self, key: str) -> str:
        """URL truy cập (local: prefix tĩnh; supabase tương lai: signed URL có hạn)."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Xóa file. Key chưa có → raise FileNotFoundError."""

    def delete_quiet(self, key: str) -> None:
        try:
            self.delete(key)
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def exists(self, key: str) -> bool:
        try:
            validate_storage_key(key)
        except ValueError:
            return False
        return self._exists(key)

    @abstractmethod
    def _exists(self, key: str) -> bool:
        ...


class LocalMediaStorage(MediaStorage):
    """Backend local disk cho môi trường chạy hiện tại (T03)."""

    def __init__(
        self,
        root: Optional[Path] = None,
        url_prefix: str = "/outputs/media",
        max_bytes: Optional[int] = None,
    ) -> None:
        self.root = Path(root) if root is not None else _default_root()
        self.url_prefix = url_prefix.rstrip("/") or "/outputs/media"
        self.max_bytes = max_bytes if max_bytes is not None else _default_max_bytes()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        clean = validate_storage_key(key)
        path = (self.root / PurePosixPath(clean)).resolve()
        root_resolved = self.root.resolve()
        if path != root_resolved and root_resolved not in path.parents:
            raise ValueError(f"storage_key vượt khỏi root: {key!r}.")
        return path

    def _exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def put_bytes(self, data: bytes, key: str, *, mime_type: str) -> str:
        if len(data) > self.max_bytes:
            raise ValueError(f"File vượt giới hạn {self.max_bytes} bytes.")
        if not mime_type or "/" not in mime_type:
            raise ValueError(f"mime_type không hợp lệ: {mime_type!r}.")
        path = self._resolve(key)
        if path.exists():
            raise FileExistsError(f"storage_key đã tồn tại: {key!r} (không ghi đè).")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".upload-", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.remove(tmp_name)
            except OSError:
                pass
            raise
        return validate_storage_key(key)

    def put_stream(
        self, stream: BinaryIO, key: str, *, mime_type: str,
        max_bytes: Optional[int] = None,
    ) -> tuple[str, int, str]:
        limit = max_bytes if max_bytes is not None else self.max_bytes
        if not mime_type or "/" not in mime_type:
            raise ValueError(f"mime_type không hợp lệ: {mime_type!r}.")
        path = self._resolve(key)
        if path.exists():
            raise FileExistsError(f"storage_key đã tồn tại: {key!r} (không ghi đè).")
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        fd, tmp_name = tempfile.mkstemp(prefix=".upload-", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > limit:
                        raise ValueError(
                            f"File vượt giới hạn {limit} bytes (đã dừng giữa stream)."
                        )
                    digest.update(chunk)
                    handle.write(chunk)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.remove(tmp_name)
            except OSError:
                pass
            raise
        return validate_storage_key(key), size, digest.hexdigest()

    def open(self, key: str) -> BinaryIO:
        path = self._resolve(key)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file: {key!r}.")
        return open(path, "rb")

    def get_access_url(self, key: str) -> str:
        clean = validate_storage_key(key)
        return f"{self.url_prefix}/{clean}"

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file: {key!r}.")
        path.unlink()


class SupabaseMediaStorage(MediaStorage):
    """Giữ chỗ cho bucket private Supabase (tạo bằng infra riêng, T03).

    Key dịch vụ chỉ ở backend; frontend nhận URL có hạn. Chưa cấu hình xong
    thì mọi thao tác báo lỗi rõ thay vì ghi nhầm sang local.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "STORAGE_BACKEND=supabase chưa được cấu hình: tạo bucket private bằng "
            "cấu hình triển khai riêng rồi triển khai SupabaseMediaStorage "
            "(SUPABASE_URL/SUPABASE_SERVICE_KEY/SUPABASE_BUCKET). "
            "Hiện dùng STORAGE_BACKEND=local."
        )

    def put_bytes(self, data: bytes, key: str, *, mime_type: str) -> str:
        raise NotImplementedError

    def put_stream(self, stream, key, *, mime_type, max_bytes=None):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def open(self, key: str) -> BinaryIO:
        raise NotImplementedError

    def get_access_url(self, key: str) -> str:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def _exists(self, key: str) -> bool:
        raise NotImplementedError


_STORAGE: Optional[MediaStorage] = None


def get_storage() -> MediaStorage:
    """Factory theo STORAGE_BACKEND (mặc định local)."""
    global _STORAGE
    if _STORAGE is not None:
        return _STORAGE
    backend = os.environ.get("STORAGE_BACKEND", "local").strip().lower()
    if backend == "local":
        _STORAGE = LocalMediaStorage()
    elif backend == "supabase":
        _STORAGE = SupabaseMediaStorage()
    else:
        raise ValueError(
            f"STORAGE_BACKEND={backend!r} không hỗ trợ (chọn 'local' hoặc 'supabase')."
        )
    return _STORAGE


def reset_storage_cache() -> None:
    global _STORAGE
    _STORAGE = None


def check_image_decodable(data: bytes) -> tuple[int, int]:
    """Kiểm tra decode ảnh bằng OpenCV; trả (width, height)."""
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không giải mã được ảnh (sai định dạng/dữ liệu hỏng).")
    height, width = image.shape[:2]
    return width, height


def write_image_strict(path: str | Path, image) -> Path:  # type: ignore[no-untyped-def]
    """Ghi ảnh, raise nếu cv2.imwrite thất bại (T03: kiểm tra trước tạo asset)."""
    import cv2

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(target), image)
    if not ok or not target.is_file():
        raise IOError(f"Ghi ảnh thất bại: {target}.")
    return target


@contextmanager
def staged_local_file(storage: MediaStorage, key: str) -> Iterator[None]:
    """Dọn file mồ côi khi khối with lỗi sau lúc file đã ghi (DB rollback...)."""
    try:
        yield
    except BaseException:
        storage.delete_quiet(key)
        raise
