"""T02 — Lỗi nghiệp vụ dùng chung cho repositories (không lộ secret/DSN)."""


class ConflictError(ValueError):
    """Trùng idempotency key / unique constraint (retry an toàn)."""


class NotFoundError(LookupError):
    """Không tìm thấy bản ghi tham chiếu."""


class PurposeError(ValueError):
    """Sai luồng reference/search (ví dụ crop search làm input generation)."""
