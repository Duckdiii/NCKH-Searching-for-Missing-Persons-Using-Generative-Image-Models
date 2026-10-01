"""T01 — Migration runner cho schema face_media (Supabase/Postgres).

Dùng psycopg3 trực tiếp, không ORM:

- Bảng lịch sử: face_media.schema_migrations(version, filename, checksum, applied_at).
- Mỗi migration chạy đúng 1 lần trong 1 transaction; checksum sha256 phát hiện drift.
- Môi trường Supabase đã có schema từ 001: dùng ``baseline`` để xác minh cấu
  trúc (9 bảng core, 2 view, 12 FK) rồi ghi nhận 001 mà KHÔNG thực thi lại
  CREATE SCHEMA. Không bao giờ chạy lại 001 trên DB hiện tại.
- Migration mới (002, ...) chạy bằng ``migrate``.

Cách chạy (không in secret ra log)::

    copy .env.example .env   # rồi điền DATABASE_URL (không commit .env)
    python database/migrate.py status     # xem trạng thái
    python database/migrate.py baseline   # 1 lần duy nhất trên DB đã có 001
    python database/migrate.py migrate    # áp dụng migration mới còn thiếu
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))
from backend.api.database import (  # noqa: E402
    DatabaseConfigError,
    get_database_url,
    normalize_database_url,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = Path(__file__).resolve().parent

# 9 bảng nghiệp vụ core của 001 (chưa tính schema_migrations do runner tạo thêm).
CORE_TABLES = [
    "assets",
    "cameras",
    "sources",
    "frames",
    "face_detections",
    "face_crops",
    "generation_jobs",
    "generated_images",
    "face_embeddings",
]
CORE_VIEWS = ["generated_image_lineage", "search_face_gallery"]
EXPECTED_FK_COUNT = 12

VERSION_RE = re.compile(r"^(\d{3})_.+\.sql$")


def discover_migrations() -> list[tuple[str, Path]]:
    found: dict[str, Path] = {}
    for search_dir in (DATABASE_DIR, DATABASE_DIR / "migrations"):
        if not search_dir.is_dir():
            continue
        for path in sorted(search_dir.glob("*.sql")):
            match = VERSION_RE.match(path.name)
            if match:
                found.setdefault(match.group(1), path)
    return sorted(found.items())


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _connect():  # type: ignore[no-untyped-def]
    import psycopg

    try:
        dsn = normalize_database_url(get_database_url())
    except DatabaseConfigError as exc:
        print(f"Lỗi cấu hình: {exc}")
        raise SystemExit(2) from None
    try:
        return psycopg.connect(dsn)
    except Exception as exc:
        print("Không kết nối được database (chi tiết đã rút gọn, không in DSN).")
        print(f"  loại lỗi: {type(exc).__name__}")
        raise SystemExit(1) from None


def ensure_history_table(conn) -> None:  # type: ignore[no-untyped-def]
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS face_media")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS face_media.schema_migrations (
                version text PRIMARY KEY,
                filename text NOT NULL,
                checksum text NOT NULL,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
    conn.commit()


def list_applied(conn) -> dict[str, tuple[str, str]]:  # type: ignore[no-untyped-def]
    with conn.cursor() as cur:
        cur.execute(
            "SELECT version, filename, checksum FROM face_media.schema_migrations"
        )
        return {row[0]: (row[1], row[2]) for row in cur.fetchall()}


def inspect_existing(conn) -> tuple[set[str], set[str], int]:  # type: ignore[no-untyped-def]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'face_media' AND table_type = 'BASE TABLE'
            """
        )
        tables = {row[0] for row in cur.fetchall()}
        cur.execute(
            """
            SELECT table_name FROM information_schema.views
            WHERE table_schema = 'face_media'
            """
        )
        views = {row[0] for row in cur.fetchall()}
        cur.execute(
            """
            SELECT count(*) FROM information_schema.table_constraints
            WHERE table_schema = 'face_media' AND constraint_type = 'FOREIGN KEY'
            """
        )
        fk_count = int(cur.fetchone()[0])
    return tables, views, fk_count


def cmd_status() -> int:
    migrations = discover_migrations()
    with _connect() as conn:
        ensure_history_table(conn)
        applied = list_applied(conn)
        for version, path in migrations:
            checksum = file_checksum(path)
            if version not in applied:
                print(f"  [pending] {version} {path.name}")
            elif applied[version][1] != checksum:
                print(
                    f"  [DRIFT] {version} {path.name}: checksum file khác DB "
                    f"(db={applied[version][1][:12]} file={checksum[:12]})"
                )
            else:
                print(f"  [applied] {version} {path.name}")
    return 0


def cmd_baseline() -> int:
    """Ghi nhận 001 đã áp dụng sau khi xác minh cấu trúc, không chạy lại DDL."""
    migrations = dict(discover_migrations())
    if "001" not in migrations:
        print("Không tìm thấy database/001_face_media.sql.")
        return 1
    with _connect() as conn:
        ensure_history_table(conn)
        applied = list_applied(conn)
        if "001" in applied:
            print("001 đã được baseline/applied trước đó — không làm gì thêm.")
            return 0
        tables, views, fk_count = inspect_existing(conn)
        missing_tables = [t for t in CORE_TABLES if t not in tables]
        missing_views = [v for v in CORE_VIEWS if v not in views]
        if missing_tables or missing_views or fk_count != EXPECTED_FK_COUNT:
            print("Baseline 001 THẤT BẠI: cấu trúc Supabase hiện tại không khớp DDL.")
            if missing_tables:
                print(f"  thiếu bảng: {', '.join(missing_tables)}")
            if missing_views:
                print(f"  thiếu view: {', '.join(missing_views)}")
            print(f"  FK: có {fk_count}, kỳ vọng {EXPECTED_FK_COUNT}.")
            print("  Không ghi baseline. Kiểm tra đúng database rồi thử lại.")
            return 1
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO face_media.schema_migrations "
                "(version, filename, checksum) VALUES ('001', %s, %s)",
                (migrations["001"].name, file_checksum(migrations["001"])),
            )
        conn.commit()
        print("Baseline 001 OK: đã ghi nhận schema hiện có (không chạy lại CREATE).")
    return 0


def cmd_migrate() -> int:
    migrations = discover_migrations()
    if not migrations:
        print("Không tìm thấy migration nào.")
        return 1
    with _connect() as conn:
        ensure_history_table(conn)
        applied = list_applied(conn)
        tables, views, fk_count = inspect_existing(conn)
        core_present = all(t in tables for t in CORE_TABLES)
        for version, path in migrations:
            checksum = file_checksum(path)
            if version in applied:
                if applied[version][1] != checksum:
                    print(
                        f"DRIFT {version} {path.name}: file đã đổi sau khi apply "
                        f"(db={applied[version][1][:12]} file={checksum[:12]}). "
                        "Tạo migration mới, không sửa file đã apply."
                    )
                    return 1
                continue
            # Bảo vệ DB đã có schema: không thực thi lại 001.
            if version == "001" and core_present:
                print(
                    "Từ chối thực thi 001 vì schema face_media đã tồn tại. "
                    "Chạy 'python database/migrate.py baseline' trước."
                )
                return 1
            sql = path.read_text(encoding="utf-8")
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO face_media.schema_migrations "
                    "(version, filename, checksum) VALUES (%s, %s, %s)",
                    (version, path.name, checksum),
                )
            conn.commit()
            print(f"Applied {version} {path.name}.")
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "status":
        return cmd_status()
    if cmd == "baseline":
        return cmd_baseline()
    if cmd == "migrate":
        return cmd_migrate()
    print(f"Lệnh không hỗ trợ: {cmd}. Dùng: status | baseline | migrate")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
