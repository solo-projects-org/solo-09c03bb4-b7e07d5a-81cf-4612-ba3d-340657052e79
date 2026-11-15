"""Waveform archive intake: frame assembly, XOR parity recovery, verified publishing.

Design invariants:

* A data frame is exactly 256 bytes, transmitted as 512 hex characters.
* The last frame is zero-padded; the true archive is the first ``size_bytes`` bytes
  of the concatenated frames (at most ``frame_count * 256``).
* One optional 256-byte parity frame holds the byte-wise XOR of all data frames.
  It can reconstruct **at most one** missing frame.
* A reconstructed frame is *tentative*: it is used in memory for assembly but is
  never stored as a received frame, so the genuine re-transmission is always
  accepted instead of being rejected as a conflicting payload.
* Two or more missing frames: wait for re-transmission; nothing is published.
* Publication requires a complete archive whose truncated bytes match the declared
  SHA-256. On mismatch all received fragments are retained for a later repair.
* Publication is atomic (temp file -> fsync -> rename -> fsync directory -> DB row).
  A crash therefore leaves either no download or the complete published archive.
* A transfer is published at most once; late/duplicate fragments never alter it.
* Same transfer + same frame number with different content is rejected.
* Archival tiering: a published archive may be copied to a separate cold volume.
  The DB row switches to ``cold`` only after the cold copy has been fsynced,
  atomically renamed into place and re-verified against the registered size and
  SHA-256. The hot copy is deleted only after that commit. A crash at any step
  leaves the row pointing at a complete, verified copy; unreferenced cold files
  and ``.part`` copies are removed on startup, never mistaken for migrated data.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:  # POSIX only; Docker image is Linux.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

FRAME_BYTES = 256
FRAME_HEX_LEN = FRAME_BYTES * 2
MIN_FRAMES = 2
MAX_FRAMES = 16
HEX_RE = re.compile(r"^[0-9a-fA-F]*$")
TID_RE = re.compile(r"[A-Za-z0-9._-]+")

STATUS_WAITING = "waiting"
STATUS_MISMATCH = "mismatch"
STATUS_PUBLISHED = "published"

# Storage tiers for a published archive. ``hot`` is the original intake volume,
# ``cold`` is the independent archive volume configured as ``cold_root``.
TIER_HOT = "hot"
TIER_COLD = "cold"


class StorageError(Exception):
    """Base class for user-facing intake errors."""

    http_status = 400

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status or self.http_status


class ConflictError(StorageError):
    http_status = 409


class NotFoundError(StorageError):
    http_status = 404


class ColdStorageError(StorageError):
    """The cold archive volume is unavailable or rejected the write."""

    http_status = 503


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_transfer_id(raw: str) -> str:
    """Transfer IDs are opaque but must be safe as a directory name."""
    value = (raw or "").strip()
    if not value:
        raise StorageError("传输号不能为空")
    if len(value) > 64:
        raise StorageError("传输号过长（最多 64 个字符）")
    if not TID_RE.fullmatch(value) or value in {".", ".."}:
        raise StorageError("传输号只能包含字母、数字、点、下划线和连字符")
    return value


def parse_hex_payload(raw: str, *, frame_no: int | None = None) -> bytes:
    data = (raw or "").strip()
    # Tolerate whitespace and an optional 0x prefix from CSV exports.
    if data[:2] in {"0x", "0X"}:
        data = data[2:]
    data = re.sub(r"\s+", "", data)
    where = f"第 {frame_no} 帧" if frame_no is not None else "校验帧"
    if len(data) != FRAME_HEX_LEN:
        raise StorageError(
            f"{where}载荷必须为 {FRAME_HEX_LEN} 个十六进制字符（256 字节），实际 {len(data)}"
        )
    if not HEX_RE.fullmatch(data):
        raise StorageError(f"{where}载荷包含非十六进制字符")
    try:
        return bytes.fromhex(data)
    except ValueError as exc:  # pragma: no cover - guarded by the regex
        raise StorageError(f"{where}载荷无法解析") from exc


def parse_size_bytes(raw: int | str) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise StorageError("原档字节数必须为整数") from exc
    if value < 1:
        raise StorageError("原档字节数必须大于 0")
    return value


def parse_frame_count(raw: int | str) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise StorageError("数据帧数必须为整数") from exc
    if not MIN_FRAMES <= value <= MAX_FRAMES:
        raise StorageError(f"数据帧数必须在 {MIN_FRAMES}～{MAX_FRAMES} 之间")
    return value


def parse_sha256(raw: str) -> str:
    value = (raw or "").strip().lower()
    if value.startswith("sha256:"):
        value = value[7:]
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise StorageError("SHA-256 必须为 64 位十六进制字符串")
    return value


def validate_declared_size(count: int, size_bytes: int) -> None:
    """A padded archive of ``count`` full frames truncates to 1..count*256 bytes.

    With at least two frames the archive must use more than ``count-1`` full
    frames, otherwise the declared frame count is inconsistent.
    """
    minimum = (count - 1) * FRAME_BYTES + 1
    maximum = count * FRAME_BYTES
    if not minimum <= size_bytes <= maximum:
        raise StorageError(
            f"原档字节数需与 {count} 个数据帧相容（{minimum}～{maximum} 字节，末帧补零）"
        )


def xor_bytes(parts: Iterable[bytes]) -> bytes:
    acc = bytearray(FRAME_BYTES)
    for part in parts:
        if len(part) != FRAME_BYTES:
            raise ValueError("XOR operand must be 256 bytes")
        for i, value in enumerate(part):
            acc[i] ^= value
    return bytes(acc)


@dataclass(frozen=True)
class Delivery:
    transfer_id: str
    frame_count: int
    size_bytes: int
    sha256: str
    created_at: str


@dataclass(frozen=True)
class ArchiveState:
    delivery: Delivery
    frames: dict[int, bytes]
    parity: bytes | None
    status: str
    published_sha256: str | None
    published_size: int | None
    published_at: str | None
    missing_frames: tuple[int, ...]
    parity_recoverable: bool
    actual_sha256: str | None
    recovered_frames: tuple[int, ...]
    tentative_frame: int | None
    """Frame currently missing but reconstructed in memory by the parity frame."""
    storage_tier: str = TIER_HOT
    """Where the published bytes currently live (``hot`` / ``cold``)."""
    cold_sha256: str | None = None
    cold_size: int | None = None
    cold_at: str | None = None
    cold_error: str | None = None
    """Last archival failure, cleared on a successful migration."""
    cold_path: str | None = None

    @property
    def received_frames(self) -> tuple[int, ...]:
        return tuple(sorted(self.frames))

    @property
    def is_complete(self) -> bool:
        return not self.missing_frames

    @property
    def can_attempt(self) -> bool:
        """Enough material exists to try assembling the archive."""
        return not self.missing_frames or (len(self.missing_frames) == 1 and self.parity is not None)

    @property
    def tentative_frame_no(self) -> int | None:
        """Missing frame that parity could rebuild, if verification has not passed."""
        if self.status != STATUS_PUBLISHED and len(self.missing_frames) == 1 and self.parity is not None:
            return self.missing_frames[0]
        return None


class Storage:
    """SQLite metadata + filesystem fragment store rooted at ``root``.

    ``cold_root`` (optional) points at a separate archive volume. Published
    archives can be migrated there; the database records which tier holds the
    authoritative bytes.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        cold_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.tmp_dir = self.root / "tmp"
        self.published_dir = self.root / "published"
        for directory in (self.root, self.tmp_dir, self.published_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.cold_root = Path(cold_root) if cold_root is not None else None
        if self.cold_root is not None:
            self._validate_cold_root()
            self.cold_tmp_dir = self.cold_root / "tmp"
            self.cold_published_dir = self.cold_root / "published"
            for directory in (self.cold_root, self.cold_tmp_dir, self.cold_published_dir):
                directory.mkdir(parents=True, exist_ok=True)
        else:
            self.cold_tmp_dir = None
            self.cold_published_dir = None
        self.db_path = self.root / "archive.sqlite3"
        self._lock_path = self.root / ".intake.lock"
        self._write_lock = threading.RLock()
        self._local = threading.local()
        self._init_db()
        self.sanitize_startup()

    def _validate_cold_root(self) -> None:
        """The cold tier must be a genuinely separate tree.

        Sharing or nesting the two roots would let hot-copy cleanup delete the
        only surviving archive (or vice versa), so refuse the configuration.
        """
        assert self.cold_root is not None
        hot = self.root.resolve()
        cold = self.cold_root.resolve()
        if hot == cold or hot in cold.parents or cold in hot.parents:
            raise ValueError("冷存储目录不能与快盘数据目录相同或互相嵌套")

    @property
    def cold_enabled(self) -> bool:
        return self.cold_root is not None

    # ------------------------------------------------------------------ locking
    @contextmanager
    def _critical(self) -> Iterator[None]:
        """Serialize mutation + publication within a process and across processes."""
        with self._write_lock:
            depth = getattr(self._local, "exclusive", 0)
            if fcntl is None or depth:
                # Re-entrant in this thread: the outermost holder already owns the
                # exclusive lock, and a second flock on a new fd would deadlock.
                self._local.exclusive = depth + 1
                try:
                    yield
                finally:
                    self._local.exclusive = depth
                return
            fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                self._local.exclusive = 1
                yield
            finally:
                self._local.exclusive = 0
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    @contextmanager
    def _shared(self) -> Iterator[None]:
        """Shared (reader) lock so a download cannot race a migration/cleanup.

        Writers take ``LOCK_EX`` through :meth:`_critical`; readers take
        ``LOCK_SH``. The lock lives on the hot volume, which every instance of
        the service shares. Without ``fcntl`` the process falls back to the
        in-process write lock: safe, merely less concurrent.
        """
        if getattr(self._local, "exclusive", 0):
            # This thread already holds the exclusive lock; sharing it is trivially
            # safe and re-locking on a new fd would block against ourselves.
            yield
            return
        if fcntl is None:
            with self._write_lock:
                yield
            return
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    # ------------------------------------------------------------------ schema
    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        # busy_timeout first: journal-mode switching also needs the write lock.
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._db() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS deliveries (
                    transfer_id  TEXT PRIMARY KEY,
                    frame_count  INTEGER NOT NULL,
                    size_bytes   INTEGER NOT NULL,
                    sha256       TEXT NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'waiting',
                    pub_sha256   TEXT,
                    pub_size     INTEGER,
                    pub_path     TEXT,
                    pub_at       TEXT,
                    actual_sha256 TEXT,
                    pub_recovered TEXT,
                    storage_tier TEXT NOT NULL DEFAULT 'hot',
                    cold_sha256  TEXT,
                    cold_size    INTEGER,
                    cold_path    TEXT,
                    cold_at      TEXT,
                    cold_error   TEXT,
                    created_at   TEXT NOT NULL,
                    updated_at   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS frames (
                    transfer_id TEXT NOT NULL,
                    frame_no    INTEGER NOT NULL,
                    payload     BLOB NOT NULL,
                    sha256      TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    PRIMARY KEY (transfer_id, frame_no),
                    FOREIGN KEY (transfer_id) REFERENCES deliveries(transfer_id)
                );

                CREATE TABLE IF NOT EXISTS parities (
                    transfer_id TEXT PRIMARY KEY,
                    payload     BLOB NOT NULL,
                    sha256      TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    FOREIGN KEY (transfer_id) REFERENCES deliveries(transfer_id)
                );

                CREATE TABLE IF NOT EXISTS audit (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    transfer_id TEXT NOT NULL,
                    event       TEXT NOT NULL,
                    detail      TEXT NOT NULL,
                    at          TEXT NOT NULL
                );
                """
            )
        with self._db() as conn:
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(deliveries)")}
        additions = {
            "actual_sha256": "TEXT",
            "pub_recovered": "TEXT",
            "storage_tier": "TEXT NOT NULL DEFAULT 'hot'",
            "cold_sha256": "TEXT",
            "cold_size": "INTEGER",
            "cold_path": "TEXT",
            "cold_at": "TEXT",
            "cold_error": "TEXT",
        }
        for name, ddl in additions.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE deliveries ADD COLUMN {name} {ddl}")

    def sanitize_startup(self) -> None:
        """Clean up artifacts of an interrupted write, then resume safe work.

        Partial files are never referenced by a published row, so deleting them
        cannot damage a valid archive. A published row whose file vanished is left
        alone: downloads fail closed. Waiting deliveries that already have enough
        material are retried, which covers a crash between rename and commit.

        Cold-tier reconciliation runs under the same exclusive lock as migration:
        partially copied files are deleted, cold rows whose copy no longer verifies
        fall back to a verified hot copy, and hot copies of cold rows are removed.
        Unreferenced cold files are never adopted as migrated archives.
        """
        partials = list(self.tmp_dir.glob("*.part"))
        if self.published_dir.is_dir():
            partials += list(self.published_dir.rglob("*.part"))
        if self.cold_tmp_dir is not None:
            partials += list(self.cold_tmp_dir.glob("*.part"))
        if self.cold_published_dir is not None:
            partials += list(self.cold_published_dir.rglob("*.part"))
        with self._critical():
            for path in partials:
                try:
                    path.unlink()
                except OSError:
                    pass
            if self.cold_enabled:
                self._reconcile_cold_locked()
        with self._db() as conn:
            rows = conn.execute(
                "SELECT transfer_id FROM deliveries WHERE status=?", (STATUS_WAITING,)
            ).fetchall()
        for row in rows:
            tid = row["transfer_id"]
            try:
                with self._critical():
                    if self.state(tid).can_attempt:
                        self._publish_locked(tid)
            except Exception:
                # Startup must not fail because one delivery cannot be assembled.
                continue

    def _reconcile_cold_locked(self) -> None:
        """Validate cold rows and delete unreferenced cold/hot copies. Holds lock."""
        assert self.cold_root is not None and self.cold_published_dir is not None
        with self._db() as conn:
            rows = conn.execute(
                "SELECT * FROM deliveries WHERE status=?", (STATUS_PUBLISHED,)
            ).fetchall()
        referenced: set[str] = set()
        for row in rows:
            tid = row["transfer_id"]
            size = int(row["pub_size"]) if row["pub_size"] is not None else None
            digest = row["pub_sha256"]
            tier = row["storage_tier"] or TIER_HOT
            hot_path = self.root / row["pub_path"] if row["pub_path"] else None
            cold_path = (
                self.cold_root / row["cold_path"]
                if self.cold_enabled and row["cold_path"]
                else None
            )
            if tier == TIER_COLD and cold_path is not None:
                if _file_matches(cold_path, size, digest):
                    referenced.add(os.path.realpath(cold_path))
                    # The hot copy is no longer needed: finish an interrupted cleanup.
                    if hot_path is not None:
                        self._delete_quietly(hot_path)
                    continue
                # The recorded cold copy is missing or corrupt. Do not keep claiming
                # it is migrated: fall back to a hot copy that still verifies.
                if hot_path is not None and _file_matches(hot_path, size, digest):
                    message = "冷盘副本校验失败，已回退快盘副本"
                    now = utcnow()
                    with self._db() as conn:
                        conn.execute("BEGIN IMMEDIATE")
                        try:
                            conn.execute(
                                """
                                UPDATE deliveries
                                   SET storage_tier=?, cold_sha256=NULL, cold_size=NULL,
                                       cold_path=NULL, cold_at=NULL, cold_error=?, updated_at=?
                                 WHERE transfer_id=?
                                """,
                                (TIER_HOT, message, now, tid),
                            )
                            self._audit(conn, tid, "cold-reverted", message)
                            conn.execute("COMMIT")
                        except Exception:
                            conn.execute("ROLLBACK")
                            raise
                else:
                    message = "冷盘副本缺失或摘要不符，拒绝下载"
                    if row["cold_error"] != message:
                        now = utcnow()
                        with self._db() as conn:
                            conn.execute("BEGIN IMMEDIATE")
                            try:
                                conn.execute(
                                    "UPDATE deliveries SET cold_error=?, updated_at=? WHERE transfer_id=?",
                                    (message, now, tid),
                                )
                                self._audit(conn, tid, "cold-invalid", message)
                                conn.execute("COMMIT")
                            except Exception:
                                conn.execute("ROLLBACK")
                                raise
            # Any cold copy not referenced by a committed cold row is a leftover
            # from an interrupted copy or a reverted row; it must not survive.
        # Snapshot the listing before deleting anything: unlinking files also
        # prunes now-empty folders, which would invalidate a lazy directory walk.
        cold_entries = sorted(self.cold_published_dir.rglob("*"), reverse=True)
        for path in cold_entries:
            if not path.is_file() or path.suffix == ".part":
                continue
            if os.path.realpath(path) in referenced:
                continue
            self._delete_quietly(path)
        # Remove directories left empty by the sweep (but keep the root).
        for folder in cold_entries:
            if folder.is_dir():
                try:
                    folder.rmdir()
                except OSError:
                    pass

    @staticmethod
    def _delete_quietly(path: Path) -> None:
        """Unlink a file, then prune its directory when it became empty."""
        try:
            path.unlink()
        except OSError:
            return
        try:
            path.parent.rmdir()  # only succeeds when the folder became empty
        except OSError:
            pass

    # ------------------------------------------------------------------ audit
    def _audit(self, conn: sqlite3.Connection, transfer_id: str, event: str, detail: str = "") -> None:
        conn.execute(
            "INSERT INTO audit(transfer_id, event, detail, at) VALUES (?,?,?,?)",
            (transfer_id, event, detail, utcnow()),
        )

    def audit_log(self, transfer_id: str) -> list[sqlite3.Row]:
        with self._db() as conn:
            return conn.execute(
                "SELECT event, detail, at FROM audit WHERE transfer_id=? ORDER BY id",
                (normalize_transfer_id(transfer_id),),
            ).fetchall()

    # -------------------------------------------------------------- registration
    def register_delivery(
        self, transfer_id: str, frame_count: int | str, size_bytes: int | str, sha256: str
    ) -> Delivery:
        tid = normalize_transfer_id(transfer_id)
        count = parse_frame_count(frame_count)
        size = parse_size_bytes(size_bytes)
        digest = parse_sha256(sha256)
        validate_declared_size(count, size)
        now = utcnow()
        with self._critical(), self._db() as conn:
            row = conn.execute(
                "SELECT * FROM deliveries WHERE transfer_id=?", (tid,)
            ).fetchone()
            if row is not None:
                existing = _delivery_from_row(row)
                if (
                    existing.frame_count == count
                    and existing.size_bytes == size
                    and existing.sha256 == digest
                ):
                    return existing
                raise ConflictError(
                    f"传输号 {tid} 已登记且参数不同，拒绝覆盖（原登记：{existing.frame_count} 帧 / "
                    f"{existing.size_bytes} 字节 / {existing.sha256[:12]}…）"
                )
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    INSERT INTO deliveries(transfer_id, frame_count, size_bytes, sha256,
                                           status, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (tid, count, size, digest, STATUS_WAITING, now, now),
                )
                self._audit(conn, tid, "registered", f"{count} 帧 / {size} 字节 / {digest}")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return Delivery(tid, count, size, digest, now)

    # --------------------------------------------------------------- frame ingest
    def ingest_frames(self, transfer_id: str, items: Sequence[tuple[int, bytes]]) -> dict:
        """Insert data frames atomically, rejecting conflicting payloads.

        The whole batch is rejected if any (frame number, payload) conflicts with a
        stored frame or with another row of the same batch.
        """
        tid = normalize_transfer_id(transfer_id)
        if not items:
            raise StorageError("CSV 中没有数据帧")
        accepted: list[int] = []
        duplicates: list[int] = []
        conflicts: list[int] = []
        batch: dict[int, bytes] = {}

        with self._critical():
            entry = self._require_delivery(tid)
            for frame_no, payload in items:
                self._validate_frame_no(entry, frame_no)
                if len(payload) != FRAME_BYTES:
                    raise StorageError(f"第 {frame_no} 帧载荷必须为 256 字节")
                if frame_no in batch:
                    if batch[frame_no] != payload:
                        conflicts.append(frame_no)
                    else:
                        duplicates.append(frame_no)
                    continue
                stored = self._get_frame(tid, frame_no)
                if stored is not None:
                    if stored != payload:
                        conflicts.append(frame_no)
                    else:
                        duplicates.append(frame_no)
                    continue
                batch[frame_no] = payload
                accepted.append(frame_no)

            if conflicts:
                raise ConflictError(
                    "同传输号同帧号已存在不同内容，拒收本批："
                    + "、".join(str(n) for n in sorted(set(conflicts)))
                )

            if accepted:
                now = utcnow()
                with self._db() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        for frame_no in accepted:
                            payload = batch[frame_no]
                            conn.execute(
                                """
                                INSERT INTO frames(transfer_id, frame_no, payload, sha256, received_at)
                                VALUES (?,?,?,?,?)
                                """,
                                (
                                    tid,
                                    frame_no,
                                    payload,
                                    hashlib.sha256(payload).hexdigest(),
                                    now,
                                ),
                            )
                        conn.execute(
                            "UPDATE deliveries SET updated_at=? WHERE transfer_id=?", (now, tid)
                        )
                        detail = "接受帧 " + ",".join(str(n) for n in accepted)
                        if duplicates:
                            detail += "；重复帧 " + ",".join(str(n) for n in sorted(set(duplicates)))
                        self._audit(conn, tid, "frames", detail)
                        conn.execute("COMMIT")
                    except Exception:
                        conn.execute("ROLLBACK")
                        raise

            outcome = self._after_ingest_locked(tid)
        return {
            "transfer_id": tid,
            "accepted": sorted(accepted),
            "duplicates": sorted(set(duplicates)),
            "conflicts": [],
            "result": outcome,
        }

    def ingest_parity(self, transfer_id: str, payload: bytes) -> dict:
        tid = normalize_transfer_id(transfer_id)
        if len(payload) != FRAME_BYTES:
            raise StorageError("校验帧载荷必须为 256 字节")
        parity_status = "accepted"
        with self._critical():
            self._require_delivery(tid)
            stored = self._get_parity(tid)
            if stored is not None:
                if stored != payload:
                    raise ConflictError("同传输号已存在内容不同的校验帧，拒收")
                parity_status = "duplicate"
            else:
                now = utcnow()
                with self._db() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        conn.execute(
                            """
                            INSERT INTO parities(transfer_id, payload, sha256, received_at)
                            VALUES (?,?,?,?)
                            """,
                            (tid, payload, hashlib.sha256(payload).hexdigest(), now),
                        )
                        conn.execute(
                            "UPDATE deliveries SET updated_at=? WHERE transfer_id=?", (now, tid)
                        )
                        self._audit(conn, tid, "parity", "校验帧已接收")
                        conn.execute("COMMIT")
                    except Exception:
                        conn.execute("ROLLBACK")
                        raise
            outcome = self._after_ingest_locked(tid)
        return {"transfer_id": tid, "parity": parity_status, "result": outcome}

    # ------------------------------------------------------------------- state
    def state(self, transfer_id: str) -> ArchiveState:
        tid = normalize_transfer_id(transfer_id)
        row = self._delivery_row(tid)
        entry = _delivery_from_row(row)
        frames = self._get_frames(tid)
        parity = self._get_parity(tid)
        missing = tuple(
            n for n in range(1, entry.frame_count + 1) if n not in frames
        )
        status = row["status"]
        recoverable = len(missing) == 1 and parity is not None
        return ArchiveState(
            delivery=entry,
            frames=frames,
            parity=parity,
            status=status,
            published_sha256=row["pub_sha256"],
            published_size=row["pub_size"],
            published_at=row["pub_at"],
            missing_frames=missing,
            parity_recoverable=recoverable,
            actual_sha256=row["actual_sha256"],
            recovered_frames=_parse_frame_list(row["pub_recovered"]),
            tentative_frame=(
                missing[0]
                if recoverable and status != STATUS_PUBLISHED
                else None
            ),
            storage_tier=(row["storage_tier"] or TIER_HOT),
            cold_sha256=row["cold_sha256"],
            cold_size=int(row["cold_size"]) if row["cold_size"] is not None else None,
            cold_at=row["cold_at"],
            cold_error=row["cold_error"],
            cold_path=row["cold_path"],
        )

    def verify_now(self, transfer_id: str) -> dict:
        """Re-attempt assembly/verification on demand (e.g. after a bad batch)."""
        tid = normalize_transfer_id(transfer_id)
        with self._critical():
            return self._after_ingest_locked(tid)

    def list_deliveries(self) -> list[ArchiveState]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT transfer_id FROM deliveries ORDER BY created_at DESC, transfer_id"
            ).fetchall()
        return [self.state(row["transfer_id"]) for row in rows]

    # -------------------------------------------------------------- publication
    def _after_ingest_locked(self, tid: str) -> dict:
        """Attempt publication. Caller must hold ``_critical``."""
        state = self.state(tid)
        if state.status == STATUS_PUBLISHED:
            return {
                "status": STATUS_PUBLISHED,
                "published": False,
                "reason": "already-published",
            }
        if state.can_attempt:
            return self._publish_locked(tid, state)
        return {
            "status": STATUS_WAITING,
            "published": False,
            "reason": "missing-frames",
            "missing_frames": list(state.missing_frames),
        }

    def _publish_locked(self, tid: str, state: ArchiveState | None = None) -> dict:
        """Assemble, truncate, verify, then atomically publish. Holds ``_critical``."""
        state = state or self.state(tid)
        entry = state.delivery
        frames = dict(state.frames)
        recovered: list[int] = []

        if state.missing_frames:
            # Exactly one missing frame and a parity frame: recover it in memory.
            # It is not persisted; the real frame may still arrive later.
            missing_no = state.missing_frames[0]
            assert state.parity is not None
            recovered_frame = self._recover(frames, state.parity, missing_no, entry.frame_count)
            if recovered_frame is None:  # pragma: no cover - guarded by state.can_attempt
                raise StorageError("校验帧不足以恢复缺失的数据帧")
            frames[missing_no] = recovered_frame
            recovered.append(missing_no)

        padded = b"".join(frames[n] for n in range(1, entry.frame_count + 1))
        body = padded[: entry.size_bytes]
        digest = hashlib.sha256(body).hexdigest()

        if digest != entry.sha256:
            now = utcnow()
            with self._db() as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    previous = conn.execute(
                        "SELECT status, actual_sha256 FROM deliveries WHERE transfer_id=?",
                        (tid,),
                    ).fetchone()
                    conn.execute(
                        """
                        UPDATE deliveries
                           SET status=?, actual_sha256=?, updated_at=?
                         WHERE transfer_id=? AND status!=?
                        """,
                        (STATUS_MISMATCH, digest, now, tid, STATUS_PUBLISHED),
                    )
                    # Record each distinct mismatch once; repeated verify clicks
                    # with the same material must not flood the audit log.
                    if previous is None or previous["actual_sha256"] != digest:
                        self._audit(
                            conn,
                            tid,
                            "digest-mismatch",
                            f"声明 {entry.sha256}，实际 {digest}；保留分片待补帧重验"
                            + (
                                f"（第 {recovered[0]} 帧为校验帧临时恢复，未落库）"
                                if recovered
                                else ""
                            ),
                        )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
            return {
                "status": STATUS_MISMATCH,
                "published": False,
                "reason": "digest-mismatch",
                "expected_sha256": entry.sha256,
                "actual_sha256": digest,
                "recovered_frames": recovered,
            }

        # Write into tmp, fsync, move to a same-filesystem staging name, then
        # atomically rename into the content-addressed final path.
        folder = self.published_dir / tid
        folder.mkdir(parents=True, exist_ok=True)
        final_name = f"{tid}-{digest}.bin"
        final_path = folder / final_name
        tmp_path = self.tmp_dir / f"{uuid.uuid4().hex}.part"
        # Deterministic staging name: after a crash the file is still recognisable
        # as the content-addressed final file minus its rename step.
        staging_path = folder / f".{final_name}.part"

        try:
            # Never trust a leftover staging file from a crashed writer; rewrite it.
            with open(tmp_path, "wb") as fh:
                fh.write(body)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, staging_path)
            with open(staging_path, "rb+") as fh:
                os.fsync(fh.fileno())
            os.replace(staging_path, final_path)
            _fsync_dir(folder)
            _fsync_dir(self.tmp_dir)
        except BaseException:
            for path in (tmp_path, staging_path):
                try:
                    path.unlink()
                except OSError:
                    pass
            raise

        now = utcnow()
        published_here = False
        with self._db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT status FROM deliveries WHERE transfer_id=?", (tid,)
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    raise NotFoundError(f"传输号 {tid} 已不存在")
                if row["status"] != STATUS_PUBLISHED:
                    conn.execute(
                        """
                        UPDATE deliveries
                           SET status=?, pub_sha256=?, pub_size=?, pub_path=?,
                               pub_at=?, actual_sha256=?, pub_recovered=?, updated_at=?
                         WHERE transfer_id=?
                        """,
                        (
                            STATUS_PUBLISHED,
                            digest,
                            entry.size_bytes,
                            str(final_path.relative_to(self.root)),
                            now,
                            digest,
                            ",".join(str(n) for n in recovered) or None,
                            now,
                            tid,
                        ),
                    )
                    self._audit(
                        conn,
                        tid,
                        "published",
                        f"{entry.size_bytes} 字节 / {digest}"
                        + (f"；第 {recovered[0]} 帧由校验帧恢复" if recovered else ""),
                    )
                    published_here = True
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return {
            "status": STATUS_PUBLISHED,
            "published": published_here,
            "sha256": digest,
            "size_bytes": entry.size_bytes,
            "recovered_frames": recovered,
        }

    def published_path(self, transfer_id: str) -> tuple[Path, str]:
        """Return (path, digest) for a published archive, verifying before download.

        The path is verified while holding the shared lock, so it points at a
        complete archive at that instant. Callers that stream the file themselves
        should prefer :meth:`read_published`, which also reads the bytes under
        that lock and therefore cannot lose the file to a concurrent migration.
        """
        with self._shared():
            path, _, digest, _ = self._locate_published_locked(transfer_id)
            return path, digest

    def read_published(self, transfer_id: str) -> tuple[bytes, str]:
        """Read and verify a published archive into memory under the shared lock.

        Archives are at most ``MAX_FRAMES * 256`` (4 KiB), so buffering is cheap.
        Holding the shared lock for the whole read means a concurrent cold
        migration or hot-copy cleanup waits, and a download can never observe a
        file that is being removed or rewritten.
        """
        with self._shared():
            path, size, digest, tier = self._locate_published_locked(transfer_id)
            data = path.read_bytes()
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                raise ConflictError(f"{_tier_label(tier)}档案在读取时校验失败，拒绝下载")
            return data, digest

    def _locate_published_locked(self, transfer_id: str) -> tuple[Path, int, str, str]:
        """Resolve the authoritative copy and verify it. Caller holds a lock."""
        tid = normalize_transfer_id(transfer_id)
        row = self._delivery_row(tid)
        if row["status"] != STATUS_PUBLISHED or not row["pub_path"]:
            raise NotFoundError("该传输号尚未发布可下载档案")
        tier = row["storage_tier"] or TIER_HOT
        size = int(row["pub_size"])
        expected = row["pub_sha256"]
        if tier == TIER_COLD:
            if not self.cold_enabled:
                raise ConflictError("该档案位于冷存储，但冷存储卷未挂载，拒绝下载")
            if not row["cold_path"]:
                raise ConflictError("该档案标记为冷存储但缺少副本路径，拒绝下载")
            path = self.cold_root / row["cold_path"]
            if not path.is_file():
                raise NotFoundError("冷存储档案文件缺失，拒绝下载")
            _verify_file(path, size, expected, label="冷存储")
            return path, size, expected, tier
        path = self.root / row["pub_path"]
        if not path.is_file():
            raise NotFoundError("已发布档案文件缺失，拒绝下载")
        _verify_file(path, size, expected, label="快盘")
        return path, size, expected, tier

    # -------------------------------------------------------------- cold tier
    def archive_to_cold(self, transfer_id: str) -> dict:
        """Copy a published archive to the cold volume, verify, switch, clean up.

        Ordering is what makes interrupted runs safe:

        1. copy to ``cold/tmp/*.part`` and fsync;
        2. rename into the content-addressed cold path and fsync its directory;
        3. re-read the cold file and compare its size and SHA-256 with the
           registered values;
        4. commit ``storage_tier='cold'`` in SQLite (the switch point);
        5. only then delete the hot copy.

        A crash before step 4 leaves the row on the hot tier; startup deletes the
        unreferenced cold file. A crash after step 4 leaves the cold row valid and
        startup removes the now-redundant hot copy. A cold file is therefore never
        treated as migrated merely because it exists.
        """
        tid = normalize_transfer_id(transfer_id)
        if not self.cold_enabled:
            raise StorageError("未配置冷存储卷，无法迁移档案")
        with self._critical():
            row = self._delivery_row(tid)
            if row["status"] != STATUS_PUBLISHED or not row["pub_path"]:
                raise NotFoundError("只有已核验发布的档案才能迁入冷存储")
            if (row["storage_tier"] or TIER_HOT) == TIER_COLD:
                # Idempotent: already migrated. Re-validate the cold copy.
                if self._cold_copy_matches(row):
                    return {
                        "transfer_id": tid,
                        "storage_tier": TIER_COLD,
                        "migrated": False,
                        "reason": "already-cold",
                        "sha256": row["pub_sha256"],
                        "size_bytes": int(row["pub_size"]),
                    }
                raise ConflictError("档案已标记为冷存储但冷盘副本校验失败，请先排查冷盘")

            hot_path = self.root / row["pub_path"]
            size = int(row["pub_size"])
            digest = row["pub_sha256"]
            if not hot_path.is_file():
                raise NotFoundError("快盘上的已发布档案缺失，无法迁移")
            if not _file_matches(hot_path, size, digest):
                raise ConflictError("快盘上的已发布档案与登记不符，拒绝迁移")

            final_path = self._cold_path_for(tid, digest)
            tmp_path = self.cold_tmp_dir / f"{uuid.uuid4().hex}.part"
            try:
                try:
                    self._copy_verified(hot_path, tmp_path, size, digest)
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(tmp_path, final_path)
                    _fsync_dir(final_path.parent)
                    _fsync_dir(self.cold_tmp_dir)
                    # Defence in depth: verify the bytes now sitting at the final name.
                    _verify_file(final_path, size, digest, label="冷存储")
                except OSError as exc:
                    raise ColdStorageError(f"冷存储卷写入失败：{exc.strerror or exc}") from exc
            except BaseException:
                for path in (tmp_path, final_path):
                    try:
                        path.unlink()
                    except OSError:
                        pass
                self._record_cold_error(tid, "冷存储复制或校验失败，档案仍保留在快盘")
                raise

            now = utcnow()
            with self._db() as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    current = conn.execute(
                        "SELECT status, pub_sha256 FROM deliveries WHERE transfer_id=?",
                        (tid,),
                    ).fetchone()
                    if current is None or current["status"] != STATUS_PUBLISHED:
                        conn.execute("ROLLBACK")
                        self._delete_quietly(final_path)
                        raise ConflictError("档案状态已变化，拒绝切换存储位置")
                    if current["pub_sha256"] != digest:
                        conn.execute("ROLLBACK")
                        self._delete_quietly(final_path)
                        raise ConflictError("档案摘要已变化，拒绝切换存储位置")
                    conn.execute(
                        """
                        UPDATE deliveries
                           SET storage_tier=?, cold_sha256=?, cold_size=?, cold_path=?,
                               cold_at=?, cold_error=NULL, updated_at=?
                         WHERE transfer_id=?
                        """,
                        (
                            TIER_COLD,
                            digest,
                            size,
                            str(final_path.relative_to(self.cold_root)),
                            now,
                            now,
                            tid,
                        ),
                    )
                    self._audit(conn, tid, "migrated-cold", f"{size} 字节 / {digest}")
                    conn.execute("COMMIT")
                except BaseException:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    raise

            # Committed: the cold copy is now authoritative. Removing the hot copy
            # is best-effort; startup finishes it if this process dies here.
            self._delete_quietly(hot_path)
            return {
                "transfer_id": tid,
                "storage_tier": TIER_COLD,
                "migrated": True,
                "sha256": digest,
                "size_bytes": size,
                "cold_path": str(final_path.relative_to(self.cold_root)),
            }

    def _cold_path_for(self, tid: str, digest: str) -> Path:
        assert self.cold_published_dir is not None
        return self.cold_published_dir / tid / f"{tid}-{digest}.bin"

    def _cold_copy_matches(self, row: sqlite3.Row) -> bool:
        if not self.cold_enabled or not row["cold_path"]:
            return False
        path = self.cold_root / row["cold_path"]
        size = int(row["pub_size"])
        return _file_matches(path, size, row["pub_sha256"])

    def _copy_verified(self, source: Path, target: Path, size: int, digest: str) -> None:
        """Stream ``source`` into ``target`` while hashing, fsync, then compare."""
        hasher = hashlib.sha256()
        written = 0
        with open(source, "rb") as src, open(target, "wb") as dst:
            while chunk := src.read(1024 * 1024):
                hasher.update(chunk)
                written += len(chunk)
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        if written != size or hasher.hexdigest() != digest:
            raise ConflictError("快盘档案在迁移过程中发生变化，已放弃本次迁移")

    def _record_cold_error(self, tid: str, message: str) -> None:
        try:
            now = utcnow()
            with self._db() as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    conn.execute(
                        "UPDATE deliveries SET cold_error=?, updated_at=? WHERE transfer_id=?",
                        (message, now, tid),
                    )
                    self._audit(conn, tid, "cold-failed", message)
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
        except Exception:
            # Failure bookkeeping must not mask the original migration error.
            pass

    # ---------------------------------------------------------------- helpers
    def _require_delivery(self, tid: str) -> Delivery:
        return _delivery_from_row(self._delivery_row(tid))

    def _delivery_row(self, tid: str) -> sqlite3.Row:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM deliveries WHERE transfer_id=?", (tid,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"传输号 {tid} 未登记")
        return row

    def _validate_frame_no(self, entry: Delivery, frame_no: int) -> None:
        if not 1 <= frame_no <= entry.frame_count:
            raise StorageError(f"帧号必须在 1～{entry.frame_count} 之间")

    def _get_frame(self, tid: str, frame_no: int) -> bytes | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT payload, sha256 FROM frames WHERE transfer_id=? AND frame_no=?",
                (tid, frame_no),
            ).fetchone()
        if row is None:
            return None
        payload = bytes(row["payload"])
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ConflictError(f"第 {frame_no} 帧存储内容校验失败，请重新上传")
        return payload

    def _get_frames(self, tid: str) -> dict[int, bytes]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT frame_no, payload, sha256 FROM frames WHERE transfer_id=?", (tid,)
            ).fetchall()
        frames: dict[int, bytes] = {}
        for row in rows:
            payload = bytes(row["payload"])
            if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise ConflictError(f"第 {row['frame_no']} 帧存储内容校验失败，请重新上传")
            frames[int(row["frame_no"])] = payload
        return frames

    def _get_parity(self, tid: str) -> bytes | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT payload, sha256 FROM parities WHERE transfer_id=?", (tid,)
            ).fetchone()
        if row is None:
            return None
        payload = bytes(row["payload"])
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ConflictError("校验帧存储内容校验失败，请重新上传")
        return payload

    @staticmethod
    def _recover(
        frames: dict[int, bytes], parity: bytes, missing_no: int, frame_count: int
    ) -> bytes | None:
        """Recover one missing frame as parity XOR every other *padded* frame."""
        others = [
            frames[n] for n in range(1, frame_count + 1) if n != missing_no and n in frames
        ]
        if len(others) != frame_count - 1:
            return None
        return xor_bytes([parity, *others])


def _parse_frame_list(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(",") if part)


def _delivery_from_row(row: sqlite3.Row) -> Delivery:
    return Delivery(
        transfer_id=row["transfer_id"],
        frame_count=int(row["frame_count"]),
        size_bytes=int(row["size_bytes"]),
        sha256=row["sha256"],
        created_at=row["created_at"],
    )


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _file_matches(path: Path, size: int | None, digest: str | None) -> bool:
    """True only when ``path`` is a regular file of the exact size and digest."""
    if size is None or digest is None:
        return False
    try:
        if not path.is_file() or path.stat().st_size != size:
            return False
        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            while chunk := fh.read(1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest() == digest
    except OSError:
        return False


def _verify_file(path: Path, size: int, digest: str, *, label: str) -> None:
    if path.stat().st_size != size:
        raise ConflictError(f"{label}档案大小与登记不符，拒绝下载")
    actual = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1024 * 1024):
            actual.update(chunk)
    if actual.hexdigest() != digest:
        raise ConflictError(f"{label}档案摘要与登记不符，拒绝下载")


def _tier_label(tier: str) -> str:
    return "冷存储" if tier == TIER_COLD else "快盘"
