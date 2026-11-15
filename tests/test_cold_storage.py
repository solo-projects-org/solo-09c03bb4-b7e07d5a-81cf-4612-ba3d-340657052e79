"""Cold-tier migration: verified copy, atomic switch, cleanup, crash and races."""

import hashlib
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.storage import (  # noqa: E402
    FRAME_BYTES,
    STATUS_PUBLISHED,
    TIER_COLD,
    TIER_HOT,
    ConflictError,
    Storage,
    StorageError,
    xor_bytes,
)


def make_archive(size: int, seed: int = 3) -> bytes:
    return bytes((i * 29 + seed * 7) % 256 for i in range(size))


def pad_frames(archive: bytes, count: int) -> list[bytes]:
    padded = archive + b"\x00" * (count * FRAME_BYTES - len(archive))
    return [padded[i * FRAME_BYTES : (i + 1) * FRAME_BYTES] for i in range(count)]


def published_store(root: Path, cold: Path, *, tid="COLD-1", count=4) -> tuple[Storage, bytes]:
    archive = make_archive(count * FRAME_BYTES - 77)
    frames = pad_frames(archive, count)
    store = Storage(root, cold)
    store.register_delivery(tid, count, len(archive), hashlib.sha256(archive).hexdigest())
    store.ingest_frames(tid, [(i, f) for i, f in enumerate(frames, 1)])
    assert store.state(tid).status == STATUS_PUBLISHED
    return store, archive


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "hot", tmp_path / "cold"


# ----------------------------------------------------------------- happy path
def test_archive_moves_to_cold_and_removes_hot_copy(roots):
    hot, cold = roots
    store, archive = published_store(hot, cold)
    hot_path = store.published_path("COLD-1")[0]
    assert hot_path.read_bytes() == archive

    result = store.archive_to_cold("COLD-1")
    assert result["migrated"] is True
    assert result["storage_tier"] == TIER_COLD

    state = store.state("COLD-1")
    assert state.storage_tier == TIER_COLD
    assert state.cold_sha256 == hashlib.sha256(archive).hexdigest()
    assert state.cold_size == len(archive)
    # The hot copy has been cleaned up; the cold copy is the only one left.
    assert not hot_path.exists()
    cold_files = list((cold / "published").rglob("*.bin"))
    assert len(cold_files) == 1
    assert cold_files[0].read_bytes() == archive
    # The original download path still serves the archive, now from cold.
    path, digest = store.published_path("COLD-1")
    assert path == cold_files[0]
    assert path.read_bytes() == archive
    assert digest == hashlib.sha256(archive).hexdigest()
    assert list((cold / "tmp").glob("*")) == []


def test_archive_is_idempotent(roots):
    hot, cold = roots
    store, archive = published_store(hot, cold)
    store.archive_to_cold("COLD-1")
    again = store.archive_to_cold("COLD-1")
    assert again["migrated"] is False
    assert again["reason"] == "already-cold"
    assert store.published_path("COLD-1")[0].read_bytes() == archive


def test_archive_requires_published_delivery(roots):
    hot, cold = roots
    store = Storage(hot, cold)
    store.register_delivery("WAIT", 3, 700, "0" * 64)
    with pytest.raises(StorageError):
        store.archive_to_cold("WAIT")
    with pytest.raises(StorageError):
        store.archive_to_cold("MISSING")


def test_archive_without_cold_volume_is_refused(tmp_path):
    store = Storage(tmp_path / "solo")
    store.register_delivery("T", 2, 400, "0" * 64)
    with pytest.raises(StorageError, match="冷存储"):
        store.archive_to_cold("T")


def test_cold_root_must_be_separate_tree(tmp_path):
    with pytest.raises(ValueError):
        Storage(tmp_path / "same", tmp_path / "same")
    with pytest.raises(ValueError):
        Storage(tmp_path / "a", tmp_path / "a" / "cold")


# ------------------------------------------------------------------- failures
def test_corrupt_hot_copy_blocks_migration(roots):
    hot, cold = roots
    store, archive = published_store(hot, cold)
    hot_path = store.published_path("COLD-1")[0]
    hot_path.write_bytes(b"rotten")
    with pytest.raises(StorageError):
        store.archive_to_cold("COLD-1")
    assert store.state("COLD-1").storage_tier == TIER_HOT
    assert not list((cold / "published").rglob("*.bin"))


def test_cold_copy_is_rechecked_after_write(roots, monkeypatch):
    hot, cold = roots
    store, archive = published_store(hot, cold)

    def corrupt_after_rename(source, target, size, digest):
        target.write_bytes(b"truncated writes")

    monkeypatch.setattr(store, "_copy_verified", corrupt_after_rename)
    with pytest.raises(StorageError):
        store.archive_to_cold("COLD-1")
    # Failed verification must not switch the tier nor leave a cold remnant.
    assert store.state("COLD-1").storage_tier == TIER_HOT
    assert store.state("COLD-1").cold_error
    assert not list((cold / "published").rglob("*.bin"))
    assert store.published_path("COLD-1")[0].read_bytes() == archive


def test_cold_write_error_is_reported_and_hot_copy_kept(roots, monkeypatch):
    hot, cold = roots
    store, archive = published_store(hot, cold)

    def boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(store, "_copy_verified", boom)
    with pytest.raises(StorageError, match="冷存储"):
        store.archive_to_cold("COLD-1")
    assert store.state("COLD-1").storage_tier == TIER_HOT
    assert store.published_path("COLD-1")[0].read_bytes() == archive


# ------------------------------------------------------ interrupted migrations
def test_interrupted_copy_before_switch_is_discarded_on_restart(roots):
    hot, cold = roots
    store, archive = published_store(hot, cold)
    # Simulate a crash after the cold file was renamed into place but before the
    # database switched tiers: a complete but unreferenced cold file remains.
    digest = hashlib.sha256(archive).hexdigest()
    stray = cold / "published" / "COLD-1" / f"COLD-1-{digest}.bin"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(archive)
    (cold / "tmp" / "abc.part").write_bytes(b"partial copy")
    assert store.state("COLD-1").storage_tier == TIER_HOT

    restarted = Storage(hot, cold)
    state = restarted.state("COLD-1")
    assert state.storage_tier == TIER_HOT
    assert not stray.exists()
    assert list((cold / "tmp").glob("*.part")) == []
    # The hot copy is intact and the migration can simply be retried.
    assert restarted.published_path("COLD-1")[0].read_bytes() == archive
    assert restarted.archive_to_cold("COLD-1")["migrated"] is True
    assert restarted.published_path("COLD-1")[0].read_bytes() == archive


def test_interrupted_cleanup_after_switch_is_finished_on_restart(roots):
    hot, cold = roots
    store, archive = published_store(hot, cold)
    hot_path = store.published_path("COLD-1")[0]
    store.archive_to_cold("COLD-1")
    # Simulate a crash after the DB commit but before the hot copy was deleted.
    hot_path.parent.mkdir(parents=True, exist_ok=True)
    hot_path.write_bytes(archive)
    assert store.state("COLD-1").storage_tier == TIER_COLD

    restarted = Storage(hot, cold)
    assert restarted.state("COLD-1").storage_tier == TIER_COLD
    assert not hot_path.exists()
    assert restarted.published_path("COLD-1")[0].read_bytes() == archive


def test_missing_cold_copy_falls_back_to_verified_hot_copy(roots):
    hot, cold = roots
    store, archive = published_store(hot, cold)
    hot_path = store.published_path("COLD-1")[0]
    store.archive_to_cold("COLD-1")
    # Both copies vanish on the cold side; the hot copy reappears intact.
    for path in (cold / "published").rglob("*.bin"):
        path.unlink()
    hot_path.parent.mkdir(parents=True, exist_ok=True)
    hot_path.write_bytes(archive)

    restarted = Storage(hot, cold)
    state = restarted.state("COLD-1")
    assert state.storage_tier == TIER_HOT
    assert restarted.published_path("COLD-1")[0] == hot_path
    assert restarted.published_path("COLD-1")[0].read_bytes() == archive


def test_missing_cold_copy_without_hot_copy_fails_closed(roots):
    hot, cold = roots
    store, _ = published_store(hot, cold)
    store.archive_to_cold("COLD-1")
    for path in (cold / "published").rglob("*.bin"):
        path.unlink()
    with pytest.raises(StorageError):
        store.published_path("COLD-1")
    restarted = Storage(hot, cold)
    with pytest.raises(StorageError):
        restarted.published_path("COLD-1")
    assert restarted.state("COLD-1").cold_error
    assert restarted.state("COLD-1").storage_tier == TIER_COLD


def test_corrupt_cold_copy_is_never_served(roots):
    hot, cold = roots
    store, _ = published_store(hot, cold)
    store.archive_to_cold("COLD-1")
    for path in (cold / "published").rglob("*.bin"):
        path.write_bytes(b"tampered")
    with pytest.raises(StorageError):
        store.published_path("COLD-1")
    with pytest.raises(StorageError):
        store.read_published("COLD-1")


def test_cold_archive_needs_mounted_cold_volume(roots):
    hot, cold = roots
    store, _ = published_store(hot, cold)
    store.archive_to_cold("COLD-1")
    without_cold = Storage(hot)  # cold volume not mounted in this instance
    state = without_cold.state("COLD-1")
    assert state.storage_tier == TIER_COLD
    with pytest.raises(StorageError, match="冷存储"):
        without_cold.published_path("COLD-1")


# ------------------------------------------------------------------ concurrency
def test_concurrent_migrations_across_instances_move_once(roots):
    hot, cold = roots
    store, archive = published_store(hot, cold)
    barrier = threading.Barrier(6)
    results: list[bool] = []
    errors: list[BaseException] = []

    def worker() -> None:
        local = Storage(hot, cold)
        try:
            barrier.wait(timeout=10)
            results.append(local.archive_to_cold("COLD-1")["migrated"])
        except BaseException as exc:  # pragma: no cover - failure branch
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors
    assert results.count(True) == 1  # exactly one instance performs the switch
    assert store.state("COLD-1").storage_tier == TIER_COLD
    assert not list((hot / "published").rglob("*.bin"))
    cold_files = list((cold / "published").rglob("*.bin"))
    assert len(cold_files) == 1 and cold_files[0].read_bytes() == archive


def test_downloads_during_migration_only_see_complete_archives(roots):
    """Readers racing the migration get the exact archive or nothing, never junk."""
    import time

    hot, cold = roots
    store, archive = published_store(hot, cold)
    digest = hashlib.sha256(archive).hexdigest()
    start = threading.Barrier(5)
    observations: list[str] = []
    errors: list[BaseException] = []

    def reader() -> None:
        local = Storage(hot, cold)
        try:
            start.wait(timeout=10)
            for _ in range(60):
                try:
                    data, got = local.read_published("COLD-1")
                except StorageError:
                    observations.append("error")
                else:
                    assert got == digest
                    observations.append("ok" if data == archive else "corrupt")
                time.sleep(0.001)
        except BaseException as exc:  # pragma: no cover - failure branch
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    start.wait(timeout=10)
    for _ in range(3):
        store.archive_to_cold("COLD-1")
    for thread in threads:
        thread.join(timeout=60)
    assert not errors
    assert "corrupt" not in observations
    assert "ok" in observations
    assert cold.exists()


def test_two_instances_download_while_other_migrates(roots):
    hot, cold = roots
    store, archive = published_store(hot, cold)
    reader_store = Storage(hot, cold)
    start = threading.Barrier(2)
    results: list[bytes] = []
    errors: list[BaseException] = []

    def archive_worker() -> None:
        try:
            start.wait(timeout=10)
            store.archive_to_cold("COLD-1")
        except BaseException as exc:  # pragma: no cover - failure branch
            errors.append(exc)

    def download_worker() -> None:
        try:
            start.wait(timeout=10)
            for _ in range(20):
                data, _ = reader_store.read_published("COLD-1")
                results.append(data)
        except BaseException as exc:  # pragma: no cover - failure branch
            errors.append(exc)

    threads = [threading.Thread(target=archive_worker), threading.Thread(target=download_worker)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors
    assert results and all(data == archive for data in results)


# ---------------------------------------------------------- late fragment after
def test_late_fragment_after_migration_does_not_republish_to_hot(roots):
    hot, cold = roots
    store, archive = published_store(hot, cold)
    frames = pad_frames(archive, 4)
    store.archive_to_cold("COLD-1")
    store.ingest_frames("COLD-1", [(1, frames[0])])  # duplicate, late
    assert store.state("COLD-1").storage_tier == TIER_COLD
    assert store.published_path("COLD-1")[0].read_bytes() == archive
    assert not list((hot / "published").rglob("*.bin"))


def test_conflicting_parity_still_rejected_after_migration(roots):
    hot, cold = roots
    store, archive = published_store(hot, cold)
    frames = pad_frames(archive, 4)
    store.ingest_parity("COLD-1", xor_bytes(frames))
    store.archive_to_cold("COLD-1")
    with pytest.raises(ConflictError):
        store.ingest_parity("COLD-1", bytes(256))
    assert store.published_path("COLD-1")[0].read_bytes() == archive
