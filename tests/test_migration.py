"""Cold-storage migration: verified copy, atomic tier switch, crash recovery."""

import hashlib
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import create_app  # noqa: E402
from app.storage import (  # noqa: E402
    FRAME_BYTES,
    STATUS_PUBLISHED,
    TIER_COLD,
    TIER_HOT,
    ConflictError,
    NotFoundError,
    Storage,
    StorageError,
    xor_bytes,
)


def make_archive(size: int, seed: int = 1) -> bytes:
    return bytes((i * 37 + seed * 11) % 256 for i in range(size))


def pad_frames(archive: bytes, count: int) -> list[bytes]:
    padded = archive + b"\x00" * (count * FRAME_BYTES - len(archive))
    return [padded[i * FRAME_BYTES : (i + 1) * FRAME_BYTES] for i in range(count)]


def publish_delivery(hot: Path, cold: Path | None, *, count=3, size=None, tid="M-1"):
    """Register + upload a complete delivery; return (store, archive, digest)."""
    archive = make_archive(size or (count * FRAME_BYTES - 100))
    frames = pad_frames(archive, count)
    digest = hashlib.sha256(archive).hexdigest()
    store = Storage(hot, cold_root=cold)
    store.register_delivery(tid, count, len(archive), digest)
    store.ingest_frames(tid, [(i, f) for i, f in enumerate(frames, 1)])
    assert store.state(tid).status == STATUS_PUBLISHED
    return store, archive, digest


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "hot", tmp_path / "cold"


def hot_file(store: Storage, tid: str) -> Path:
    return store.root / store._delivery_row(tid)["pub_path"]


def cold_file(store: Storage, tid: str) -> Path:
    return store.cold_root / store._delivery_row(tid)["pub_path"]


# ------------------------------------------------------------------ happy path
def test_migrate_to_cold_switches_tier_and_cleans_hot(roots):
    hot, cold = roots
    store, archive, digest = publish_delivery(hot, cold, tid="MIG-1")
    hot_path = hot_file(store, "MIG-1")
    assert hot_path.exists()

    result = store.migrate_to_cold("MIG-1")
    assert result["migrated"] is True
    assert result["tier"] == TIER_COLD
    assert result["hot_copy_removed"] is True

    state = store.state("MIG-1")
    assert state.tier == TIER_COLD
    assert not hot_path.exists()  # fast-disk copy cleaned up
    cold_path = cold_file(store, "MIG-1")
    assert cold_path.read_bytes() == archive

    # Download path is unchanged and serves the exact registered bytes.
    data, dl_digest, tier = store.read_published("MIG-1")
    assert data == archive
    assert dl_digest == digest
    assert tier == TIER_COLD
    path, _ = store.published_path("MIG-1")
    assert path == cold_path

    events = [row["event"] for row in store.audit_log("MIG-1")]
    assert "migrate-copy" in events
    assert "migrate-switch" in events
    assert "migrate-cleanup" in events


def test_migrate_is_idempotent(roots):
    hot, cold = roots
    store, archive, _ = publish_delivery(hot, cold, tid="MIG-2")
    store.migrate_to_cold("MIG-2")
    again = store.migrate_to_cold("MIG-2")
    assert again["migrated"] is False
    assert again["reason"] == "already-cold"
    assert store.read_published("MIG-2")[0] == archive
    # Exactly one copy remains, on the cold tier.
    assert not hot_file(store, "MIG-2").exists()
    assert cold_file(store, "MIG-2").read_bytes() == archive


def test_migrate_requires_published_delivery(roots):
    hot, cold = roots
    store = Storage(hot, cold_root=cold)
    store.register_delivery("WAIT-1", 3, 600, "00" * 32)
    with pytest.raises(ConflictError, match="已核验发布"):
        store.migrate_to_cold("WAIT-1")
    with pytest.raises(NotFoundError):
        store.migrate_to_cold("NOPE")


def test_migrate_without_cold_volume_fails_and_keeps_hot(tmp_path):
    store, archive, _ = publish_delivery(tmp_path / "hot", None, tid="NOCOLD")
    with pytest.raises(StorageError, match="冷存储"):
        store.migrate_to_cold("NOCOLD")
    assert store.state("NOCOLD").tier == TIER_HOT
    assert store.read_published("NOCOLD")[0] == archive


def test_migrate_aborts_when_hot_source_is_corrupt(roots):
    hot, cold = roots
    store, archive, _ = publish_delivery(hot, cold, tid="CORRUPT-SRC")
    hot_file(store, "CORRUPT-SRC").write_bytes(b"tampered")
    with pytest.raises(ConflictError, match="不符"):
        store.migrate_to_cold("CORRUPT-SRC")
    # Nothing switched, nothing left behind on the cold tier.
    assert store.state("CORRUPT-SRC").tier == TIER_HOT
    assert not cold_file(store, "CORRUPT-SRC").exists()
    assert list(store.cold_tmp_dir.glob("*.part")) == []


# ------------------------------------------------------------------ crash safety
def test_interrupted_copy_leaves_no_partial_and_retry_succeeds(roots):
    hot, cold = roots
    store, archive, digest = publish_delivery(hot, cold, tid="CRASH-COPY")
    # Simulate a crash mid-copy: a cold temp fragment exists, tier still hot.
    store.cold_tmp_dir.mkdir(parents=True, exist_ok=True)
    (store.cold_tmp_dir / "deadbeef.part").write_bytes(b"half-written garbage")

    restarted = Storage(hot, cold_root=cold)  # restart cleans cold partials
    assert list(restarted.cold_tmp_dir.glob("*.part")) == []
    assert restarted.state("CRASH-COPY").tier == TIER_HOT
    assert restarted.read_published("CRASH-COPY")[0] == archive

    result = restarted.migrate_to_cold("CRASH-COPY")
    assert result["migrated"] is True
    assert restarted.read_published("CRASH-COPY") == (archive, digest, TIER_COLD)


def test_interrupted_switch_is_completed_by_retry(roots):
    hot, cold = roots
    store, archive, digest = publish_delivery(hot, cold, tid="CRASH-SWITCH")
    # Simulate a crash after the cold rename but before the DB switch:
    # a complete, verified cold copy exists while the tier is still hot.
    rel = store._delivery_row("CRASH-SWITCH")["pub_path"]
    cold_final = cold / rel
    cold_final.parent.mkdir(parents=True, exist_ok=True)
    cold_final.write_bytes(archive)

    restarted = Storage(hot, cold_root=cold)
    # Hot copy is intact, so startup keeps the hot tier authoritative...
    assert restarted.state("CRASH-SWITCH").tier == TIER_HOT
    # ...and a retried migration reuses the re-verified cold copy.
    result = restarted.migrate_to_cold("CRASH-SWITCH")
    assert result["copied"] is False  # no redundant recopy
    assert result["migrated"] is True
    assert restarted.read_published("CRASH-SWITCH") == (archive, digest, TIER_COLD)
    assert not (hot / rel).exists()


def test_interrupted_switch_with_lost_hot_copy_self_heals(roots):
    hot, cold = roots
    store, archive, digest = publish_delivery(hot, cold, tid="CRASH-HEAL")
    rel = store._delivery_row("CRASH-HEAL")["pub_path"]
    cold_final = cold / rel
    cold_final.parent.mkdir(parents=True, exist_ok=True)
    cold_final.write_bytes(archive)
    (hot / rel).unlink()  # hot copy lost before the switch was recorded

    restarted = Storage(hot, cold_root=cold)
    assert restarted.state("CRASH-HEAL").tier == TIER_COLD
    assert restarted.read_published("CRASH-HEAL") == (archive, digest, TIER_COLD)
    events = [row["event"] for row in restarted.audit_log("CRASH-HEAL")]
    assert "migrate-resume" in events


def test_interrupted_cleanup_is_finished_at_startup(roots):
    hot, cold = roots
    store, archive, digest = publish_delivery(hot, cold, tid="CRASH-CLEAN")
    store.migrate_to_cold("CRASH-CLEAN")
    # Simulate a crash during cleanup: tier is cold, hot copy reappears.
    rel = store._delivery_row("CRASH-CLEAN")["pub_path"]
    leftover = hot / rel
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_bytes(archive)

    restarted = Storage(hot, cold_root=cold)
    assert not leftover.exists()  # startup reconciliation removed it
    assert restarted.read_published("CRASH-CLEAN") == (archive, digest, TIER_COLD)


def test_corrupt_cold_copy_rolls_back_to_hot_at_startup(roots):
    hot, cold = roots
    store, archive, digest = publish_delivery(hot, cold, tid="ROLLBACK")
    store.migrate_to_cold("ROLLBACK")
    rel = store._delivery_row("ROLLBACK")["pub_path"]
    # Cold copy tampered with; the hot copy unexpectedly survives -> roll back.
    cold_file(store, "ROLLBACK").write_bytes(b"bit rot")
    (hot / rel).parent.mkdir(parents=True, exist_ok=True)
    (hot / rel).write_bytes(archive)

    restarted = Storage(hot, cold_root=cold)
    assert restarted.state("ROLLBACK").tier == TIER_HOT
    assert restarted.read_published("ROLLBACK") == (archive, digest, TIER_HOT)
    events = [row["event"] for row in restarted.audit_log("ROLLBACK")]
    assert "migrate-rollback" in events


def test_unverified_cold_leftover_is_never_treated_as_migrated(roots):
    hot, cold = roots
    store, archive, digest = publish_delivery(hot, cold, tid="LEFTOVER")
    # A corrupt file sits at the cold final path (e.g. tampered leftover).
    rel = store._delivery_row("LEFTOVER")["pub_path"]
    cold_final = cold / rel
    cold_final.parent.mkdir(parents=True, exist_ok=True)
    cold_final.write_bytes(b"definitely not the archive")

    restarted = Storage(hot, cold_root=cold)
    assert restarted.state("LEFTOVER").tier == TIER_HOT  # no flip to bad copy
    assert restarted.read_published("LEFTOVER") == (archive, digest, TIER_HOT)

    result = restarted.migrate_to_cold("LEFTOVER")  # recopies from the hot source
    assert result["copied"] is True
    assert restarted.read_published("LEFTOVER") == (archive, digest, TIER_COLD)
    assert cold_final.read_bytes() == archive


def test_both_copies_lost_fails_closed(roots):
    hot, cold = roots
    store, archive, _ = publish_delivery(hot, cold, tid="LOST")
    store.migrate_to_cold("LOST")
    cold_file(store, "LOST").unlink()
    restarted = Storage(hot, cold_root=cold)
    with pytest.raises(NotFoundError):
        restarted.read_published("LOST")


# ------------------------------------------------------------------ concurrency
def test_concurrent_migrations_and_downloads_stay_consistent(roots):
    hot, cold = roots
    store, archive, digest = publish_delivery(hot, cold, count=6, tid="RACE-MIG")
    barrier = threading.Barrier(6)
    errors: list[BaseException] = []

    def migrator() -> None:
        local = Storage(hot, cold_root=cold)  # a second web instance
        try:
            barrier.wait(timeout=10)
            local.migrate_to_cold("RACE-MIG")
        except BaseException as exc:  # pragma: no cover - failure branch
            errors.append(exc)

    def downloader() -> None:
        local = Storage(hot, cold_root=cold)
        try:
            barrier.wait(timeout=10)
            for _ in range(40):
                data, dl_digest, _tier = local.read_published("RACE-MIG")
                assert data == archive
                assert dl_digest == digest
        except BaseException as exc:  # pragma: no cover - failure branch
            errors.append(exc)

    threads = [threading.Thread(target=migrator) for _ in range(2)] + [
        threading.Thread(target=downloader) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors
    assert store.state("RACE-MIG").tier == TIER_COLD
    assert store.read_published("RACE-MIG") == (archive, digest, TIER_COLD)
    # Exactly one cold copy, no hot copy, no temp fragments anywhere.
    assert cold_file(store, "RACE-MIG").read_bytes() == archive
    assert not hot_file(store, "RACE-MIG").exists()
    assert list(store.cold_tmp_dir.glob("*.part")) == []
    assert list(store.tmp_dir.glob("*.part")) == []


# ------------------------------------------------------------------ web flow
@pytest.fixture
def web_client(tmp_path):
    from fastapi.testclient import TestClient

    app = create_app(tmp_path / "data", cold_root=tmp_path / "cold")
    return TestClient(app), tmp_path


def upload_complete(client, tid: str, count: int, size: int) -> bytes:
    archive = bytes((i * 13 + 5) % 256 for i in range(size))
    padded = archive + b"\x00" * (count * FRAME_BYTES - size)
    frames = [padded[i * FRAME_BYTES : (i + 1) * FRAME_BYTES] for i in range(count)]
    digest = hashlib.sha256(archive).hexdigest()
    client.post(
        "/deliveries",
        data={"transfer_id": tid, "frame_count": str(count),
              "size_bytes": str(size), "sha256": digest},
    )
    csv_text = "".join(f"{i},{f.hex()}\n" for i, f in enumerate(frames, 1))
    client.post(f"/deliveries/{tid}/frames", data={"csv_text": csv_text})
    return archive


def test_web_migration_flow_keeps_download_url(web_client):
    client, tmp_path = web_client
    archive = upload_complete(client, "WEB-M1", 4, 900)
    digest = hashlib.sha256(archive).hexdigest()

    detail = client.get("/deliveries/WEB-M1")
    assert "热层·采集盘" in detail.text
    assert "/deliveries/WEB-M1/migrate" in detail.text

    before = client.get("/deliveries/WEB-M1/download")
    assert before.status_code == 200
    assert before.headers["x-archive-tier"] == TIER_HOT

    response = client.post("/deliveries/WEB-M1/migrate", follow_redirects=False)
    assert response.status_code == 303
    assert "ok=" in response.headers["location"]

    followed = client.post("/deliveries/WEB-M1/migrate")  # follows the redirect
    assert "已在冷层" in followed.text  # idempotent notice rendered on the page

    detail = client.get("/deliveries/WEB-M1")
    assert "冷层·归档卷" in detail.text

    # Same download URL, same bytes, now served from the cold tier.
    after = client.get("/deliveries/WEB-M1/download")
    assert after.status_code == 200
    assert after.content == archive
    assert after.headers["x-archive-sha256"] == digest
    assert after.headers["x-archive-tier"] == TIER_COLD

    # The fast-disk copy is gone; the cold copy holds the archive.
    assert not list((tmp_path / "data" / "published" / "WEB-M1").glob("*.bin"))
    cold_files = list((tmp_path / "cold" / "published" / "WEB-M1").glob("*.bin"))
    assert len(cold_files) == 1
    assert cold_files[0].read_bytes() == archive

    # The index page shows the storage tier.
    assert "冷层·归档卷" in client.get("/").text


def test_web_migrate_unpublished_is_rejected(web_client):
    client, _ = web_client
    client.post(
        "/deliveries",
        data={"transfer_id": "WEB-M2", "frame_count": "3", "size_bytes": "600",
              "sha256": "00" * 32},
    )
    response = client.post("/deliveries/WEB-M2/migrate", follow_redirects=False)
    assert response.status_code == 303
    assert "err=" in response.headers["location"]
    assert client.get("/deliveries/WEB-M2/download").status_code == 404


def test_api_migration_flow(web_client):
    client, _ = web_client
    archive = upload_complete(client, "API-M1", 3, 700)
    state = client.get("/api/deliveries/API-M1").json()
    assert state["tier"] == TIER_HOT
    assert state["tier_label"] == "热层·采集盘"

    result = client.post("/api/deliveries/API-M1/migrate")
    assert result.status_code == 200
    assert result.json()["tier"] == TIER_COLD

    state = client.get("/api/deliveries/API-M1").json()
    assert state["tier"] == TIER_COLD
    assert state["tier_label"] == "冷层·归档卷"

    download = client.get("/api/deliveries/API-M1/download")
    assert download.content == archive
    assert download.headers["x-archive-tier"] == TIER_COLD

    again = client.post("/api/deliveries/API-M1/migrate").json()
    assert again["migrated"] is False


def test_migrate_disabled_without_cold_volume(tmp_path):
    from fastapi.testclient import TestClient

    client = TestClient(create_app(tmp_path / "data"))
    archive = upload_complete(client, "WEB-M3", 3, 600)
    detail = client.get("/deliveries/WEB-M3")
    assert "/deliveries/WEB-M3/migrate" not in detail.text  # no button
    response = client.post("/deliveries/WEB-M3/migrate", follow_redirects=False)
    assert response.status_code == 303
    assert "err=" in response.headers["location"]
    assert client.get("/deliveries/WEB-M3/download").content == archive
    api = client.post("/api/deliveries/WEB-M3/migrate")
    assert api.status_code == 400
