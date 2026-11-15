"""FastAPI web application for waveform archive intake and verified delivery."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import Body, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .csvintake import MAX_CSV_BYTES, parse_frames_csv
from .storage import (
    FRAME_BYTES,
    MAX_FRAMES,
    STATUS_MISMATCH,
    STATUS_PUBLISHED,
    STATUS_WAITING,
    TIER_COLD,
    TIER_HOT,
    ArchiveState,
    Storage,
    StorageError,
    normalize_transfer_id,
    parse_hex_payload,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("ARCHIVE_DATA_DIR", BASE_DIR.parent / "data"))


def _cold_root_from_env() -> str | None:
    return os.environ.get("ARCHIVE_COLD_DIR", "").strip() or None

STATUS_LABELS = {
    STATUS_WAITING: "等待补帧",
    STATUS_MISMATCH: "摘要不符",
    STATUS_PUBLISHED: "已核验发布",
}

TIER_LABELS = {
    TIER_HOT: "快盘（采集卷）",
    TIER_COLD: "冷存储卷",
}


def api_state(state: ArchiveState) -> dict[str, Any]:
    published = state.status == STATUS_PUBLISHED
    return {
        "transfer_id": state.delivery.transfer_id,
        "frame_count": state.delivery.frame_count,
        "size_bytes": state.delivery.size_bytes,
        "expected_sha256": state.delivery.sha256,
        "status": state.status,
        "status_label": STATUS_LABELS.get(state.status, state.status),
        "received_frames": list(state.received_frames),
        # A published archive is complete even when one frame was reconstructed
        # from parity: it is not stored, so it is not listed as received.
        "missing_frames": [] if published else list(state.missing_frames),
        "recovered_frames": list(state.recovered_frames),
        "has_parity": state.parity is not None,
        "parity_recoverable": state.parity_recoverable,
        "published": (
            {
                "sha256": state.published_sha256,
                "size_bytes": state.published_size,
                "at": state.published_at,
            }
            if published
            else None
        ),
        "actual_sha256": state.actual_sha256,
        "storage": (
            {
                "tier": state.storage_tier,
                "tier_label": TIER_LABELS.get(state.storage_tier, state.storage_tier),
                "cold": (
                    {
                        "sha256": state.cold_sha256,
                        "size_bytes": state.cold_size,
                        "at": state.cold_at,
                    }
                    if state.storage_tier == TIER_COLD
                    else None
                ),
                "error": state.cold_error,
            }
            if published
            else None
        ),
    }


_UNSET = object()


def create_app(
    data_root: str | os.PathLike[str] | None = None,
    cold_root: str | os.PathLike[str] | None | object = _UNSET,
) -> FastAPI:
    root = Path(data_root) if data_root is not None else DATA_ROOT
    if cold_root is _UNSET:
        cold_root = _cold_root_from_env()
    storage = Storage(root, cold_root)  # type: ignore[arg-type]
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

    app = FastAPI(
        title="地震波形原档接收台",
        description=(
            "分批接收卫星链路的波形分片，支持单帧 XOR 校验帧恢复、"
            "按原档字节数截断并以 SHA-256 核验后才开放下载。"
        ),
        version="1.0.0",
    )
    app.state.storage = storage
    app.state.data_root = root
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    # ------------------------------------------------------------------ helpers
    def render(request: Request, name: str, context: dict[str, Any], status_code: int = 200) -> HTMLResponse:
        merged = {
            "request": request,
            "statuses": STATUS_LABELS,
            "frame_bytes": FRAME_BYTES,
            "min_frames": 2,
            "max_frames": 16,
            **context,
        }
        return templates.TemplateResponse(request, name, merged, status_code=status_code)

    def error_page(request: Request, message: str, status_code: int) -> HTMLResponse:
        return render(request, "error.html", {"message": message}, status_code=status_code)

    @app.exception_handler(StorageError)
    def storage_error_handler(request: Request, exc: StorageError) -> Response:
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": exc.message}, status_code=exc.status_code)
        return error_page(request, exc.message, exc.status_code)

    def state_view(state: ArchiveState) -> dict[str, Any]:
        delivery = state.delivery
        tentative = state.tentative_frame_no
        ledger = []
        for frame_no in range(1, delivery.frame_count + 1):
            if frame_no in state.frames:
                frame_state = "received"
            elif frame_no in state.recovered_frames:
                frame_state = "recovered"  # published via parity, never stored
            elif frame_no in state.missing_frames:
                # With parity, this frame can be reconstructed tentatively.
                frame_state = "tentative" if frame_no == tentative else "missing"
            else:
                frame_state = "missing"
            ledger.append({"frame_no": frame_no, "state": frame_state})
        return {
            "state": state,
            "delivery": delivery,
            "ledger": ledger,
            "status_label": STATUS_LABELS.get(state.status, state.status),
            "received_count": len(state.frames),
            "missing": list(state.missing_frames),
            "recovered": list(state.recovered_frames),
            "tentative": tentative,
            "recoverable": tentative is not None,
            "has_parity": state.parity is not None,
            "downloadable": state.status == STATUS_PUBLISHED,
            "tier": state.storage_tier,
            "tier_label": TIER_LABELS.get(state.storage_tier, state.storage_tier),
            "cold_enabled": storage.cold_enabled,
            "is_cold": state.storage_tier == TIER_COLD,
            "cold_error": state.cold_error,
        }

    def outcome_note(state: ArchiveState) -> str:
        note = STATUS_LABELS.get(state.status, state.status)
        if state.status == STATUS_MISMATCH and state.actual_sha256:
            note += f"（实际 {state.actual_sha256[:16]}…，分片已保留）"
        if state.missing_frames and state.status != STATUS_PUBLISHED:
            note += "（缺帧 " + ",".join(map(str, state.missing_frames)) + "）"
        if state.recovered_frames and state.status == STATUS_PUBLISHED:
            note += "（第 " + ",".join(map(str, state.recovered_frames)) + " 帧由校验帧恢复）"
        return note

    def redirect_detail(transfer_id: str, key: str, message: str) -> RedirectResponse:
        return RedirectResponse(
            f"/deliveries/{transfer_id}?{key}={quote(message, safe='')}", status_code=303
        )

    # -------------------------------------------------------------------- pages
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        states = storage.list_deliveries()
        return render(
            request,
            "index.html",
            {
                "items": [state_view(s) for s in states],
                "counts": {
                    "total": len(states),
                    STATUS_WAITING: sum(1 for s in states if s.status == STATUS_WAITING),
                    STATUS_MISMATCH: sum(1 for s in states if s.status == STATUS_MISMATCH),
                    STATUS_PUBLISHED: sum(1 for s in states if s.status == STATUS_PUBLISHED),
                },
            },
        )

    @app.post("/deliveries")
    def create_delivery(
        transfer_id: str = Form(...),
        frame_count: str = Form(...),
        size_bytes: str = Form(...),
        sha256: str = Form(...),
    ) -> Response:
        delivery = storage.register_delivery(transfer_id, frame_count, size_bytes, sha256)
        return RedirectResponse(f"/deliveries/{delivery.transfer_id}", status_code=303)

    @app.get("/deliveries/{transfer_id}", response_class=HTMLResponse)
    def delivery_detail(request: Request, transfer_id: str) -> HTMLResponse:
        state = storage.state(transfer_id)
        return render(
            request,
            "detail.html",
            {
                "view": state_view(state),
                "audit": storage.audit_log(transfer_id),
                "notice": request.query_params.get("ok"),
                "problem": request.query_params.get("err"),
            },
        )

    @app.post("/deliveries/{transfer_id}/frames")
    def upload_frames_page(
        transfer_id: str,
        csv_text: str = Form(""),
        csv_file: UploadFile | None = File(None),
    ) -> Response:
        try:
            payload = _read_csv_payload(csv_text, csv_file)
            frames, parity = parse_frames_csv(payload, transfer_id)
            if frames:
                storage.ingest_frames(transfer_id, frames)
            if parity is not None:
                storage.ingest_parity(transfer_id, parity)
        except StorageError as exc:
            return redirect_detail(transfer_id, "err", exc.message)
        return redirect_detail(transfer_id, "ok", outcome_note(storage.state(transfer_id)))

    @app.post("/deliveries/{transfer_id}/parity")
    def upload_parity_page(
        transfer_id: str,
        parity_hex: str = Form(""),
        parity_file: UploadFile | None = File(None),
    ) -> Response:
        try:
            raw = parity_hex
            if parity_file is not None and parity_file.filename:
                data = parity_file.file.read(MAX_CSV_BYTES + 1)
                if len(data) > MAX_CSV_BYTES:
                    raise StorageError("校验帧文件过大")
                try:
                    raw = data.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise StorageError("校验帧文本文件必须为 UTF-8") from exc
            raw = raw.strip()
            if not raw:
                raise StorageError("请填写或上传校验帧")
            storage.ingest_parity(transfer_id, parse_hex_payload(raw))
        except StorageError as exc:
            return redirect_detail(transfer_id, "err", exc.message)
        return redirect_detail(transfer_id, "ok", outcome_note(storage.state(transfer_id)))

    @app.post("/deliveries/{transfer_id}/verify")
    def reverify_page(transfer_id: str) -> Response:
        try:
            storage.verify_now(transfer_id)
        except StorageError as exc:
            return redirect_detail(transfer_id, "err", exc.message)
        return redirect_detail(transfer_id, "ok", outcome_note(storage.state(transfer_id)))

    def archive_response(transfer_id: str) -> Response:
        """Buffer the verified archive under the shared lock, then stream it.

        Handing a path to ``FileResponse`` would release the lock before the file
        is read, and a concurrent cold migration could delete the hot copy in
        between. Reading the (at most 4 KiB) archive under the lock avoids that
        race; the response body is then already complete and verified.
        """
        data, digest = storage.read_published(transfer_id)
        tid = normalize_transfer_id(transfer_id)
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{tid}.bin"',
                "Content-Length": str(len(data)),
                "X-Archive-SHA256": digest,
                "Cache-Control": "no-store",
            },
        )

    @app.get("/deliveries/{transfer_id}/download")
    def download(transfer_id: str) -> Response:
        return archive_response(transfer_id)

    @app.post("/deliveries/{transfer_id}/archive")
    def archive_page(transfer_id: str) -> Response:
        try:
            result = storage.archive_to_cold(transfer_id)
        except StorageError as exc:
            return redirect_detail(transfer_id, "err", exc.message)
        note = (
            f"已迁入冷存储（{result['size_bytes']} 字节，SHA-256 核验一致）"
            if result["migrated"]
            else "档案已在冷存储，无需重复迁移"
        )
        return redirect_detail(transfer_id, "ok", note)

    # ---------------------------------------------------------------------- API
    @app.post("/api/deliveries")
    def api_create(payload: dict = Body(...)) -> JSONResponse:
        delivery = storage.register_delivery(
            str(payload.get("transfer_id", "")),
            payload.get("frame_count", ""),
            payload.get("size_bytes", ""),
            str(payload.get("sha256", "")),
        )
        return JSONResponse(
            {
                "transfer_id": delivery.transfer_id,
                "frame_count": delivery.frame_count,
                "size_bytes": delivery.size_bytes,
                "sha256": delivery.sha256,
                "status": storage.state(delivery.transfer_id).status,
            },
            status_code=201,
        )

    @app.post("/api/deliveries/{transfer_id}/frames")
    def api_frames(transfer_id: str, payload: dict = Body(...)) -> JSONResponse:
        rows = payload.get("frames", [])
        if not isinstance(rows, list) or not rows:
            raise StorageError("frames 必须是非空数组")
        if len(rows) > MAX_FRAMES:
            raise StorageError(f"单批最多 {MAX_FRAMES} 帧")
        items = []
        for row in rows:
            if not isinstance(row, dict):
                raise StorageError("每帧需包含整数 frame_no 与十六进制 payload_hex")
            try:
                frame_no = int(row["frame_no"])
            except (KeyError, TypeError, ValueError) as exc:
                raise StorageError("每帧需包含整数 frame_no") from exc
            items.append(
                (frame_no, parse_hex_payload(str(row.get("payload_hex", "")), frame_no=frame_no))
            )
        return JSONResponse(storage.ingest_frames(transfer_id, items))

    @app.post("/api/deliveries/{transfer_id}/parity")
    def api_parity(transfer_id: str, payload: dict = Body(...)) -> JSONResponse:
        parity = parse_hex_payload(str(payload.get("payload_hex", "")))
        return JSONResponse(storage.ingest_parity(transfer_id, parity))

    @app.post("/api/deliveries/{transfer_id}/verify")
    def api_verify(transfer_id: str) -> JSONResponse:
        return JSONResponse(storage.verify_now(transfer_id))

    @app.get("/api/deliveries")
    def api_list() -> JSONResponse:
        return JSONResponse({"deliveries": [api_state(s) for s in storage.list_deliveries()]})

    @app.get("/api/deliveries/{transfer_id}")
    def api_detail(transfer_id: str) -> JSONResponse:
        return JSONResponse(api_state(storage.state(transfer_id)))

    @app.get("/api/deliveries/{transfer_id}/download")
    def api_download(transfer_id: str) -> Response:
        return archive_response(transfer_id)

    @app.post("/api/deliveries/{transfer_id}/archive")
    def api_archive(transfer_id: str) -> JSONResponse:
        return JSONResponse(storage.archive_to_cold(transfer_id))

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "data_dir": str(root),
                "cold_dir": str(storage.cold_root) if storage.cold_enabled else None,
            }
        )

    return app


def _read_csv_payload(csv_text: str, csv_file: UploadFile | None) -> bytes:
    if csv_file is not None and csv_file.filename:
        data = csv_file.file.read(MAX_CSV_BYTES + 1)
        if len(data) > MAX_CSV_BYTES:
            raise StorageError("CSV 文件过大（上限 256 KiB）")
        if not data.strip():
            raise StorageError("上传的 CSV 文件为空")
        return data
    if csv_text.strip():
        data = csv_text.encode("utf-8")
        if len(data) > MAX_CSV_BYTES:
            raise StorageError("粘贴的 CSV 内容过大（上限 256 KiB）")
        return data
    raise StorageError("请粘贴 CSV 内容或选择 CSV 文件")


app = create_app()
