"""FastAPI entry point for the EdgeSplit V1 router."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from v1 import V1Error, V1Orchestrator, V1Request, V1Settings
from v2 import V2Error, V2Orchestrator, V2Settings


ROOT = Path(__file__).resolve().parents[1]


class GenerateBody(BaseModel):
    prompt: str = Field(min_length=1)
    n_predict: int = Field(default=64, ge=1, le=512)
    slot_id: int = Field(default=0, ge=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 42


def settings_from_env() -> V1Settings:
    return V1Settings(
        laptop_url=os.environ.get("EDGESPLIT_LAPTOP_URL", "http://127.0.0.1:8080"),
        phone_url=os.environ.get("EDGESPLIT_PHONE_URL", "http://YOUR_PHONE_IP:8081"),
        phone_upload_url=os.environ.get(
            "EDGESPLIT_PHONE_UPLOAD_URL", "http://YOUR_PHONE_IP:8090"
        ),
        laptop_state_dir=Path(
            os.environ.get("EDGESPLIT_LAPTOP_STATE_DIR", str(ROOT / "state"))
        ),
        benchmark_db=Path(
            os.environ.get("EDGESPLIT_BENCH_DB", str(ROOT / "bench/edgesplit.sqlite3"))
        ),
        model_name=os.environ.get("EDGESPLIT_MODEL_NAME", "Qwen3-0.6B"),
        quant=os.environ.get("EDGESPLIT_QUANT", "Q4_K_M"),
        runtime_label=os.environ.get(
            "EDGESPLIT_RUNTIME_LABEL",
            "laptop:e920c523e3b8a0163fe498af5bf90df35ff51d25;"
            "phone:official-prebuilt-b10034",
        ),
        timeout_seconds=float(os.environ.get("EDGESPLIT_HTTP_TIMEOUT_SECONDS", "120")),
        keep_state_files=os.environ.get("EDGESPLIT_KEEP_STATE_FILES", "") == "1",
    )


def v2_settings_from_env() -> V2Settings:
    phone_url = os.environ.get("EDGESPLIT_V2_PHONE_URL", "http://YOUR_V2_PHONE_IP:8081")
    return V2Settings(
        laptop_url=os.environ.get("EDGESPLIT_LAPTOP_URL", "http://127.0.0.1:8080"),
        phone_url=phone_url,
        phone_host=os.environ.get("EDGESPLIT_V2_PHONE_HOST", "YOUR_V2_PHONE_IP"),
        phone_port=int(os.environ.get("EDGESPLIT_V2_PHONE_PORT", "8091")),
        benchmark_db=Path(
            os.environ.get("EDGESPLIT_BENCH_DB", str(ROOT / "bench/edgesplit.sqlite3"))
        ),
        model_name=os.environ.get("EDGESPLIT_MODEL_NAME", "Qwen3-0.6B"),
        quant=os.environ.get("EDGESPLIT_QUANT", "Q4_K_M"),
        llama_cpp_commit=os.environ.get(
            "EDGESPLIT_LLAMA_CPP_COMMIT", "e920c523e3b8a0163fe498af5bf90df35ff51d25"
        ),
        runtime_label=os.environ.get(
            "EDGESPLIT_V2_RUNTIME_LABEL",
            "laptop:e920c523e3b8a0163fe498af5bf90df35ff51d25;"
            "phone:e920c523e3b8a0163fe498af5bf90df35ff51d25;patch:edgesplit-v2",
        ),
        timeout_seconds=float(os.environ.get("EDGESPLIT_HTTP_TIMEOUT_SECONDS", "120")),
    )


def create_app(
    orchestrator: V1Orchestrator | None = None,
    v2_orchestrator: V2Orchestrator | None = None,
) -> FastAPI:
    active_orchestrator = orchestrator or V1Orchestrator(settings_from_env())
    active_v2_orchestrator = v2_orchestrator or V2Orchestrator(v2_settings_from_env())
    app = FastAPI(title="EdgeSplit router", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "modes": "v1,v2"}

    @app.post("/v1/generate")
    def generate(body: GenerateBody) -> dict[str, object]:
        try:
            result = active_orchestrator.run(V1Request(**body.model_dump()))
        except V1Error as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "content": result.content,
            "benchmark_row_id": result.benchmark_row_id,
            "state_filename": result.state_filename,
            "state_bytes": result.state_bytes,
            "ttft_seconds": result.ttft_seconds,
            "decode_tokens_per_second": result.decode_tokens_per_second,
            "output_tokens": result.output_tokens,
        }

    @app.post("/v2/generate")
    def generate_v2(body: GenerateBody) -> dict[str, object]:
        try:
            result = active_v2_orchestrator.run(V1Request(**body.model_dump()))
        except V2Error as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "content": result.content,
            "benchmark_row_id": result.benchmark_row_id,
            "sequence_length": result.sequence_length,
            "inner_state_bytes": result.inner_state_bytes,
            "tcp_frame_bytes": result.tcp_frame_bytes,
            "transfer_seconds": result.transfer_seconds,
            "ttft_seconds": result.ttft_seconds,
            "decode_tokens_per_second": result.decode_tokens_per_second,
            "output_tokens": result.output_tokens,
        }

    return app


app = create_app()
