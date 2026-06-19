from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

import anyio
from dotenv import find_dotenv, load_dotenv
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)


dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path=dotenv_path or None)
if os.getenv("HUGGING_FACE_HUB_TOKEN") and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {value!r}") from exc


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


SupportedModel = Literal["mrt2_small", "mrt2_base"]
BackendName = Literal["mlxfn", "mlx", "jax"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]
StorageBackend = Literal["local", "s3"]
GenerationMode = Literal["text-to-audio", "audio-style-transfer", "inpainting"]

SUPPORTED_MODELS: tuple[SupportedModel, ...] = ("mrt2_small", "mrt2_base")
SUPPORTED_BACKENDS: tuple[BackendName, ...] = ("mlxfn", "mlx", "jax")
MODEL_ALIASES: dict[str, SupportedModel] = {
    "small": "mrt2_small",
    "mrt2-small": "mrt2_small",
    "mrt2_small": "mrt2_small",
    "google/magenta-realtime-2:mrt2_small": "mrt2_small",
    "base": "mrt2_base",
    "mrt2-base": "mrt2_base",
    "mrt2_base": "mrt2_base",
    "google/magenta-realtime-2": "mrt2_base",
    "google/magenta-realtime-2:mrt2_base": "mrt2_base",
}
MODEL_DURATION_LIMITS_SECONDS: dict[SupportedModel, float] = {
    "mrt2_small": 60.0,
    "mrt2_base": 60.0,
}
FRAMES_PER_SECOND = 25
SAMPLE_RATE = 48_000


def _normalize_model_name(value: str) -> SupportedModel:
    model_name = MODEL_ALIASES.get(value.strip().lower())
    if model_name is None:
        valid_values = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported model {value!r}. Use one of: {valid_values}.")
    return model_name


def _normalize_backend(value: str | None) -> BackendName:
    backend = (value or "mlxfn").strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        valid_values = ", ".join(SUPPORTED_BACKENDS)
        raise ValueError(f"Unsupported backend {value!r}. Use one of: {valid_values}.")
    return backend  # type: ignore[return-value]


def _env_model_list(name: str, default: str) -> list[SupportedModel]:
    value = os.getenv(name, default)
    return [_normalize_model_name(item) for item in value.split(",") if item.strip()]


DEFAULT_MODEL_NAME = _normalize_model_name(
    os.getenv("MAGENTA_RT_DEFAULT_MODEL", os.getenv("MAGENTA_RT_MODEL", "mrt2_small"))
)
DEFAULT_BACKEND = _normalize_backend(os.getenv("MAGENTA_RT_BACKEND", "mlxfn"))
PRELOAD_MODEL_NAMES = _env_model_list("MAGENTA_RT_PRELOAD_MODELS", DEFAULT_MODEL_NAME)
MAX_DURATION_SECONDS = _env_float("MAGENTA_RT_MAX_DURATION", 60.0)
MAX_BATCH_SIZE = _env_int("MAGENTA_RT_MAX_BATCH_SIZE", 4)
MAX_UPLOAD_BYTES = _env_int("MAGENTA_RT_MAX_UPLOAD_BYTES", 100 * 1024 * 1024)
MLX_BITS = os.getenv("MAGENTA_RT_MLX_BITS", "8").strip()
MLX_BITS_VALUE = None if MLX_BITS.lower() in {"", "none", "false"} else int(MLX_BITS)
MLX_QUANTIZE_GROUP_SIZE = (
    None
    if not os.getenv("MAGENTA_RT_MLX_QUANTIZE_GROUP_SIZE")
    else _env_int("MAGENTA_RT_MLX_QUANTIZE_GROUP_SIZE", 64)
)
MODEL_CHECKPOINT = os.getenv("MAGENTA_RT_CHECKPOINT") or None
MLXFN_WARMUP_STEPS = _env_int("MAGENTA_RT_MLXFN_WARMUP_STEPS", 5)
AUTO_DOWNLOAD = _env_bool("MAGENTA_RT_AUTO_DOWNLOAD", False)
OUTPUT_DIR = Path(os.getenv("MAGENTA_RT_OUTPUT_DIR", "outputs"))
UI_DIST_DIR = Path(
    os.getenv(
        "MAGENTA_RT_UI_DIST_DIR",
        str(Path(__file__).resolve().parents[2] / "ui" / "dist"),
    )
)
STORAGE_BUCKET = (
    os.getenv("MAGENTA_RT_STORAGE_BUCKET")
    or os.getenv("AWS_S3_BUCKET")
    or os.getenv("R2_BUCKET")
    or os.getenv("R2_BUCKET_NAME")
)
STORAGE_PREFIX = os.getenv("MAGENTA_RT_STORAGE_PREFIX", "magenta-rt/jobs").strip("/")
STORAGE_ENDPOINT_URL = (
    os.getenv("MAGENTA_RT_STORAGE_ENDPOINT_URL")
    or os.getenv("AWS_ENDPOINT_URL_S3")
    or os.getenv("R2_ENDPOINT_URL")
)
STORAGE_REGION = (
    os.getenv("MAGENTA_RT_STORAGE_REGION")
    or os.getenv("AWS_REGION")
    or os.getenv("AWS_DEFAULT_REGION")
    or "us-east-1"
)
STORAGE_PUBLIC_BASE_URL = os.getenv("MAGENTA_RT_STORAGE_PUBLIC_BASE_URL")
PRESIGNED_URL_EXPIRES = _env_int("MAGENTA_RT_PRESIGNED_URL_EXPIRES", 3600)

Duration = Annotated[
    float,
    Field(
        gt=0,
        le=MAX_DURATION_SECONDS,
        description="Generated audio duration in seconds.",
    ),
]


class GenerateAudioRequest(BaseModel):
    model: SupportedModel = Field(
        default=DEFAULT_MODEL_NAME,
        description="Magenta RT 2 model to use: mrt2_small or mrt2_base.",
    )
    backend: BackendName | None = Field(
        default=None,
        description="Optional backend override: mlxfn, mlx, or jax.",
    )
    prompt: str = Field(..., min_length=1, description="Text prompt describing the music.")
    duration: Duration = 4.0
    temperature: float = Field(default=1.3, gt=0.0, le=5.0)
    top_k: int = Field(default=40, ge=0, le=1024)
    cfg_musiccoca: float = Field(default=3.0, ge=-1.0, le=7.0)
    cfg_notes: float = Field(default=1.0, ge=-1.0, le=7.0)
    cfg_drums: float = Field(default=1.0, ge=-1.0, le=7.0)
    batch_size: int = Field(
        default=1,
        ge=1,
        le=MAX_BATCH_SIZE,
        description="Number of clips to generate. Batch outputs are returned as a ZIP.",
    )
    seed: int = Field(default=0, ge=0, description="Seed used by the MusicCoCa text mapper.")
    audio_style_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="For uploaded audio prompts, 0 keeps text style and 1 favors source-audio style.",
    )

    @field_validator("model", mode="before")
    @classmethod
    def normalize_model(cls, value: str) -> SupportedModel:
        if not isinstance(value, str):
            raise ValueError("model must be a string")
        return _normalize_model_name(value)

    @field_validator("backend", mode="before")
    @classmethod
    def normalize_backend(cls, value: str | None) -> BackendName | None:
        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise ValueError("backend must be a string")
        return _normalize_backend(value)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        prompt = value.strip()
        if not prompt:
            raise ValueError("prompt must not be blank")
        return prompt

    @model_validator(mode="after")
    def duration_must_fit_model(self) -> GenerateAudioRequest:
        model_limit = min(MAX_DURATION_SECONDS, MODEL_DURATION_LIMITS_SECONDS[self.model])
        if self.duration > model_limit:
            raise ValueError(
                f"duration must be <= {model_limit:g}s for model {self.model!r}"
            )
        return self


class RealtimeAudioRequest(GenerateAudioRequest):
    batch_size: int = Field(
        default=1,
        ge=1,
        le=1,
        description="Realtime streams generate one clip per connection.",
    )
    chunk_frames: int = Field(
        default=1,
        ge=1,
        le=25,
        description="Model frames per WebSocket audio chunk. 1 frame is 40 ms.",
    )
    audio_format: Literal["f32le"] = Field(
        default="f32le",
        description="Binary chunk format: interleaved stereo little-endian float32.",
    )


class ModelAssetsResponse(BaseModel):
    magenta_home: str | None
    resources_ready: bool
    models: dict[str, bool]
    details: dict[str, str]


class HealthResponse(BaseModel):
    status: str
    model: str
    backend: str
    loaded: bool
    storage_backend: str
    available_models: list[str]
    available_backends: list[str]
    loaded_models: list[str]
    preload_models: list[str]
    model_duration_limits_seconds: dict[str, float]
    max_duration_seconds: float
    max_batch_size: int
    sample_rate: int
    frames_per_second: int
    assets: ModelAssetsResponse
    load_error: str | None = None


class CreateJobResponse(BaseModel):
    id: str
    status: JobStatus
    status_url: str


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    mode: GenerationMode
    model: str
    backend: str
    duration: float
    frames: int
    output_count: int | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    download_url: str | None = None
    download_content_type: str | None = None
    error: str | None = None
    storage_backend: StorageBackend | None = None
    storage_key: str | None = None
    sample_rate: int | None = None


@dataclass(frozen=True)
class AudioInput:
    sample_rate: int
    samples: object


@dataclass(frozen=True)
class RuntimeGenerationRequest:
    mode: GenerationMode
    controls: GenerateAudioRequest
    source_audio: AudioInput | None = None


@dataclass(frozen=True)
class GenerationResult:
    waveforms: list[object]
    model_name: SupportedModel
    backend: BackendName
    sample_rate: int


@dataclass(frozen=True)
class AudioArtifact:
    content: bytes
    content_type: str
    filename: str
    output_count: int


@dataclass(frozen=True)
class StoredAudio:
    backend: StorageBackend
    key: str
    content_type: str
    filename: str
    output_count: int
    local_path: Path | None = None


@dataclass
class JobRecord:
    id: str
    request: RuntimeGenerationRequest
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    stored_audio: StoredAudio | None = None
    sample_rate: int | None = None
    error: str | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _frames_for_duration(duration: float) -> int:
    return max(1, round(duration * FRAMES_PER_SECOND))


class AudioStorage:
    def __init__(self) -> None:
        self.backend: StorageBackend = "s3" if STORAGE_BUCKET else "local"
        self._s3_client = None

    def save_artifact(self, job_id: str, artifact: AudioArtifact) -> StoredAudio:
        if self.backend == "s3":
            return self._save_s3(job_id, artifact)
        return self._save_local(job_id, artifact)

    def download_url(self, stored_audio: StoredAudio, request: Request) -> str:
        if stored_audio.backend == "s3":
            return self._s3_download_url(stored_audio.key)
        return str(request.url_for("download_job_audio", job_id=stored_audio.key))

    def _save_local(self, job_id: str, artifact: AudioArtifact) -> StoredAudio:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"{job_id}-{artifact.filename}"
        path.write_bytes(artifact.content)
        return StoredAudio(
            backend="local",
            key=job_id,
            content_type=artifact.content_type,
            filename=path.name,
            output_count=artifact.output_count,
            local_path=path,
        )

    def _save_s3(self, job_id: str, artifact: AudioArtifact) -> StoredAudio:
        if STORAGE_BUCKET is None:
            raise RuntimeError("MAGENTA_RT_STORAGE_BUCKET is required for S3 storage.")

        key_parts = [part for part in (STORAGE_PREFIX, f"{job_id}-{artifact.filename}") if part]
        key = "/".join(key_parts)
        self._s3().put_object(
            Bucket=STORAGE_BUCKET,
            Key=key,
            Body=artifact.content,
            ContentType=artifact.content_type,
        )
        return StoredAudio(
            backend="s3",
            key=key,
            content_type=artifact.content_type,
            filename=artifact.filename,
            output_count=artifact.output_count,
        )

    def _s3_download_url(self, key: str) -> str:
        if STORAGE_PUBLIC_BASE_URL:
            return f"{STORAGE_PUBLIC_BASE_URL.rstrip('/')}/{key}"

        if STORAGE_BUCKET is None:
            raise RuntimeError("MAGENTA_RT_STORAGE_BUCKET is required for S3 storage.")

        return self._s3().generate_presigned_url(
            "get_object",
            Params={"Bucket": STORAGE_BUCKET, "Key": key},
            ExpiresIn=PRESIGNED_URL_EXPIRES,
        )

    def _s3(self):
        if self._s3_client is not None:
            return self._s3_client

        import boto3
        from botocore.config import Config

        access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("R2_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("R2_SECRET_ACCESS_KEY")

        kwargs = {
            "service_name": "s3",
            "region_name": STORAGE_REGION,
            "endpoint_url": STORAGE_ENDPOINT_URL,
            "config": Config(signature_version="s3v4"),
        }
        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key

        self._s3_client = boto3.client(**kwargs)
        return self._s3_client


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, generation_request: RuntimeGenerationRequest) -> JobRecord:
        job = JobRecord(
            id=uuid4().hex,
            request=generation_request,
            status="queued",
            created_at=_utc_now(),
        )
        async with self._lock:
            self._jobs[job.id] = job
        return job

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def mark_running(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = "running"
            job.started_at = _utc_now()
            return job

    async def mark_succeeded(
        self,
        job_id: str,
        stored_audio: StoredAudio,
        sample_rate: int,
    ) -> None:
        async with self._lock:
            job = self._jobs[job_id]
            job.status = "succeeded"
            job.completed_at = _utc_now()
            job.stored_audio = stored_audio
            job.sample_rate = sample_rate

    async def mark_failed(self, job_id: str, error: str) -> None:
        async with self._lock:
            job = self._jobs[job_id]
            job.status = "failed"
            job.completed_at = _utc_now()
            job.error = error


class ModelRuntime:
    def __init__(self) -> None:
        self.models: dict[tuple[BackendName, SupportedModel], object] = {}
        self.lock = asyncio.Lock()
        self.load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return bool(self.models)

    @property
    def loaded_models(self) -> list[str]:
        return [f"{backend}:{model}" for backend, model in sorted(self.models)]

    def load_model(
        self,
        model_name: SupportedModel,
        backend: BackendName = DEFAULT_BACKEND,
        controls: GenerateAudioRequest | None = None,
    ) -> object:
        key = (backend, model_name)
        if key in self.models:
            return self.models[key]

        logger.info("Loading Magenta RT 2 model %s with %s backend", model_name, backend)

        try:
            model = self._create_model(model_name, backend, controls)
        except Exception as exc:
            self.load_error = str(exc)
            raise

        self.models[key] = model
        self.load_error = None
        logger.info("Loaded Magenta RT 2 model %s with %s backend", model_name, backend)
        return model

    def _create_model(
        self,
        model_name: SupportedModel,
        backend: BackendName,
        controls: GenerateAudioRequest | None,
    ) -> object:
        temperature = controls.temperature if controls else 1.3
        top_k = controls.top_k if controls else 40
        cfg_musiccoca = controls.cfg_musiccoca if controls else 3.0
        cfg_notes = controls.cfg_notes if controls else 1.0
        cfg_drums = controls.cfg_drums if controls else 1.0

        if AUTO_DOWNLOAD:
            self._download_missing_assets(model_name, backend)

        if backend == "mlxfn":
            from magenta_rt import MagentaRT2Mlxfn

            return MagentaRT2Mlxfn(
                size=model_name,
                temperature=temperature,
                top_k=top_k,
                cfg_musiccoca=cfg_musiccoca,
                cfg_notes=cfg_notes,
                cfg_drums=cfg_drums,
                warmup_steps=MLXFN_WARMUP_STEPS,
            )

        if backend == "mlx":
            from magenta_rt import MagentaRT2Mlx

            return MagentaRT2Mlx(
                size=model_name,
                checkpoint=MODEL_CHECKPOINT,
                temperature=temperature,
                top_k=top_k,
                cfg_musiccoca=cfg_musiccoca,
                cfg_notes=cfg_notes,
                cfg_drums=cfg_drums,
                bits=MLX_BITS_VALUE,
                quantize_group_size=MLX_QUANTIZE_GROUP_SIZE,
            )

        from magenta_rt import MagentaRT2Jax

        return MagentaRT2Jax(
            size=model_name,
            checkpoint=MODEL_CHECKPOINT,
            temperature=temperature,
            top_k=top_k,
            cfg_musiccoca=cfg_musiccoca,
            cfg_notes=cfg_notes,
            cfg_drums=cfg_drums,
        )

    def _download_missing_assets(self, model_name: SupportedModel, backend: BackendName) -> None:
        import subprocess
        import sys

        def run_mrt(*args: str) -> None:
            executable = shutil.which("mrt")
            if executable:
                subprocess.run([executable, *args], check=True)
                return

            runner = (
                "import sys; "
                "from magenta_rt.cli import main; "
                "sys.argv = ['mrt', *sys.argv[1:]]; "
                "main()"
            )
            subprocess.run([sys.executable, "-c", runner, *args], check=True)

        assets = model_assets()
        if not assets.resources_ready:
            run_mrt("models", "init")
        if backend == "mlxfn" and not assets.models.get(model_name, False):
            run_mrt("models", "download", model_name)
        if backend in {"mlx", "jax"}:
            checkpoint_name = MODEL_CHECKPOINT or f"{model_name}.safetensors"
            if checkpoint_name not in assets.details.get("checkpoints", ""):
                run_mrt("checkpoints", "download", checkpoint_name)

    def load_preconfigured_models(self) -> None:
        for model_name in PRELOAD_MODEL_NAMES:
            try:
                self.load_model(model_name, DEFAULT_BACKEND)
            except Exception as exc:
                logger.warning("Could not preload %s: %s", model_name, exc)
                self.load_error = str(exc)

    def generate(self, request: RuntimeGenerationRequest) -> GenerationResult:
        controls = request.controls
        backend = controls.backend or DEFAULT_BACKEND
        model = self.load_model(controls.model, backend, controls)
        waveforms = []

        for index in range(controls.batch_size):
            style = self._style_embedding(model, request, seed=controls.seed + index)
            waveform, _state = model.generate(
                style=style,
                cfg_musiccoca=controls.cfg_musiccoca,
                cfg_notes=controls.cfg_notes,
                cfg_drums=controls.cfg_drums,
                temperature=controls.temperature,
                top_k=controls.top_k,
                frames=_frames_for_duration(controls.duration),
            )
            waveforms.append(waveform)

        return GenerationResult(
            waveforms=waveforms,
            model_name=controls.model,
            backend=backend,
            sample_rate=SAMPLE_RATE,
        )

    def _style_embedding(
        self,
        model: object,
        request: RuntimeGenerationRequest,
        seed: int,
    ) -> object:
        controls = request.controls
        text_embedding = model.embed_style(
            controls.prompt,
            use_mapper=True,
            seed=seed,
        )
        if request.source_audio is None:
            return text_embedding

        import numpy as np
        from magenta_rt.audio import Waveform

        source_waveform = Waveform(request.source_audio.samples, request.source_audio.sample_rate)
        audio_embedding = model.embed_style(source_waveform, pool_across_time=True)
        text_weight = 1.0 - controls.audio_style_weight
        return np.asarray(text_embedding) * text_weight + np.asarray(audio_embedding) * (
            controls.audio_style_weight
        )


runtime = ModelRuntime()
jobs = JobStore()
audio_storage = AudioStorage()


def model_assets() -> ModelAssetsResponse:
    try:
        from magenta_rt import paths

        magenta_home = paths.magenta_home()
        resources_dir = paths.resources_dir()
        musiccoca_dir = paths.musiccoca_dir()
        spectrostream_dir = paths.spectrostream_dir()
        checkpoints_dir = paths.checkpoints_dir()
        models_dir = paths.models_dir()
    except Exception as exc:
        return ModelAssetsResponse(
            magenta_home=None,
            resources_ready=False,
            models={model: False for model in SUPPORTED_MODELS},
            details={"error": str(exc)},
        )

    musiccoca_files = [
        "audio_preprocessor.tflite",
        "music_encoder.tflite",
        "pretrained_vector_quantizer.tflite",
        "spm.model",
        "text_encoder.tflite",
    ]
    spectrostream_files = [
        "decoder.safetensors",
        "encoder.safetensors",
        "quantizer.safetensors",
    ]
    resources_ready = all((musiccoca_dir / item).exists() for item in musiccoca_files) and all(
        (spectrostream_dir / item).exists() for item in spectrostream_files
    )
    models = {
        model: (models_dir / model / f"{model}.mlxfn").exists()
        and (models_dir / model / f"{model}_state.safetensors").exists()
        for model in SUPPORTED_MODELS
    }
    checkpoint_names = []
    if checkpoints_dir.exists():
        checkpoint_names = sorted(path.name for path in checkpoints_dir.glob("*.safetensors"))

    return ModelAssetsResponse(
        magenta_home=str(magenta_home),
        resources_ready=resources_ready,
        models=models,
        details={
            "resources_dir": str(resources_dir),
            "models_dir": str(models_dir),
            "checkpoints_dir": str(checkpoints_dir),
            "checkpoints": ", ".join(checkpoint_names),
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    await anyio.to_thread.run_sync(runtime.load_preconfigured_models)
    app.state.runtime = runtime
    yield


app = FastAPI(
    title="Magenta RT 2 API",
    summary="Generate music with Google's Magenta RealTime 2.",
    version="0.1.0",
    lifespan=lifespan,
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("MAGENTA_RT_CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _waveform_to_wav_bytes(waveform: object) -> bytes:
    buffer = io.BytesIO()
    waveform.write(buffer, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _waveform_to_f32le_bytes(waveform: object) -> tuple[bytes, int, int]:
    import numpy as np

    samples = np.asarray(waveform.samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2:
        raise ValueError(f"Expected waveform samples [samples, channels], got {samples.shape}")
    if samples.shape[1] == 1:
        samples = np.repeat(samples, 2, axis=1)
    if samples.shape[1] != 2:
        raise ValueError(f"Realtime streaming expects mono/stereo audio, got {samples.shape[1]}")
    samples = np.ascontiguousarray(samples)
    return samples.tobytes(), int(samples.shape[0]), int(samples.shape[1])


def _generation_result_to_artifact(result: GenerationResult) -> AudioArtifact:
    wav_outputs = [_waveform_to_wav_bytes(waveform) for waveform in result.waveforms]
    if len(wav_outputs) == 1:
        return AudioArtifact(
            content=wav_outputs[0],
            content_type="audio/wav",
            filename=f"magenta-rt-2-{result.model_name}.wav",
            output_count=1,
        )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, wav_bytes in enumerate(wav_outputs, start=1):
            archive.writestr(
                f"magenta-rt-2-{result.model_name}-{index:02d}.wav",
                wav_bytes,
            )

    return AudioArtifact(
        content=buffer.getvalue(),
        content_type="application/zip",
        filename=f"magenta-rt-2-{result.model_name}-batch.zip",
        output_count=len(wav_outputs),
    )


def _job_response(job: JobRecord, request: Request) -> JobResponse:
    download_url = None
    download_content_type = None
    storage_backend = None
    storage_key = None

    if job.stored_audio is not None:
        download_url = audio_storage.download_url(job.stored_audio, request)
        download_content_type = job.stored_audio.content_type
        storage_backend = job.stored_audio.backend
        storage_key = job.stored_audio.key

    controls = job.request.controls
    return JobResponse(
        id=job.id,
        status=job.status,
        mode=job.request.mode,
        model=controls.model,
        backend=controls.backend or DEFAULT_BACKEND,
        duration=controls.duration,
        frames=_frames_for_duration(controls.duration),
        output_count=job.stored_audio.output_count if job.stored_audio else None,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        download_url=download_url,
        download_content_type=download_content_type,
        error=job.error,
        storage_backend=storage_backend,
        storage_key=storage_key,
        sample_rate=job.sample_rate,
    )


def _text_runtime_request(request: GenerateAudioRequest) -> RuntimeGenerationRequest:
    return RuntimeGenerationRequest(mode="text-to-audio", controls=request)


def _artifact_response(artifact: AudioArtifact, result: GenerationResult) -> Response:
    return Response(
        content=artifact.content,
        media_type=artifact.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{artifact.filename}"',
            "X-Model": result.model_name,
            "X-Backend": result.backend,
            "X-Sample-Rate": str(result.sample_rate),
            "X-Output-Count": str(artifact.output_count),
        },
    )


async def _generate_artifact(
    runtime_request: RuntimeGenerationRequest,
) -> tuple[GenerationResult, AudioArtifact]:
    async with runtime.lock:
        try:
            result = await anyio.to_thread.run_sync(runtime.generate, runtime_request)
            artifact = _generation_result_to_artifact(result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Magenta RT 2 assets are missing: {exc}",
            ) from exc
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Magenta RT 2 backend dependency is missing: {exc}",
            ) from exc
        except Exception as exc:
            if _looks_like_missing_magenta_asset(exc):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Magenta RT 2 assets are missing or incomplete. "
                        "Run `uv run mrt models init` and "
                        f"`uv run mrt models download {runtime_request.controls.model}`."
                    ),
                ) from exc
            logger.exception("Audio generation failed.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result, artifact


def _looks_like_missing_magenta_asset(exc: Exception) -> bool:
    message = str(exc).lower()
    missing_markers = (
        "failed to open",
        "not found",
        "no such file",
        "missing",
        "could not open",
    )
    return any(marker in message for marker in missing_markers) and (
        "magenta-rt-v2" in message
        or "musiccoca" in message
        or "spectrostream" in message
        or ".mlxfn" in message
        or ".safetensors" in message
        or ".tflite" in message
    )


async def _load_uploaded_audio(audio: UploadFile) -> AudioInput:
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded audio file exceeds {MAX_UPLOAD_BYTES} bytes.",
        )

    try:
        import soundfile as sf

        samples, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode uploaded audio: {exc}",
        ) from exc

    return AudioInput(sample_rate=int(sample_rate), samples=samples)


def _build_generation_request_from_form(
    *,
    model: str,
    backend: str | None,
    prompt: str,
    duration: float,
    temperature: float,
    top_k: int,
    cfg_musiccoca: float,
    cfg_notes: float,
    cfg_drums: float,
    batch_size: int,
    seed: int,
    audio_style_weight: float,
) -> GenerateAudioRequest:
    try:
        return GenerateAudioRequest(
            model=model,
            backend=backend,
            prompt=prompt,
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            cfg_musiccoca=cfg_musiccoca,
            cfg_notes=cfg_notes,
            cfg_drums=cfg_drums,
            batch_size=batch_size,
            seed=seed,
            audio_style_weight=audio_style_weight,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


async def _run_generation_job(job_id: str) -> None:
    job = await jobs.mark_running(job_id)
    if job is None:
        logger.error("Cannot run missing job %s", job_id)
        return

    try:
        async with runtime.lock:
            result = await anyio.to_thread.run_sync(runtime.generate, job.request)
            artifact = _generation_result_to_artifact(result)

        stored_audio = await anyio.to_thread.run_sync(
            audio_storage.save_artifact,
            job.id,
            artifact,
        )
        await jobs.mark_succeeded(job.id, stored_audio, result.sample_rate)
    except Exception as exc:
        logger.exception("Job %s failed.", job_id)
        await jobs.mark_failed(job_id, str(exc))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    effective_duration_limits = {
        model_name: min(MAX_DURATION_SECONDS, model_limit)
        for model_name, model_limit in MODEL_DURATION_LIMITS_SECONDS.items()
    }
    status_value = "ok" if runtime.load_error is None else "error"
    return HealthResponse(
        status=status_value,
        model=DEFAULT_MODEL_NAME,
        backend=DEFAULT_BACKEND,
        loaded=runtime.loaded,
        storage_backend=audio_storage.backend,
        available_models=list(SUPPORTED_MODELS),
        available_backends=list(SUPPORTED_BACKENDS),
        loaded_models=runtime.loaded_models,
        preload_models=PRELOAD_MODEL_NAMES,
        model_duration_limits_seconds=effective_duration_limits,
        max_duration_seconds=MAX_DURATION_SECONDS,
        max_batch_size=MAX_BATCH_SIZE,
        sample_rate=SAMPLE_RATE,
        frames_per_second=FRAMES_PER_SECOND,
        assets=model_assets(),
        load_error=runtime.load_error,
    )


@app.post("/jobs", response_model=CreateJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    generation_request: GenerateAudioRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> CreateJobResponse:
    job = await jobs.create(_text_runtime_request(generation_request))
    background_tasks.add_task(_run_generation_job, job.id)
    return CreateJobResponse(
        id=job.id,
        status=job.status,
        status_url=str(request.url_for("get_job", job_id=job.id)),
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, request: Request) -> JobResponse:
    job = await jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_response(job, request)


@app.get("/jobs/{job_id}/audio", include_in_schema=False)
async def download_job_audio(job_id: str) -> FileResponse:
    job = await jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "succeeded" or job.stored_audio is None:
        raise HTTPException(status_code=409, detail=f"Job is {job.status}.")
    if job.stored_audio.backend != "local" or job.stored_audio.local_path is None:
        raise HTTPException(status_code=404, detail="Local audio file is not available.")
    if not job.stored_audio.local_path.exists():
        raise HTTPException(status_code=404, detail="Local audio file is missing.")

    return FileResponse(
        job.stored_audio.local_path,
        media_type=job.stored_audio.content_type,
        filename=job.stored_audio.filename,
    )


@app.post(
    "/v1/audio/generations",
    responses={
        200: {
            "content": {"audio/wav": {}, "application/zip": {}},
            "description": "Generated WAV audio, or a ZIP of WAVs when batch_size > 1.",
        }
    },
)
async def generate_audio(request: GenerateAudioRequest) -> Response:
    result, artifact = await _generate_artifact(_text_runtime_request(request))
    return _artifact_response(artifact, result)


@app.post(
    "/v1/audio/variations",
    responses={
        200: {
            "content": {"audio/wav": {}, "application/zip": {}},
            "description": "Generated music using a text prompt and uploaded audio as style.",
        }
    },
)
async def generate_audio_variation(
    audio: Annotated[UploadFile, File(description="Source audio style file.")],
    prompt: Annotated[str, Form(min_length=1)],
    model: Annotated[str, Form()] = DEFAULT_MODEL_NAME,
    backend: Annotated[str | None, Form()] = None,
    duration: Annotated[float, Form(gt=0, le=MAX_DURATION_SECONDS)] = 4.0,
    temperature: Annotated[float, Form(gt=0.0, le=5.0)] = 1.3,
    top_k: Annotated[int, Form(ge=0, le=1024)] = 40,
    cfg_musiccoca: Annotated[float, Form(ge=-1.0, le=7.0)] = 3.0,
    cfg_notes: Annotated[float, Form(ge=-1.0, le=7.0)] = 1.0,
    cfg_drums: Annotated[float, Form(ge=-1.0, le=7.0)] = 1.0,
    batch_size: Annotated[int, Form(ge=1, le=MAX_BATCH_SIZE)] = 1,
    seed: Annotated[int, Form(ge=0)] = 0,
    audio_style_weight: Annotated[float, Form(ge=0.0, le=1.0)] = 0.5,
) -> Response:
    controls = _build_generation_request_from_form(
        model=model,
        backend=backend,
        prompt=prompt,
        duration=duration,
        temperature=temperature,
        top_k=top_k,
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
        cfg_drums=cfg_drums,
        batch_size=batch_size,
        seed=seed,
        audio_style_weight=audio_style_weight,
    )
    source_audio = await _load_uploaded_audio(audio)
    runtime_request = RuntimeGenerationRequest(
        mode="audio-style-transfer",
        controls=controls,
        source_audio=source_audio,
    )
    result, artifact = await _generate_artifact(runtime_request)
    return _artifact_response(artifact, result)


@app.post("/jobs/variations", response_model=CreateJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_variation_job(
    background_tasks: BackgroundTasks,
    request: Request,
    audio: Annotated[UploadFile, File(description="Source audio style file.")],
    prompt: Annotated[str, Form(min_length=1)],
    model: Annotated[str, Form()] = DEFAULT_MODEL_NAME,
    backend: Annotated[str | None, Form()] = None,
    duration: Annotated[float, Form(gt=0, le=MAX_DURATION_SECONDS)] = 4.0,
    temperature: Annotated[float, Form(gt=0.0, le=5.0)] = 1.3,
    top_k: Annotated[int, Form(ge=0, le=1024)] = 40,
    cfg_musiccoca: Annotated[float, Form(ge=-1.0, le=7.0)] = 3.0,
    cfg_notes: Annotated[float, Form(ge=-1.0, le=7.0)] = 1.0,
    cfg_drums: Annotated[float, Form(ge=-1.0, le=7.0)] = 1.0,
    batch_size: Annotated[int, Form(ge=1, le=MAX_BATCH_SIZE)] = 1,
    seed: Annotated[int, Form(ge=0)] = 0,
    audio_style_weight: Annotated[float, Form(ge=0.0, le=1.0)] = 0.5,
) -> CreateJobResponse:
    controls = _build_generation_request_from_form(
        model=model,
        backend=backend,
        prompt=prompt,
        duration=duration,
        temperature=temperature,
        top_k=top_k,
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
        cfg_drums=cfg_drums,
        batch_size=batch_size,
        seed=seed,
        audio_style_weight=audio_style_weight,
    )
    source_audio = await _load_uploaded_audio(audio)
    job = await jobs.create(
        RuntimeGenerationRequest(
            mode="audio-style-transfer",
            controls=controls,
            source_audio=source_audio,
        )
    )
    background_tasks.add_task(_run_generation_job, job.id)
    return CreateJobResponse(
        id=job.id,
        status=job.status,
        status_url=str(request.url_for("get_job", job_id=job.id)),
    )


@app.post("/v1/audio/inpaint")
async def generate_audio_inpaint() -> Response:
    raise HTTPException(
        status_code=501,
        detail=(
            "Magenta RT 2's public Python API exposes text/audio style conditioning, "
            "but not Stable Audio-style inpainting."
        ),
    )


@app.post("/jobs/inpaint", response_model=CreateJobResponse)
async def create_inpaint_job() -> CreateJobResponse:
    raise HTTPException(
        status_code=501,
        detail=(
            "Magenta RT 2's public Python API exposes text/audio style conditioning, "
            "but not Stable Audio-style inpainting."
        ),
    )


@app.websocket("/v1/audio/realtime")
async def stream_realtime_audio(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        start_message = await websocket.receive_json()
        if not isinstance(start_message, dict):
            await websocket.send_json({"type": "error", "detail": "Start message must be a JSON object."})
            await websocket.close(code=1003)
            return

        message_type = start_message.pop("type", "start")
        if message_type != "start":
            await websocket.send_json({"type": "error", "detail": "First message must have type 'start'."})
            await websocket.close(code=1003)
            return

        try:
            controls = RealtimeAudioRequest.model_validate(start_message)
        except ValidationError as exc:
            await websocket.send_json({"type": "error", "detail": exc.errors()})
            await websocket.close(code=1008)
            return

        runtime_request = RuntimeGenerationRequest(mode="text-to-audio", controls=controls)
        backend = controls.backend or DEFAULT_BACKEND
        total_frames = _frames_for_duration(controls.duration)
        chunk_frames = min(controls.chunk_frames, total_frames)

        await websocket.send_json(
            {
                "type": "ready",
                "model": controls.model,
                "backend": backend,
                "sample_rate": SAMPLE_RATE,
                "channels": 2,
                "audio_format": controls.audio_format,
                "model_frame_samples": SAMPLE_RATE // FRAMES_PER_SECOND,
                "frames_per_second": FRAMES_PER_SECOND,
                "chunk_frames": chunk_frames,
                "chunk_duration_seconds": chunk_frames / FRAMES_PER_SECOND,
                "total_frames": total_frames,
                "duration": controls.duration,
            }
        )

        async with runtime.lock:
            try:
                model = await anyio.to_thread.run_sync(
                    runtime.load_model,
                    controls.model,
                    backend,
                    controls,
                )
                style = await anyio.to_thread.run_sync(
                    runtime._style_embedding,
                    model,
                    runtime_request,
                    controls.seed,
                )
            except Exception as exc:
                await websocket.send_json(
                    {"type": "error", "detail": _realtime_exception_detail(exc, controls.model)}
                )
                await websocket.close(code=1011)
                return

        state = None
        sent_frames = 0
        chunk_index = 0

        while sent_frames < total_frames:
            frames_this_chunk = min(chunk_frames, total_frames - sent_frames)

            def generate_chunk(current_state):
                return model.generate(
                    style=style,
                    cfg_musiccoca=controls.cfg_musiccoca,
                    cfg_notes=controls.cfg_notes,
                    cfg_drums=controls.cfg_drums,
                    temperature=controls.temperature,
                    top_k=controls.top_k,
                    frames=frames_this_chunk,
                    state=current_state,
                )

            async with runtime.lock:
                try:
                    waveform, state = await anyio.to_thread.run_sync(generate_chunk, state)
                    payload, sample_count, channels = _waveform_to_f32le_bytes(waveform)
                except Exception as exc:
                    await websocket.send_json(
                        {"type": "error", "detail": _realtime_exception_detail(exc, controls.model)}
                    )
                    await websocket.close(code=1011)
                    return

            await websocket.send_json(
                {
                    "type": "chunk",
                    "index": chunk_index,
                    "model_frames": frames_this_chunk,
                    "samples": sample_count,
                    "channels": channels,
                    "start_seconds": sent_frames / FRAMES_PER_SECOND,
                    "duration_seconds": frames_this_chunk / FRAMES_PER_SECOND,
                    "byte_length": len(payload),
                }
            )
            await websocket.send_bytes(payload)

            sent_frames += frames_this_chunk
            chunk_index += 1

        await websocket.send_json(
            {
                "type": "done",
                "chunks": chunk_index,
                "model_frames": sent_frames,
                "duration_seconds": sent_frames / FRAMES_PER_SECOND,
            }
        )
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        logger.info("Realtime audio WebSocket disconnected.")


def _realtime_exception_detail(exc: Exception, model_name: SupportedModel) -> str:
    if _looks_like_missing_magenta_asset(exc):
        return (
            "Magenta RT 2 assets are missing or incomplete. "
            "Run `uv run mrt models init` and "
            f"`uv run mrt models download {model_name}`."
        )
    return str(exc)


@app.post("/generate", include_in_schema=False)
async def generate_alias(request: GenerateAudioRequest) -> Response:
    return await generate_audio(request)


if UI_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=UI_DIST_DIR, html=True), name="dashboard")
