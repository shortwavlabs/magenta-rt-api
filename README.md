# Magenta RT 2 API

FastAPI server and React dashboard for Google's Magenta RealTime 2 (`magenta-rt`).

## Setup

```bash
uv sync
```

Magenta RT 2 stores model resources under `~/Documents/Magenta/magenta-rt-v2` by default. Download the shared MusicCoCa/SpectroStream resources and at least one exported MLX model:

```bash
uv run mrt models init
uv run mrt models download mrt2_small
```

`mrt2_base` is higher quality but much heavier:

```bash
uv run mrt models download mrt2_base
```

Start the API:

```bash
uv run --env-file .env magenta-rt-api --host 0.0.0.0 --port 8000
```

## Dashboard

```bash
cd ui
pnpm install
pnpm dev
```

The Vite dashboard proxies `/api` to `http://localhost:8000` by default.

To build static assets that the FastAPI app can serve:

```bash
cd ui
pnpm build
cd ..
uv run magenta-rt-api --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`.

## Generate Audio

```bash
curl -X POST http://localhost:8000/v1/audio/generations \
  -H "Content-Type: application/json" \
  --output magenta.wav \
  -d '{
    "model": "mrt2_small",
    "backend": "mlxfn",
    "prompt": "disco funk bassline with crisp drums",
    "duration": 4,
    "temperature": 1.3,
    "top_k": 40,
    "cfg_musiccoca": 3.0,
    "cfg_notes": 1.0,
    "cfg_drums": 1.0,
    "batch_size": 1,
    "seed": 0
  }'
```

When `batch_size` is greater than `1`, synchronous generation returns an `application/zip` containing one WAV per clip.

## Audio Style Transfer

Magenta RT 2 can embed audio with MusicCoCa. The variation endpoint blends the text prompt style and uploaded source-audio style, then generates a fresh clip:

```bash
curl -X POST http://localhost:8000/v1/audio/variations \
  -F audio=@loop.wav \
  -F model=mrt2_small \
  -F prompt="sunlit house groove with soft piano chords" \
  -F duration=6 \
  -F audio_style_weight=0.65 \
  --output variation.wav
```

The async version is `POST /jobs/variations`.

Stable Audio-style inpainting routes are present for dashboard/API compatibility, but return `501` because Magenta RT 2's public Python API does not expose that inpainting surface.

## Async Jobs

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mrt2_small",
    "prompt": "minimal techno pulse, dry drums, warm bass",
    "duration": 5
  }'
```

Poll `GET /jobs/{id}` until `status` is `succeeded`. Without object storage configured, job outputs are written under `outputs/` and served by the local API.

## Endpoints

- `GET /health`
- `POST /jobs`
- `POST /jobs/variations`
- `POST /jobs/inpaint` compatibility endpoint, returns `501`
- `GET /jobs/{id}`
- `POST /v1/audio/generations`
- `POST /v1/audio/variations`
- `POST /v1/audio/inpaint` compatibility endpoint, returns `501`
- `POST /generate` alias for text generation

## Configuration

| Environment variable | Default | Description |
| --- | --- | --- |
| `HF_TOKEN` | unset | Hugging Face token if your environment needs authenticated model downloads. |
| `MAGENTA_HOME` | `~/Documents/Magenta` | Upstream Magenta asset root. Assets are stored in `$MAGENTA_HOME/magenta-rt-v2`. |
| `MAGENTA_RT_DEFAULT_MODEL` | `mrt2_small` | Default request model. |
| `MAGENTA_RT_BACKEND` | `mlxfn` | Backend: `mlxfn`, `mlx`, or `jax`. |
| `MAGENTA_RT_PRELOAD_MODELS` | default model | Comma-separated models to preload at startup. Set empty to lazy-load only. |
| `MAGENTA_RT_MAX_DURATION` | `60` | API-wide duration cap in seconds. |
| `MAGENTA_RT_MAX_BATCH_SIZE` | `4` | Maximum generated clips per request. |
| `MAGENTA_RT_MLX_BITS` | `8` | Quantization for the Python MLX backend. Ignored by `mlxfn`. |
| `MAGENTA_RT_MLXFN_WARMUP_STEPS` | `5` | Warmup steps for the exported MLX function backend. |
| `MAGENTA_RT_AUTO_DOWNLOAD` | `false` | If true, invoke upstream download commands when assets are missing. |
| `MAGENTA_RT_OUTPUT_DIR` | `outputs` | Local output directory for job artifacts. |
| `MAGENTA_RT_STORAGE_BUCKET` | unset | S3/R2 bucket for job artifacts. Enables S3-compatible storage. |
| `MAGENTA_RT_STORAGE_PREFIX` | `magenta-rt/jobs` | Object key prefix for uploaded WAV/ZIP artifacts. |
| `MAGENTA_RT_STORAGE_ENDPOINT_URL` | unset | S3-compatible endpoint URL, such as Cloudflare R2. |
| `MAGENTA_RT_STORAGE_REGION` | `us-east-1` | S3 region. Use `auto` for Cloudflare R2 if desired. |
| `MAGENTA_RT_STORAGE_PUBLIC_BASE_URL` | unset | Optional public/CDN base URL. If unset, the API generates presigned URLs. |
| `MAGENTA_RT_PRESIGNED_URL_EXPIRES` | `3600` | Presigned download URL lifetime in seconds. |
