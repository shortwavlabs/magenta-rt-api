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

## Realtime Audio Streaming

Magenta RT 2's native realtime host path is the C++ `RealtimeRunner`: it runs an inference thread at 25 model frames per second, writes 48 kHz stereo audio into lock-free ring buffers, and lets an audio callback pull arbitrary buffer sizes with `read_audio_stereo`.

This API exposes a Python WebSocket stream built on the same stateful generation idea. It calls the Python model's `generate(..., state=state)` repeatedly, keeps the returned transformer state, and streams each generated chunk to the client. It is useful for browser/network clients and prototyping, but the C++ runner is still the lowest-latency path for DAW/plugin-style realtime audio.

Connect to:

```text
ws://localhost:8000/v1/audio/realtime
```

Send a JSON start message first:

```json
{
  "type": "start",
  "model": "mrt2_small",
  "backend": "mlxfn",
  "prompt": "liquid drum and bass, rolling bass, glossy pads",
  "duration": 8,
  "chunk_frames": 1,
  "temperature": 1.2,
  "top_k": 40,
  "cfg_musiccoca": 3.0,
  "cfg_notes": 1.0,
  "cfg_drums": 1.0,
  "seed": 0
}
```

The server replies with:

1. A JSON `ready` message describing sample rate, channels, format, and chunk sizing.
2. For each audio chunk, a JSON `chunk` message followed by one binary WebSocket message.
3. A JSON `done` message when `duration` has been generated.

Binary chunks are interleaved stereo little-endian float32 (`f32le`) at 48 kHz:

```text
L0, R0, L1, R1, L2, R2, ...
```

`chunk_frames` controls latency and overhead. One MRT2 frame is 1920 samples, or 40 ms at 48 kHz:

| `chunk_frames` | Audio per binary chunk | Notes |
| --- | --- | --- |
| `1` | 40 ms | Lowest API latency, more WebSocket messages. |
| `2` | 80 ms | Good browser default. |
| `5` | 200 ms | Lower message overhead, less realtime-feeling. |
| `25` | 1 second | Useful for simple clients and debugging. |

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

## API Call Options

FastAPI also exposes generated schema docs at `http://localhost:8000/docs` while the server is running. The tables below explain the request fields in plain language.

### Text Generation

Use `POST /v1/audio/generations` for synchronous text-to-audio generation, or `POST /jobs` for the same request in the background.

Both endpoints accept a JSON body:

| Field | Type | Default | Range / values | What it does |
| --- | --- | --- | --- | --- |
| `model` | string | `MAGENTA_RT_DEFAULT_MODEL`, usually `mrt2_small` | `mrt2_small`, `mrt2_base`; aliases include `small`, `base`, and `google/magenta-realtime-2` | Chooses the Magenta RT 2 model size. `mrt2_small` is faster and easier to run; `mrt2_base` is higher quality but much heavier. |
| `backend` | string or `null` | `MAGENTA_RT_BACKEND`, usually `mlxfn` | `mlxfn`, `mlx`, `jax` | Chooses the inference backend. `mlxfn` uses the exported MLX function downloaded by `mrt models download`; `mlx` builds the Python MLX model from a checkpoint; `jax` uses the JAX backend and raw checkpoint. |
| `prompt` | string | required | non-empty text | Text style description passed through MusicCoCa. Genre, instrumentation, groove, texture, production style, and energy level usually matter more than long narrative prompts. |
| `duration` | number | `4.0` | `> 0`, capped by `MAGENTA_RT_MAX_DURATION` and model duration limits | Output length in seconds. Internally this becomes `duration * 25` model frames at 48 kHz. |
| `temperature` | number | `1.3` | `> 0` to `5.0` | Sampling randomness. Lower values are steadier and more conservative; higher values explore more and can become less controlled. |
| `top_k` | integer | `40` | `0` to `1024` | Limits sampling to the most likely token choices. Smaller values are more focused; larger values allow more variety. `0` leaves the backend with the broadest allowed sampling behavior. |
| `cfg_musiccoca` | number | `3.0` | `-1.0` to `7.0` | Classifier-free guidance strength for the text/audio style conditioning. Higher values follow the MusicCoCa style prompt more strongly. |
| `cfg_notes` | number | `1.0` | `-1.0` to `7.0` | Guidance strength for note conditioning. The current API does not expose explicit note arrays, so this mostly controls how strongly the model treats masked note conditioning. |
| `cfg_drums` | number | `1.0` | `-1.0` to `7.0` | Guidance strength for drum conditioning. The current API does not expose explicit drum triggers, so this mostly controls the model's masked drum-conditioning behavior. |
| `batch_size` | integer | `1` | `1` to `MAGENTA_RT_MAX_BATCH_SIZE` | Number of independent clips to generate from the same settings. `1` returns a WAV; values above `1` return a ZIP of WAV files. |
| `seed` | integer | `0` | `>= 0` | Seed passed to the MusicCoCa text mapper. Batch generation increments this seed for each clip, so `seed: 10` with `batch_size: 3` uses `10`, `11`, and `12`. |
| `audio_style_weight` | number | `0.5` | `0.0` to `1.0` | Only used by audio-style endpoints. It can be included in text-generation JSON, but it has no effect unless an audio file is uploaded. |

Realtime WebSocket streams accept the same fields, with two differences:

| Field | Default | Range / values | What it does |
| --- | --- | --- | --- |
| `chunk_frames` | `1` | `1` to `25` | Number of 40 ms MRT2 frames generated per binary WebSocket chunk. |
| `audio_format` | `f32le` | `f32le` | Binary chunk encoding. Currently always interleaved stereo float32 little-endian. |

`batch_size` is fixed to `1` for realtime streams.

Synchronous generation returns:

- `audio/wav` when `batch_size` is `1`
- `application/zip` when `batch_size` is greater than `1`

Useful response headers:

| Header | Meaning |
| --- | --- |
| `X-Model` | Model used for generation. |
| `X-Backend` | Backend used for generation. |
| `X-Sample-Rate` | Output sample rate, currently `48000`. |
| `X-Output-Count` | Number of generated clips in the response. |

### Audio Style Transfer

Use `POST /v1/audio/variations` for synchronous audio-style generation, or `POST /jobs/variations` for the background version.

These endpoints use `multipart/form-data`, not JSON. They accept the same generation controls as text generation, plus a required audio file:

| Field | Type | Required | What it does |
| --- | --- | --- | --- |
| `audio` | file | yes | Source audio file to embed with MusicCoCa. WAV, MP3, FLAC, OGG, and M4A usually work if `soundfile` can decode them in your environment. |
| `prompt` | string | yes | Text style prompt to blend with the uploaded source-audio style. |
| `audio_style_weight` | number | no | Blend between text style and uploaded audio style. `0.0` uses only the text embedding; `1.0` uses only the uploaded-audio embedding; `0.5` blends them evenly. |
| `model`, `backend`, `duration`, `temperature`, `top_k`, `cfg_musiccoca`, `cfg_notes`, `cfg_drums`, `batch_size`, `seed` | form fields | no | Same meaning, defaults, and validation as the JSON text-generation fields. |

Example with more controls:

```bash
curl -X POST http://localhost:8000/v1/audio/variations \
  -F audio=@loop.wav \
  -F model=mrt2_small \
  -F backend=mlxfn \
  -F prompt="driving synthwave with gated drums and wide pads" \
  -F duration=8 \
  -F temperature=1.15 \
  -F top_k=32 \
  -F cfg_musiccoca=3.4 \
  -F cfg_notes=1.0 \
  -F cfg_drums=1.2 \
  -F batch_size=1 \
  -F seed=4 \
  -F audio_style_weight=0.7 \
  --output styled.wav
```

### Jobs

Job endpoints return immediately and generate in the background. They are better for cloud deployments, long clips, batch generation, or requests that should write to local/S3-compatible storage.

| Endpoint | Input | Output |
| --- | --- | --- |
| `POST /jobs` | Same JSON body as `/v1/audio/generations` | `202` with `id`, `status`, and `status_url`. |
| `POST /jobs/variations` | Same multipart form as `/v1/audio/variations` | `202` with `id`, `status`, and `status_url`. |
| `GET /jobs/{id}` | job ID in URL | Current job state and download metadata. |
| `GET /jobs/{id}/audio` | job ID in URL | Local generated artifact, only when local storage is used and the job succeeded. |

`GET /jobs/{id}` returns these fields:

| Field | Meaning |
| --- | --- |
| `status` | `queued`, `running`, `succeeded`, or `failed`. |
| `mode` | `text-to-audio` or `audio-style-transfer`. |
| `model`, `backend`, `duration`, `frames` | Effective generation settings. `frames` is the model frame count at 25 frames per second. |
| `output_count` | Number of generated clips, available after success. |
| `download_url` | URL for the WAV/ZIP artifact after success. Local storage points back to this API; S3/R2 can return a public or presigned URL. |
| `download_content_type` | `audio/wav` or `application/zip` after success. |
| `storage_backend`, `storage_key` | Where the artifact was written. |
| `sample_rate` | Output sample rate after success. |
| `error` | Failure detail when `status` is `failed`. |

### Health

`GET /health` is useful for dashboards and deployment checks.

| Field | Meaning |
| --- | --- |
| `status` | `ok` when the server is healthy, or `error` if the last model load/generation attempt failed. |
| `loaded` / `loaded_models` | Whether any model is loaded in memory and which backend/model pairs are loaded. |
| `available_models` / `available_backends` | Values accepted by request options. |
| `assets.resources_ready` | Whether shared MusicCoCa/SpectroStream files are present under `$MAGENTA_HOME/magenta-rt-v2/resources`. |
| `assets.models` | Whether each exported `mlxfn` model is present under `$MAGENTA_HOME/magenta-rt-v2/models`. |
| `load_error` | Last model loading error, commonly a missing `.mlxfn`, `.safetensors`, or resource file. |

### Inpainting Compatibility

`POST /v1/audio/inpaint` and `POST /jobs/inpaint` are present so the dashboard/API shape stays close to the Stable Audio testbench, but they return `501`. Magenta RT 2's public Python API currently exposes text/audio style conditioning, not Stable Audio-style region inpainting.

## Endpoints

- `GET /health`
- `POST /jobs`
- `POST /jobs/variations`
- `POST /jobs/inpaint` compatibility endpoint, returns `501`
- `GET /jobs/{id}`
- `POST /v1/audio/generations`
- `POST /v1/audio/variations`
- `POST /v1/audio/inpaint` compatibility endpoint, returns `501`
- `WebSocket /v1/audio/realtime`
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
