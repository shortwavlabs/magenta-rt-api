# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/app/.venv \
    MAGENTA_HOME=/workspace/Magenta \
    MAGENTA_RT_BACKEND=mlxfn \
    MAGENTA_RT_DEFAULT_MODEL=mrt2_small \
    MAGENTA_RT_PRELOAD_MODELS=

ENV PATH="${VIRTUAL_ENV}/bin:/root/.local/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY pyproject.toml README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --python 3.12

COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --python 3.12

EXPOSE 8000

CMD ["magenta-rt-api", "--host", "0.0.0.0", "--port", "8000"]
