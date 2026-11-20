# syntax=docker/dockerfile:1
FROM python:3.13.18-slim-bookworm

# system libraries for opencv
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# install segger; uv reads the package indexes from pyproject.toml in /src
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN --mount=from=ghcr.io/astral-sh/uv:0.13.9,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv venv "$VIRTUAL_ENV" && uv pip install .

RUN useradd --create-home segger
USER segger
WORKDIR /data

ENTRYPOINT ["segger"]
CMD ["--help"]
