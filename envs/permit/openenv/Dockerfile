# Repository-root context: podman build -f envs/permit/openenv/Dockerfile -t permitforge-space .
# v1 origin: tasks/permit-forge/instances/003bd328a392-s47/tests/Dockerfile (native builder stage).
# Native stage is native_build/Dockerfile verbatim except for COPY source prefixes.
FROM debian:bookworm-20250224-slim@sha256:12c396bd585df7ec21d5679bb6a83d4878bc4415ce926c9e5ea6426d23c60bdc AS builder
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/* && printf '%s\n' 'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20250224T000000Z bookworm main' > /etc/apt/sources.list && apt-get update && apt-get install -y --no-install-recommends gfortran python3
COPY envs/permit/core/native_build/model-source/ /build/
COPY envs/permit/core/native_build/build-models.sh /build/build-models.sh
COPY envs/permit/core/native_build/instrument.py /build/instrument.py
RUN python3 /build/instrument.py /build/aermod
RUN sh /build/build-models.sh

FROM debian:bookworm-20250224-slim@sha256:12c396bd585df7ec21d5679bb6a83d4878bc4415ce926c9e5ea6426d23c60bdc AS runtime
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/* && printf '%s\n' 'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20250224T000000Z bookworm main' > /etc/apt/sources.list && apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/uv && /opt/uv/bin/pip install --no-cache-dir uv==0.11.32
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/opt/permit-user PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=false ENABLE_WEB_INTERFACE=true \
    PERMIT_TASKS_DIR=/app/envs/permit/tasks PERMIT_MODELS_DIR=/opt/models \
    PERMIT_V1_DIR=/app/convert/v1-src PERMIT_FIELDS_JOBS=1 PERMIT_FIELDS_CACHE_GB=8 \
    MAX_CONCURRENT_ENVS=1 PATH=/app/envs/permit/openenv/.venv/bin:$PATH
COPY --from=builder /opt/models/ /opt/models/
COPY envs/permit/core/pyproject.toml /app/envs/permit/core/pyproject.toml
COPY envs/permit/core/permit_core/ /app/envs/permit/core/permit_core/
COPY envs/permit/openenv/pyproject.toml envs/permit/openenv/uv.lock /app/envs/permit/openenv/
COPY envs/permit/openenv/permit_openenv/ /app/envs/permit/openenv/permit_openenv/
COPY envs/permit/openenv/space_root/ /app/envs/permit/openenv/space_root/
RUN /opt/uv/bin/uv sync --project envs/permit/openenv --frozen --no-dev --no-cache --python /usr/bin/python3
COPY envs/permit/tasks/data/ /app/envs/permit/tasks/data/
COPY envs/permit/tasks/met/ /app/envs/permit/tasks/met/
COPY envs/permit/tasks/permit-v2-eval/ /app/envs/permit/tasks/permit-v2-eval/
COPY envs/permit/tasks/permit-v2-train/ /app/envs/permit/tasks/permit-v2-train/
RUN useradd --create-home --home-dir /opt/permit-user --uid 1000 user && mkdir -p /app/convert/v1-src /app/convert/cache /app/results/rollouts && chown -R user:user /app/convert /app/results
USER user
# Sequential tasks, at most four native launches; runtime cold work uses one.
RUN python envs/permit/openenv/space_root/bake_showcases.py --jobs 4
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz')"
CMD ["uvicorn", "permit_openenv.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--ws-ping-timeout", "1800"]
