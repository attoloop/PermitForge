# Builds the v1 verifier's native models exactly as permit-forge v1 did (tests/Dockerfile, builder stage):
# Debian bookworm 2025-02-24 snapshot gfortran, AERMOD with instrument.py (per-year statistics), static binaries.
FROM debian:bookworm-20250224-slim@sha256:12c396bd585df7ec21d5679bb6a83d4878bc4415ce926c9e5ea6426d23c60bdc AS builder
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/* && printf '%s\n' 'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20250224T000000Z bookworm main' > /etc/apt/sources.list && apt-get update && apt-get install -y --no-install-recommends gfortran python3
COPY model-source/ /build/
COPY build-models.sh /build/build-models.sh
COPY instrument.py /build/instrument.py
RUN python3 /build/instrument.py /build/aermod
RUN sh /build/build-models.sh
FROM scratch
COPY --from=builder /opt/models/ /
