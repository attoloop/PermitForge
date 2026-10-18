#!/bin/sh
# VM launcher; install the documented frozen extras first. No rental management.
set -eu
exec uv run --project train/cloud --frozen --no-sync python -m train.cloud.run "$@"
