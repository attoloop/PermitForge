#!/bin/sh
# VM launcher; execute from the unpacked PermitForge root. No rental management.
set -eu
exec uv run --project train/cloud --frozen python -m train.cloud.run "$@"
