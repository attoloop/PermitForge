#!/bin/sh
# Build static aermod (instrumented) + bpipprm into native_build/bin/. Usage: sh build.sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$here/bin"
podman build --quiet --output "type=local,dest=$here/bin" "$here"
sha256sum "$here/bin/aermod" "$here/bin/bpipprm" > "$here/bin/SHA256SUMS"
cat "$here/bin/SHA256SUMS"
