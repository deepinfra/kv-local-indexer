#!/usr/bin/env bash
# Build the kv_local_indexer wheel as a PORTABLE manylinux wheel.
#
# Why this works WITHOUT the TRT-LLM base image: the compiled extension links
# only the standard C runtime (libc / libm / libgcc_s + the loader) -- no
# libpython, no libstdc++, nothing from the engine image. So a wheel built in a
# stock manylinux box runs in the engine image as long as the engine's glibc is
# >= the build box's glibc. (Engine today: glibc 2.39 / cp312. manylinux_2_28
# pins the floor at glibc 2.28, comfortably below that.)
#
# This means the build needs NOTHING from localhost:30500, so it can also run in
# plain GitHub Actions CI.
#
# Usage:
#   build_wheel.sh                 # default manylinux_2_28
#   build_wheel.sh manylinux_2_34  # bump the glibc floor if ever needed
#
# Requires: docker + network (rustup, crates.io, and the pinned dynamo git rev).
set -euo pipefail

FLAVOR="${1:-manylinux_2_28}"
IMAGE="quay.io/pypa/${FLAVOR}_x86_64"

CRATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${CRATE_DIR}/dist"
mkdir -p "${OUT_DIR}"

echo "Building kv_local_indexer wheel in ${IMAGE} ..."
docker run --rm \
  -v "${CRATE_DIR}":/io \
  "${IMAGE}" bash -lc '
    set -euo pipefail
    cd /io
    # cp312 toolchain ships in the manylinux image.
    export PATH=/opt/python/cp312-cp312/bin:${PATH}
    # libclang: a transitive dynamo build-dep uses bindgen.
    dnf install -y -q clang >/dev/null
    # Rust toolchain (pinned to match rust-toolchain.toml).
    export RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
    export PATH=/opt/cargo/bin:${PATH}
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain 1.93.1
    export CARGO_TARGET_DIR=/tmp/kv_target
    pip install -q -U maturin
    # --compatibility manylinux_2_28: tag + verify the wheel only links allowed
    # system libs (it does), producing a portable wheel.
    maturin build --release -i python3.12 \
      --compatibility "'"${FLAVOR}"'" --out /io/dist
  '

echo "Done. Wheels in ${OUT_DIR}:"
ls -1 "${OUT_DIR}"/*.whl
