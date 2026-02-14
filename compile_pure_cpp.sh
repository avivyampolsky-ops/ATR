#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/compilation/cpp/build"

mkdir -p "${BUILD_DIR}"
cmake -S "${ROOT_DIR}/compilation" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j

BIN_SRC="${ROOT_DIR}/compilation/cpp/bin/main_cpp"
BIN_DST="${ROOT_DIR}/main_cpp"
if [ -f "${BIN_SRC}" ]; then
  cp -f "${BIN_SRC}" "${BIN_DST}"
fi
