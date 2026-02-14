#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/compilation/pybind"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

mkdir -p "${BUILD_DIR}"
cd "${ROOT_DIR}"
rm -rf "compilation/pybind/temp" "compilation/pybind/lib"
"${PYTHON_BIN}" "compilation/setup_translation_gpu_cpp.py" build_ext --inplace \
  --force \
  --build-temp "compilation/pybind/temp" \
  --build-lib "compilation/pybind/lib"
