#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DATA_DIR="${1:-/data/2025_stu/lr/A/data}"
DST_DATA_DIR="${ROOT_DIR}/data"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${DST_DATA_DIR}"

mapfile -t datasets < <(cd "${ROOT_DIR}" && PYTHONPATH=src "${PYTHON_BIN}" - <<'PY'
from pdc2_imvc.configs.configure import get_required_data_files
for x in get_required_data_files():
    print(x)
PY
)

missing=()
linked=()

for f in "${datasets[@]}"; do
  src="${SRC_DATA_DIR}/${f}"
  dst="${DST_DATA_DIR}/${f}"
  if [[ -f "${src}" ]]; then
    ln -sfn "${src}" "${dst}"
    linked+=("${f}")
  else
    missing+=("${f}")
  fi
done

echo "Linked ${#linked[@]} files into ${DST_DATA_DIR}"
for f in "${linked[@]}"; do
  echo "  - ${f}"
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Missing ${#missing[@]} files in ${SRC_DATA_DIR}:" >&2
  for f in "${missing[@]}"; do
    echo "  - ${f}" >&2
  done
  exit 2
fi

echo "All configure datasets are linked."
