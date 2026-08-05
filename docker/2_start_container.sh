#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
CONTAINER_NAME="${FFS_CONTAINER_NAME:-ffs}"
PROJECT_IN_CONTAINER="${FFS_PROJECT_IN_CONTAINER:-/workspace/$(basename "${PROJECT_ROOT}")}"

if ! docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "未找到容器 ${CONTAINER_NAME}。请先运行 docker/1_create_container.sh。" >&2
  exit 1
fi

if [[ -n "${DISPLAY:-}" ]] && command -v xhost >/dev/null 2>&1; then
  xhost +local:root >/dev/null 2>&1 || true
fi

docker start "${CONTAINER_NAME}" >/dev/null
echo "容器 ${CONTAINER_NAME} 已启动。"

if [[ "${FFS_NO_ATTACH:-0}" != "1" ]]; then
  exec docker exec -it \
    -e DISPLAY="${DISPLAY:-}" \
    -w "${PROJECT_IN_CONTAINER}" \
    "${CONTAINER_NAME}" bash
fi
