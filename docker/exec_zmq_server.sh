#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${FFS_CONTAINER_NAME:-ffs}"
PROJECT_DIR="${FFS_PROJECT_IN_CONTAINER:-/workspace/Fast-FoundationStereo}"
PYTHON_BIN="${FFS_PYTHON_IN_CONTAINER:-/opt/conda/envs/my/bin/python}"

if ! docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "未找到容器 ${CONTAINER_NAME}。请先运行 docker/1_create_container.sh。" >&2
  exit 1
fi
docker start "${CONTAINER_NAME}" >/dev/null

docker exec -it \
  -w "${PROJECT_DIR}" \
  "${CONTAINER_NAME}" \
  "${PYTHON_BIN}" scripts/zmq_stereo_server.py "$@"
