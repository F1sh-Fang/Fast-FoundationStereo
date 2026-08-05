#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
WORKSPACE_ROOT=$(dirname "${PROJECT_ROOT}")
CONTAINER_NAME="${FFS_CONTAINER_NAME:-ffs}"
IMAGE_NAME="${FFS_IMAGE_NAME:-ffs:latest}"
PROJECT_IN_CONTAINER="${FFS_PROJECT_IN_CONTAINER:-/workspace/$(basename "${PROJECT_ROOT}")}"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "容器 ${CONTAINER_NAME} 已存在，请直接运行 docker/2_start_container.sh。"
  echo "如需重建，请先自行备份容器内数据，再手动删除该容器。"
  exit 0
fi

if [[ -n "${DISPLAY:-}" ]] && command -v xhost >/dev/null 2>&1; then
  xhost +local:root >/dev/null 2>&1 || true
fi

echo "首次创建持久化容器 ${CONTAINER_NAME} ..."
docker run --gpus all \
  --runtime nvidia \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --detach \
  --network host \
  --name "${CONTAINER_NAME}" \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc host \
  -e DISPLAY="${DISPLAY:-}" \
  -v "${WORKSPACE_ROOT}:/workspace" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp:/tmp \
  -v /home:/home \
  -v /mnt:/mnt \
  -w "${PROJECT_IN_CONTAINER}" \
  "${IMAGE_NAME}" sleep infinity >/dev/null

echo "容器已创建。正在检查一次性 Python 运行环境..."
"${SCRIPT_DIR}/3_prepare_python_env.sh"

if [[ "${FFS_NO_ATTACH:-0}" != "1" ]]; then
  exec "${SCRIPT_DIR}/2_start_container.sh"
fi
