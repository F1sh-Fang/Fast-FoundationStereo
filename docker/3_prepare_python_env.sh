#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${FFS_CONTAINER_NAME:-ffs}"
PYTHON_BIN="${FFS_PYTHON_IN_CONTAINER:-/opt/conda/envs/my/bin/python}"
UV_BIN="${FFS_UV_IN_CONTAINER:-/opt/conda/envs/my/bin/uv}"

if ! docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "未找到容器 ${CONTAINER_NAME}。" >&2
  exit 1
fi
docker start "${CONTAINER_NAME}" >/dev/null

if docker exec "${CONTAINER_NAME}" "${PYTHON_BIN}" -c \
  "import torch, torchvision, zmq; assert torch.__version__.startswith('2.6.0')" \
  >/dev/null 2>&1; then
  if docker exec "${CONTAINER_NAME}" "${PYTHON_BIN}" -c \
    "import onnxconverter_common" >/dev/null 2>&1; then
    echo "Python 环境已就绪，无需下载。"
    exit 0
  fi
  echo "Torch 环境已就绪，补装 TensorRT 11 所需的 ONNX FP16 转换器..."
  docker exec "${CONTAINER_NAME}" "${UV_BIN}" pip install \
    --python "${PYTHON_BIN}" onnxconverter-common \
    -i https://mirrors.aliyun.com/pypi/simple/
  exit 0
fi

echo "检测到旧镜像的 Torch/TorchVision 或 pyzmq 不兼容。"
echo "下面只在当前持久化容器中修复一次，后续启动不会重复下载。"
docker exec "${CONTAINER_NAME}" "${UV_BIN}" pip install \
  --python "${PYTHON_BIN}" \
  torch==2.6.0 torchvision==0.21.0 xformers==0.0.29.post3 \
  --index-url https://download.pytorch.org/whl/cu124
docker exec "${CONTAINER_NAME}" "${UV_BIN}" pip install \
  --python "${PYTHON_BIN}" \
  pyzmq -i https://mirrors.aliyun.com/pypi/simple/
docker exec "${CONTAINER_NAME}" "${PYTHON_BIN}" -c \
  "import torch, torchvision, zmq; assert torch.__version__.startswith('2.6.0'); print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'zmq', zmq.__version__)"
