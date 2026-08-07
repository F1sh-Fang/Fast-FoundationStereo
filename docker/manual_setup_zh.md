# Fast FoundationStereo 手动容器安装流程

本文记录从 NVIDIA CUDA 基础镜像创建持久化 `ffs` 容器，并在容器内逐步安装、验证依赖的流程。该方式便于定位具体是哪一步改变了 Torch 或 TensorRT 版本。

最终使用的主要版本：

| 组件 | 版本 |
|---|---|
| 基础镜像 | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` |
| Python | `3.12` |
| Torch | `2.6.0+cu124` |
| TorchVision | `0.21.0+cu124` |
| XFormers | `0.0.29.post3` |
| TensorRT Python | `10.11.0.33` |
| TensorRT Debian | `10.11.0.33-1+cuda12.9` |

不要安装 `nvidia-modelopt[torch]`。它可能重新解析 Torch 依赖，并下载另一套 Torch/CUDA 运行库。

## 1. 在宿主机准备离线安装包

进入工程：

```bash
cd /home/f1sh/DexHand/Fast-FoundationStereo
mkdir -p docker/packages
```

下载 Miniconda，`-c` 支持断点续传：

```bash
wget -c \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -O docker/packages/Miniconda3-latest-Linux-x86_64.sh
```

下载 TensorRT 10.11 本地 APT 仓库：

```bash
wget -c \
  https://developer.download.nvidia.com/compute/tensorrt/10.11.0/local_installers/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9_1.0-1_amd64.deb \
  -O docker/packages/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9_1.0-1_amd64.deb
```

确认文件存在：

```bash
ls -lh docker/packages
```

## 2. 创建持久化容器

拉取基础镜像：

```bash
docker pull nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

确保没有同名旧容器，再从工程根目录创建 `ffs`：

```bash
cd /home/f1sh/DexHand/Fast-FoundationStereo

docker run --gpus all \
  --runtime nvidia \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --detach \
  --network host \
  --name ffs \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc host \
  -e DISPLAY="${DISPLAY:-}" \
  -v "$(dirname "$PWD"):/workspace" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp:/tmp \
  -v /home:/home \
  -v /mnt:/mnt \
  -w /workspace/Fast-FoundationStereo \
  nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 \
  sleep infinity
```

进入容器：

```bash
docker exec -it -w /workspace/Fast-FoundationStereo ffs bash
```

从下一节开始，命令均在容器内执行。

## 3. 安装系统基础依赖

```bash
apt-get update
apt-get install -y --no-install-recommends \
  build-essential ca-certificates cmake curl ffmpeg git \
  libturbojpeg-dev pkg-config wget zstd \
  libx11-xcb1 libxcb-xinerama0 libxcb-icccm4 \
  libxcb-render-util0 libxcb-shape0 libxcb-keysyms1 \
  libxcb-image0 libxkbcommon-x11-0 libxcb-cursor0 \
  libxcb-xkb1 libxcb-render0 libxcb-shm0 libxcb-sync1 \
  libxcb-xfixes0 libxcb-randr0 libxcb-xtest0 \
  libsm6 libxext6 libxkbcommon0
```

此时不要从在线 APT 源安装任何 `libnvinfer` 包。

## 4. 安装并初始化 Conda

```bash
bash docker/packages/Miniconda3-latest-Linux-x86_64.sh \
  -b -p /opt/conda
```

```bash
source /opt/conda/etc/profile.d/conda.sh

conda tos accept --override-channels \
  --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels \
  --channel https://repo.anaconda.com/pkgs/r

conda create -y -n my python=3.12
conda activate my
```

让以后进入容器时自动激活 `my`：

```bash
conda init bash
grep -qxF 'conda activate my' ~/.bashrc || \
  echo 'conda activate my' >> ~/.bashrc
```

重新进入 Bash 后，提示符应自动出现 `(my)`：

```bash
bash
which python
python --version
```

后续命令默认已经处于 `(my)` 环境，不再重复执行 `source` 或 `conda activate`。

## 5. 安装并验证 Torch

```bash
pip install --no-cache-dir uv
```

```bash
UV_NO_CACHE=1 uv pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  xformers==0.0.29.post3 \
  --index-url https://download.pytorch.org/whl/cu124
```

```bash
python -c "import torch, torchvision, xformers; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('xformers:', xformers.__version__); print('CUDA available:', torch.cuda.is_available())"
```

预期版本为 `2.6.0+cu124`、`0.21.0+cu124` 和 `0.0.29.post3`，且 CUDA 应为 `True`。

## 6. 安装项目 Python 依赖

```bash
cd /workspace/Fast-FoundationStereo

UV_NO_CACHE=1 uv pip install \
  -r requirements.txt \
  numpy==1.26.4 \
  opencv-contrib-python==4.11.0.86
```

安装推理和导出依赖：

```bash
UV_NO_CACHE=1 uv pip install \
  onnx onnxruntime-gpu pycuda cuda-python h5py \
  tensorrt-cu12==10.11.0.33 \
  tensorrt-lean-cu12==10.11.0.33 \
  tensorrt-dispatch-cu12==10.11.0.33
```

补充运行时 C++ 标准库：

```bash
conda install -y -c conda-forge libstdcxx-ng
```

再次确认 TensorRT 的 Python 安装没有替换 Torch：

```bash
python -c "import torch, torchvision, xformers, tensorrt as trt; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('xformers:', xformers.__version__); print('TensorRT Python:', trt.__version__)"
```

## 7. 安装系统 TensorRT 和 trtexec

注册工程目录中已经下载好的本地仓库：

```bash
dpkg -i docker/packages/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9_1.0-1_amd64.deb

cp \
  /var/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9/nv-tensorrt-local-5BF87A98-keyring.gpg \
  /usr/share/keyrings/

apt-get update
```

检查 APT 能看到 `10.11.0.33-1+cuda12.9`：

```bash
apt-cache policy \
  libnvinfer10 libnvinfer-lean10 libnvinfer-plugin10 \
  libnvinfer-vc-plugin10 libnvinfer-dispatch10 \
  libnvonnxparsers10 libnvinfer-bin
```

必须锁定 `libnvinfer-bin` 的全部直接依赖，否则 APT 可能从 NVIDIA 在线源选择 CUDA 13 对应的最新版：

```bash
apt-get install -y --no-install-recommends \
  libnvinfer10=10.11.0.33-1+cuda12.9 \
  libnvinfer-lean10=10.11.0.33-1+cuda12.9 \
  libnvinfer-plugin10=10.11.0.33-1+cuda12.9 \
  libnvinfer-vc-plugin10=10.11.0.33-1+cuda12.9 \
  libnvinfer-dispatch10=10.11.0.33-1+cuda12.9 \
  libnvonnxparsers10=10.11.0.33-1+cuda12.9 \
  libnvinfer-bin=10.11.0.33-1+cuda12.9
```

`trtexec` 的实际路径不是 `/usr/bin/trtexec`。建立通用软链接：

```bash
ln -sf /usr/src/tensorrt/bin/trtexec /usr/local/bin/trtexec
```

TensorRT 10.11 的 `trtexec` 没有独立的 `--version` 参数，使用帮助输出确认版本：

```bash
trtexec --help 2>&1 | grep -m1 'TensorRT v'
python -c "import tensorrt as trt; print(trt.__version__)"
```

预期分别包含 `TensorRT v101100` 和 `10.11.0.33`。

## 8. 清理安装缓存

只删除缓存，不删除已经安装的环境：

```bash
pip cache purge
uv cache clean || true
conda clean -afy
apt-get clean
rm -rf /root/.cache/uv /var/lib/apt/lists/*
```

TensorRT 安装完成后可以注销并删除本地 APT 仓库的容器内副本。宿主机工程中的原始 `.deb` 仍会保留：

```bash
dpkg -r nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9
rm -rf /var/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9
```

## 9. 最终验证

```bash
cd /workspace/Fast-FoundationStereo

python -c "import torch, torchvision, xformers, tensorrt as trt, zmq; assert torch.__version__ == '2.6.0+cu124'; assert torchvision.__version__ == '0.21.0+cu124'; assert xformers.__version__ == '0.0.29.post3'; assert trt.__version__ == '10.11.0.33'; assert torch.cuda.is_available(); print('environment OK')"

dpkg-query -W -f='${Package} ${Version}\n' \
  libnvinfer10 libnvinfer-bin libnvinfer-plugin10

trtexec --help 2>&1 | grep -m1 'TensorRT v'
```

## 10. 日常使用

退出容器不会删除环境：

```bash
exit
```

在宿主机启动并进入同一个容器：

```bash
docker start ffs
docker exec -it -w /workspace/Fast-FoundationStereo ffs bash
```

直接在容器内启动 PyTorch ZMQ 推理服务：

```bash
cd /workspace/Fast-FoundationStereo
python scripts/zmq_stereo_server.py \
  --model_file weights/23-36-37/model_best_bp2_serialize.pth \
  --valid_iters 4 \
  --max_disp 192 \
  --zmin 0.05 \
  --zmax 2.0
```

退出容器后，在宿主机设置 Docker 重启时自动恢复 `ffs`：

```bash
docker update --restart unless-stopped ffs
```

不需要将该容器 `docker commit` 成大型镜像。所有手工安装内容都保存在 `ffs` 容器的可写层中；不要执行 `docker rm ffs`、`docker container prune` 或会删除停止容器的 `docker system prune`。
