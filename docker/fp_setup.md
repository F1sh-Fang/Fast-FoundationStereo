# 从基础 CUDA 容器手动配置 foundation

本节是从零开始的独立流程：不依赖现有 `ffs` 容器或 `ffs-snapshot` 镜像，直接从 `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` 创建名为 `foundation` 的容器，按 Fast-FoundationStereo 和 `foundationpose_cpp` 两份安装流程配置，最后提交为 `foundation:cuda12.4-trt10.11` 镜像。

本流程不把项目源码复制进镜像。源码、模型、测试数据通过 bind mount 提供，因此后续修改代码不需要重新制作镜像。

## A. 宿主机准备

以下命令在宿主机执行：

```bash
export FFS_ROOT=/home/f1sh/DexHand/Fast-FoundationStereo
```

确认基础镜像、两个项目和 TensorRT/CV-CUDA 包存在：

```bash
docker pull nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

```bash
test -f "$FFS_ROOT/docker/packages/cvcuda-lib-0.12.0_beta-cuda12-x86_64-linux.deb" && test -f "$FFS_ROOT/docker/packages/cvcuda-dev-0.12.0_beta-cuda12-x86_64-linux.deb" && echo cvcuda-packages-OK
```

如果需要离线安装 TensorRT，确认本地仓库目录存在：

```bash
test -f "$FFS_ROOT/docker/packages/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9/Packages" && echo tensorrt-local-repo-OK
```

Miniconda 安装包如果不存在，进入容器后会从 Anaconda 在线下载。

## B. 从基础镜像创建 foundation 容器

下面保留与你现有部署一致的 GPU 参数：`--gpus all` 负责请求全部 GPU，`--runtime nvidia` 显式指定 NVIDIA runtime，`--env NVIDIA_DISABLE_REQUIRE=1` 避免驱动/容器 CUDA 版本检查阻止启动。较新的 NVIDIA Container Toolkit 通常可仅使用 `--gpus all`，但保留这两个兼容参数更稳妥；同时保留 `/home` 和 `/mnt` 挂载，确保宿主机路径在容器内可见。

先检查是否已有同名容器：

```bash
docker ps -a --filter name=^/foundation$
```

确认没有需要保留的同名容器后，创建容器：

```bash
docker run --gpus all \
  --runtime nvidia \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --detach \
  --network host \
  --ipc host \
  --name foundation \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e DISPLAY="${DISPLAY:-}" \
  -v /home:/home \
  -v /home/f1sh/DexHand/foundationpose_cpp:/workspace/foundationpose_cpp \
  -v /home/f1sh/DexHand/Fast-FoundationStereo:/workspace/Fast-FoundationStereo \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp:/tmp \
  -v /mnt:/mnt \
  -w /workspace \
  nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 \
  sleep infinity
```

进入容器：

```bash
docker exec -it foundation bash
```

下面从 C 节开始的命令均在 `foundation` 容器内执行。容器内已经是 `root`，不要加 `sudo`。

## C. 系统基础依赖

```bash
apt-get update
```

```bash
apt-get install -y --no-install-recommends \
  build-essential ca-certificates cmake curl ffmpeg git wget zstd \
  libturbojpeg-dev pkg-config \
  libx11-xcb1 libxcb-xinerama0 libxcb-icccm4 libxcb-render-util0 \
  libxcb-shape0 libxcb-keysyms1 libxcb-image0 libxkbcommon-x11-0 \
  libxcb-cursor0 libxcb-xkb1 libxcb-render0 libxcb-shm0 libxcb-sync1 \
  libxcb-xfixes0 libxcb-randr0 libxcb-xtest0 libsm6 libxext6 \
  libxkbcommon0
```

## D. Miniconda、PyTorch 和 Fast-FoundationStereo Python 环境

```bash
if test -f /workspace/Fast-FoundationStereo/docker/packages/Miniconda3-latest-Linux-x86_64.sh; then \
  bash /workspace/Fast-FoundationStereo/docker/packages/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda; \
else \
  wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/Miniconda3-latest-Linux-x86_64.sh; \
  bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda; \
fi
```

```bash
source /opt/conda/etc/profile.d/conda.sh
```

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

```bash
conda create -y -n my python=3.12 && conda activate my
```

让以后通过 `docker exec ... bash` 进入容器时自动激活环境：

```bash
conda init bash
```

```bash
grep -qxF 'conda activate my' /root/.bashrc || echo 'conda activate my' >> /root/.bashrc
```

```bash
pip install --no-cache-dir uv
```

```bash
UV_NO_CACHE=1 uv pip install torch==2.6.0 torchvision==0.21.0 xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124
```

```bash
cd /workspace/Fast-FoundationStereo
```

```bash
UV_NO_CACHE=1 uv pip install -r requirements.txt numpy==1.26.4 opencv-contrib-python==4.11.0.86
```

## E. TensorRT 10.11 Python 和系统开发包

先安装 Python 推理依赖。TensorRT 的 PyPI 包是占位包，真实 wheel 位于 NVIDIA 专用索引，因此必须保留 `--extra-index-url https://pypi.nvidia.com`；否则可能触发 `wheel-stub` 下载并出现 `Downloaded wheel and sha256 don't match`：

```bash
UV_NO_CACHE=1 uv pip install --extra-index-url https://pypi.nvidia.com \
  onnx onnxruntime-gpu pycuda cuda-python h5py \
  tensorrt-cu12==10.11.0.33 \
  tensorrt-lean-cu12==10.11.0.33 \
  tensorrt-dispatch-cu12==10.11.0.33
```

如果之前已经执行过失败的命令，直接用上面的命令重试即可；已经成功安装的包会被复用，不需要先卸载。这个问题与 CUDA 驱动显示为 13.0 无关，当前容器的 CUDA Toolkit 仍由 `/usr/local/cuda` 决定。

```bash
conda install -y -c conda-forge libstdcxx-ng
```

配置 TensorRT 本地仓库。如果本地目录存在，就使用它；否则保留基础镜像已有的 NVIDIA 在线源：

```bash
TRT_REPO=/workspace/Fast-FoundationStereo/docker/packages/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9
```

```bash
if test -f "$TRT_REPO/Packages"; then \
  cp "$TRT_REPO/nv-tensorrt-local-5BF87A98-keyring.gpg" /usr/share/keyrings/; \
  printf '%s\n' "deb [signed-by=/usr/share/keyrings/nv-tensorrt-local-5BF87A98-keyring.gpg] file:$TRT_REPO /" > /etc/apt/sources.list.d/nv-tensorrt-local.list; \
  printf '%s\n' \
    'Package: libnvinfer* libnvonnxparsers* libnvparsers* tensorrt*' \
    'Pin: origin ""' \
    'Pin-Priority: 1001' \
    > /etc/apt/preferences.d/99-tensorrt-local; \
  apt-get update; \
else \
  echo '本地 TensorRT 仓库不存在，将使用在线源'; \
  apt-get update; \
fi
```

```bash
apt-cache policy libnvinfer10 libnvinfer-dev libnvonnxparsers-dev libnvinfer-bin
```

如果本地仓库和在线源提供相同版本，APT 不会因为版本号相同就自动选择本地源。因此上面的 pin 将本地 `file:` 源设为 `1001`，高于在线源的 `600`。在有本地仓库时，以上命令的候选版本表中应看到 `file:/workspace/Fast-FoundationStereo/docker/packages/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9`，候选版本必须为 `10.11.0.33-1+cuda12.9`。如果候选仍来自 `https://developer.download.nvidia.com`，先不要安装，检查 pin 和源：

```bash
cat /etc/apt/preferences.d/99-tensorrt-local
apt-cache policy libnvinfer10
```

安装运行库、开发包和 `trtexec`。本地仓库存在时，不再使用包名解析，而是把每个 `.deb` 的绝对路径直接传给 APT；这样目标 TensorRT 包必定来自本地文件：

```bash
TRT_VERSION=10.11.0.33-1+cuda12.9
```

```bash
if test -f "$TRT_REPO/libnvinfer10_${TRT_VERSION}_amd64.deb"; then \
  apt-get install -y --reinstall --no-install-recommends \
    "$TRT_REPO/libnvinfer10_${TRT_VERSION}_amd64.deb" \
    "$TRT_REPO/libnvinfer-lean10_${TRT_VERSION}_amd64.deb" \
    "$TRT_REPO/libnvinfer-plugin10_${TRT_VERSION}_amd64.deb" \
    "$TRT_REPO/libnvinfer-vc-plugin10_${TRT_VERSION}_amd64.deb" \
    "$TRT_REPO/libnvinfer-dispatch10_${TRT_VERSION}_amd64.deb" \
    "$TRT_REPO/libnvonnxparsers10_${TRT_VERSION}_amd64.deb" \
    "$TRT_REPO/libnvinfer-bin_${TRT_VERSION}_amd64.deb" \
    "$TRT_REPO/libnvinfer-dev_${TRT_VERSION}_amd64.deb" \
    "$TRT_REPO/libnvinfer-headers-dev_${TRT_VERSION}_amd64.deb" \
    "$TRT_REPO/libnvinfer-headers-plugin-dev_${TRT_VERSION}_amd64.deb" \
    "$TRT_REPO/libnvinfer-plugin-dev_${TRT_VERSION}_amd64.deb" \
    "$TRT_REPO/libnvonnxparsers-dev_${TRT_VERSION}_amd64.deb"; \
else \
  echo '本地 TensorRT .deb 不完整，回退到在线 APT 源'; \
  apt-get install -y --no-install-recommends \
    libnvinfer10="$TRT_VERSION" \
    libnvinfer-lean10="$TRT_VERSION" \
    libnvinfer-plugin10="$TRT_VERSION" \
    libnvinfer-vc-plugin10="$TRT_VERSION" \
    libnvinfer-dispatch10="$TRT_VERSION" \
    libnvonnxparsers10="$TRT_VERSION" \
    libnvinfer-bin="$TRT_VERSION" \
    libnvinfer-dev="$TRT_VERSION" \
    libnvinfer-headers-dev="$TRT_VERSION" \
    libnvinfer-headers-plugin-dev="$TRT_VERSION" \
    libnvinfer-plugin-dev="$TRT_VERSION" \
    libnvonnxparsers-dev="$TRT_VERSION"; \
fi
```

```bash
ln -sf /usr/src/tensorrt/bin/trtexec /usr/local/bin/trtexec
```

```bash
trtexec --help 2>&1 | grep -m1 'TensorRT v' && python -c 'import tensorrt as trt; print(trt.__version__)'
```

## F. FoundationPose C++ 依赖和 CV-CUDA

```bash
apt-get install -y --no-install-recommends libopencv-dev libeigen3-dev libgoogle-glog-dev libgtest-dev libassimp-dev
```

```bash
apt-get install -y /workspace/Fast-FoundationStereo/docker/packages/cvcuda-lib-0.12.0_beta-cuda12-x86_64-linux.deb /workspace/Fast-FoundationStereo/docker/packages/cvcuda-dev-0.12.0_beta-cuda12-x86_64-linux.deb
```

如果宿主机没有这两个 CV-CUDA 包，先下载后再执行上面的安装命令：

```bash
wget -c https://github.com/CVCUDA/CV-CUDA/releases/download/v0.12.0-beta/cvcuda-lib-0.12.0_beta-cuda12-x86_64-linux.deb -O /workspace/Fast-FoundationStereo/docker/packages/cvcuda-lib-0.12.0_beta-cuda12-x86_64-linux.deb
```

```bash
wget -c https://github.com/CVCUDA/CV-CUDA/releases/download/v0.12.0-beta/cvcuda-dev-0.12.0_beta-cuda12-x86_64-linux.deb -O /workspace/Fast-FoundationStereo/docker/packages/cvcuda-dev-0.12.0_beta-cuda12-x86_64-linux.deb
```

验证：

```bash
dpkg-query -W libcvcuda0 cvcuda0-dev && test -f /opt/nvidia/cvcuda0/lib/x86_64-linux-gnu/cmake/cvcuda/cvcuda-config.cmake && test -f /opt/nvidia/cvcuda0/lib/x86_64-linux-gnu/cmake/nvcv_types/nvcv_types-config.cmake
```

## G. 编译和验证两个项目

查询 GPU 计算能力：

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

当前 RTX 4000 Ada 的 `8.9` 对应 CMake 值 `89`。按实际 GPU 修改：

```bash
cmake -S /workspace/foundationpose_cpp \
  -B /workspace/foundationpose_cpp/build-foundation \
  -DENABLE_TENSORRT=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_PREFIX_PATH=/opt/nvidia/cvcuda0
```

```bash
cmake --build /workspace/foundationpose_cpp/build-foundation -j"$(nproc)"
```

FoundationPose 模型和测试数据放在：

```text
/workspace/foundationpose_cpp/models/scorer_hwc.onnx
/workspace/foundationpose_cpp/models/refiner_hwc.onnx
/workspace/foundationpose_cpp/test_data/mustard0
```

转换 Engine 并运行测试：

```bash
cd /workspace/foundationpose_cpp && bash tools/cvt_onnx2trt.bash
```

```bash
/workspace/foundationpose_cpp/build-foundation/bin/simple_tests --gtest_filter=foundationpose_test.test
```

回归 Fast-FoundationStereo：

```bash
cd /workspace/Fast-FoundationStereo && python scripts/run_demo.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth --left_file demo_data/left.png --right_file demo_data/right.png --intrinsic_file demo_data/K.txt --out_dir output/foundation-check --remove_invisible 0 --denoise_cloud 1 --scale 1 --get_pc 1 --valid_iters 8 --max_disp 192 --zfar 100
```

## H. 清理并提交 foundation 镜像

两个项目和测试都成功后，删除只在安装阶段使用的本地 TensorRT APT 源配置。已经安装的 TensorRT 不受影响：

```bash
rm -f /etc/apt/sources.list.d/nv-tensorrt-local.list /etc/apt/preferences.d/99-tensorrt-local /usr/share/keyrings/nv-tensorrt-local-5BF87A98-keyring.gpg
```

清理缓存：

```bash
conda clean -afy && python -m pip cache purge || true && uv cache clean || true && apt-get clean && rm -rf /var/lib/apt/lists/* /root/.nv/ComputeCache /root/.cache/torch_extensions /tmp/Miniconda3-latest-Linux-x86_64.sh
```

退出容器，在宿主机停止并提交：

```bash
exit
```

```bash
docker stop foundation
```

```bash
docker commit foundation foundation:cuda12.4-trt10.11
```

验证镜像：

```bash
docker image inspect foundation:cuda12.4-trt10.11 --format '{{.Config.Image}} {{.Size}}'
```

以后直接使用已创建容器：

```bash
docker start foundation
```

```bash
docker exec -it -w /workspace/foundationpose_cpp foundation bash
```

该镜像的分层关系为：

```text
nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
  -> foundation:cuda12.4-trt10.11
```

它不依赖 `ffs-snapshot`。源码、模型和测试数据属于 bind mount，不会进入镜像。

如果以后删除了 `foundation` 容器，可以从最终镜像重新创建：

```bash
docker run --gpus all \
  --runtime nvidia \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --detach \
  --network host \
  --ipc host \
  --name foundation \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e DISPLAY="${DISPLAY:-}" \
  -v /home:/home \
  -v /home/f1sh/DexHand/foundationpose_cpp:/workspace/foundationpose_cpp \
  -v /home/f1sh/DexHand/Fast-FoundationStereo:/workspace/Fast-FoundationStereo \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp:/tmp \
  -v /mnt:/mnt \
  -w /workspace/foundationpose_cpp \
  foundation:cuda12.4-trt10.11 \
  sleep infinity
```

---

本章适用于现有 `ffs` 已完成 Fast-FoundationStereo 环境，但还没有挂载 `foundationpose_cpp` 的情况。流程恢复为最初的快照路线：先将 `ffs` commit 为过渡镜像 -> 保留原容器为 `ffs-original` -> 从过渡镜像创建带双项目挂载的 `foundation` -> 在 `foundation` 中继续安装依赖、编译和测试 -> 清理后 commit 最终镜像。

由于现有 `ffs` 没有 FoundationPose 挂载，不能直接给它追加 bind mount；必须从快照镜像重新创建容器。最终关系为：

```text
现有 ffs 容器
  -> docker commit ffs ffs-snapshot:20260821-pre-foundationpose
  -> docker rename ffs ffs-original
  -> 从 ffs-snapshot 创建带双项目挂载的 foundation 容器
  -> 在 foundation 中安装、清理并 commit foundation:cuda12.4-trt10.11
```

第一次 `docker commit` 只保存当前 FFS 环境；第二次 `docker commit` 在此基础上保存 FoundationPose 新增依赖。两次都会复用父镜像层，不会复制一份完整基础镜像。

当前 `ffs` 不需要挂载 `foundationpose_cpp`；新建 `foundation` 时再挂载以下两个目录：

```text
/workspace/foundationpose_cpp
/workspace/Fast-FoundationStereo
```

`docker commit` 不会包含 bind mount 中的项目源码、模型和测试数据，所以新 `foundation` 必须重新指定挂载参数。

## I. 检查已有 ffs

以下命令在宿主机执行。确认 `ffs` 存在且当前没有同名 `foundation` 容器：

```bash
docker ps -a --filter name=^/ffs$ --size
```

```bash
docker ps -a --filter name=^/foundation$ --size
```

查看当前 `ffs` 的挂载、工作目录和镜像：

```bash
docker inspect ffs --format 'Image={{.Config.Image}} WorkDir={{.Config.WorkingDir}} Mounts={{range .Mounts}}{{.Source}} -> {{.Destination}}; {{end}}'
```

当前 `ffs` 只需要能看到 Fast-FoundationStereo 挂载：

```text
/home/f1sh/DexHand/Fast-FoundationStereo -> /workspace/Fast-FoundationStereo
```

先停止并提交当前 `ffs`，生成可回退的过渡快照：

```bash
docker stop ffs
```

```bash
docker commit ffs ffs-snapshot:20260821-pre-foundationpose
```

确认快照存在：

```bash
docker image inspect ffs-snapshot:20260821-pre-foundationpose --format '{{.Id}} {{.Size}}'
```

再将原容器改名为备份容器，保留它作为最终回退点：

```bash
docker rename ffs ffs-original
```

从过渡快照创建新的 `foundation` 容器，并在创建时挂载两个项目：

```bash
docker run --gpus all \
  --runtime nvidia \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --detach \
  --network host \
  --ipc host \
  --name foundation \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e DISPLAY="${DISPLAY:-}" \
  -v /home:/home \
  -v /home/f1sh/DexHand/foundationpose_cpp:/workspace/foundationpose_cpp \
  -v /home/f1sh/DexHand/Fast-FoundationStereo:/workspace/Fast-FoundationStereo \
  -v /home/f1sh/DexHand/Fast-FoundationStereo/docker/packages/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9:/var/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp:/tmp \
  -v /mnt:/mnt \
  -w /workspace/foundationpose_cpp \
  ffs-snapshot:20260821-pre-foundationpose \
  sleep infinity
```

验证新容器和两个项目挂载：

```bash
docker exec foundation bash -lc 'set -e; test -f /workspace/foundationpose_cpp/CMakeLists.txt; test -f /workspace/Fast-FoundationStereo/readme.md; /opt/conda/envs/my/bin/python -c "import torch; import tensorrt as trt; print(torch.cuda.is_available()); print(trt.__version__)"; echo foundation-created-OK'
```

## J. 在 foundation 中完成剩余安装和验证

进入新创建的容器：

```bash
docker exec -it foundation bash
```

FFS 原有环境已经包含在 `ffs-snapshot` 中；现在在 `foundation` 内继续执行本文前面的 C～F 节中尚未完成的步骤，并完成 FoundationPose 构建：

- TensorRT Python 和系统开发包
- OpenCV、Eigen、glog、GTest、Assimp
- CV-CUDA
- CMake、编译、模型转换和 FoundationPose 测试
- Fast-FoundationStereo 回归测试

所有测试通过后再继续下一节。

## K. 清理 foundation 缓存

进入容器：

```bash
docker exec -it foundation bash
```

在容器内执行：

```bash
conda clean -afy
```

```bash
python -m pip cache purge || true
```

```bash
uv cache clean || true
```

```bash
apt-get clean
```

如果安装阶段使用过挂载进来的 TensorRT 本地仓库，提交前删除可能失效的本地源配置（仓库本身是只读 bind mount，不会进入镜像）：

```bash
rm -f /etc/apt/sources.list.d/nv-tensorrt-local.list /etc/apt/sources.list.d/nv-tensorrt-local-ubuntu2204-10.11.0-cuda-12.9.list /etc/apt/preferences.d/99-tensorrt-local /usr/share/keyrings/nv-tensorrt-local-5BF87A98-keyring.gpg
```

```bash
rm -rf /var/lib/apt/lists/* /root/.nv/ComputeCache /root/.cache/torch_extensions
```

不要删除以下内容：

```text
/opt/conda/envs/my
/usr/local/cuda
/usr/src/tensorrt
/opt/nvidia/cvcuda0
```

退出容器：

```bash
exit
```

注意：如果某些缓存已经存在于 `ffs-snapshot` 父镜像层，当前删除只会让它们在最终镜像中不可见，不会回收父层本身的磁盘占用。

## L. 停止 foundation 并 commit 最终镜像

清理完成后退出容器并停止 `foundation`：

```bash
docker stop foundation
```

提交当前软件环境为最终镜像：

```bash
docker commit foundation foundation:cuda12.4-trt10.11
```

确认镜像存在：

```bash
docker image ls foundation:cuda12.4-trt10.11
```

```bash
docker image inspect foundation:cuda12.4-trt10.11 --format '{{.Id}} {{.Size}}'
```

此时镜像保存了软件环境，但不包含 bind mount 的项目文件。

最终镜像基于 `ffs-snapshot:20260821-pre-foundationpose`；这次 commit 只增加 `foundation` 容器中新安装依赖和配置产生的可写层。`ffs-original` 仍然保留为回退容器。

## M. 验证 foundation 容器

容器已经在 I 节创建；这里仅验证容器状态和两个挂载：

```bash
docker ps -a --filter name=^/foundation$
```

```bash
docker start foundation
```

## N. 从最终镜像重新创建并验证（可选）

如果以后删除了已经提交的 `foundation` 容器，可以从最终镜像重新创建；源码和模型仍通过 bind mount 提供：

```bash
docker run --gpus all \
  --runtime nvidia \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --detach \
  --network host \
  --ipc host \
  --name foundation \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e DISPLAY="${DISPLAY:-}" \
  -v /home:/home \
  -v /home/f1sh/DexHand/foundationpose_cpp:/workspace/foundationpose_cpp \
  -v /home/f1sh/DexHand/Fast-FoundationStereo:/workspace/Fast-FoundationStereo \
  -v /home/f1sh/DexHand/Fast-FoundationStereo/docker/packages/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9:/var/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp:/tmp \
  -v /mnt:/mnt \
  -w /workspace/foundationpose_cpp \
  foundation:cuda12.4-trt10.11 \
  sleep infinity
```

启动或进入容器：

```bash
docker start foundation
```

```bash
docker exec foundation bash -lc 'set -e; test -f /workspace/foundationpose_cpp/CMakeLists.txt; test -f /workspace/Fast-FoundationStereo/readme.md; /opt/conda/envs/my/bin/python -c "import torch; import tensorrt as trt; print(torch.cuda.is_available()); print(trt.__version__)"; echo foundation-container-OK'
```

进入容器：

```bash
docker exec -it -w /workspace/foundationpose_cpp foundation bash
```

## O. 日常使用与回退

日常启动和进入：

```bash
docker start foundation
```

```bash
docker exec -it -w /workspace/foundationpose_cpp foundation bash
```

停止：

```bash
docker stop foundation
```

如果新 `foundation` 创建或验证失败，原始环境仍然保留为 `ffs-original`：

```bash
docker start ffs-original
```

```bash
docker exec -it -w /workspace/Fast-FoundationStereo ffs-original bash
```

如果需要重新创建 `foundation`，删除失败的 `foundation` 容器后，从 `foundation:cuda12.4-trt10.11` 再执行本章 N 节的 `docker run`。

确认新 `foundation` 正常后，查看占用：

```bash
docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Size}}' | grep -E '^(NAMES|ffs|foundation)($|[[:space:]])'
```

```bash
docker system df -v
```

最终 FoundationPose 和 Fast-FoundationStereo 都验证完成后，`ffs-original` 仍可作为回退环境；确认不再需要时再删除。
