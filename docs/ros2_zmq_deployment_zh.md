# Fast FoundationStereo：D405 + ROS2 + Docker/ZMQ 实时部署

该部署把 ROS2 和 GPU 推理解耦：宿主机订阅 RealSense D405 的左右校正红外图像，使用 ZMQ 把最新一帧发到无 ROS 环境的 `ffs` 容器；容器完成 Fast FoundationStereo 推理后，将视差和米制深度发回宿主机并发布为标准 ROS2 消息。

## 架构与接口

```text
D405 /infra1 + /infra2 + CameraInfo
             |
             v
fast_foundationstereo_ros (宿主机，ROS2)
  PUB tcp://0.0.0.0:5560  -- 左右 mono8 + 标定 -->  ffs 容器
  SUB tcp://127.0.0.1:5561 <-- disparity + depth -- GPU 推理服务
             |
             +-- /fast_foundationstereo/depth        sensor_msgs/Image, 32FC1, m
             +-- /fast_foundationstereo/disparity    stereo_msgs/DisparityImage, px
             +-- /fast_foundationstereo/camera_info  sensor_msgs/CameraInfo
             +-- /fast_foundationstereo/inference_ms std_msgs/Float32
             +-- /fast_foundationstereo/points       sensor_msgs/PointCloud2（可选）
```

5560/5561 特意避开了 FoundationPose 默认使用的 5555/5556 端口。两侧都采用较小的 ZMQ 队列并主动丢弃旧帧，GPU 来不及处理时不会不断累积延迟。

输出消息继承左红外图像的时间戳和 optical frame。深度仍处在左红外光学坐标系中，因此不需要发布新的 TF。

## 1. 确认 D405 话题

先在 ROS2 工作区启动新增的双目相机 launch：

```bash
cd /home/f1sh/DexHand/sfp_pick_ws
source install/setup.bash
ros2 launch camera d405_stereo.launch.py
```

`d405.launch.py` 只打开 RGB/Depth，不能提供 Fast FoundationStereo 所需的左右双目图像；`d405_stereo.launch.py` 显式打开 `enable_infra1` 和 `enable_infra2`，并设置 `depth_module.infra_profile:=640x480x30`。该专用 launch 关闭了硬件 Depth/Color、对齐和深度滤波，因为深度由 Fast FoundationStereo 自己计算。

另一个终端检查四个输入话题：

```bash
ros2 topic list | grep -E 'infra[12]/(image_rect_raw|camera_info)'
```

默认使用：

```text
/camera/camera/infra1/image_rect_raw
/camera/camera/infra2/image_rect_raw
/camera/camera/infra1/camera_info
/camera/camera/infra2/camera_info
```

Fast FoundationStereo 要求输入已经去畸变并完成双目极线校正，所以应使用 `image_rect_raw`，不要改成未校正的 `image_raw`。如果当前 `d405.launch.py` 没有发布左右红外流，需要在传给 RealSense `rs_launch.py` 的参数中开启 `enable_infra1:=true` 和 `enable_infra2:=true`；实际参数名以本机安装的 realsense2_camera 版本为准。

桥接节点默认从左右 `CameraInfo.P` 自动计算基线：

```text
baseline = abs(P_right[0,3] / P_right[0,0] - P_left[0,3] / P_left[0,0])
```

因此无需手填 D405 基线。如果驱动发出的 `P` 不包含平移量，可通过 launch 参数 `baseline_override:=0.018` 手工设置，数值必须以实际标定为准，不要直接照抄示例。

## 2. 构建宿主机 ROS2 包

ROS2 包保存在本工程的 `ros2/fast_foundationstereo_ros`。可以让现有工作区直接发现它，无需复制源码：

```bash
cd /home/f1sh/DexHand/sfp_pick_ws
source /opt/ros/humble/setup.bash
colcon build \
  --base-paths src /home/f1sh/DexHand/Fast-FoundationStereo/ros2 \
  --packages-select fast_foundationstereo_ros
source install/setup.bash
```

宿主机需要 `cv_bridge`、`message_filters`、`stereo_msgs`、NumPy 和 pyzmq。若 rosdep 尚未安装它们：

```bash
rosdep install --from-paths \
  /home/f1sh/DexHand/Fast-FoundationStereo/ros2/fast_foundationstereo_ros \
  --ignore-src -r -y
```

## 3. 启动宿主机桥接节点

```bash
source /home/f1sh/DexHand/sfp_pick_ws/install/setup.bash
ros2 launch fast_foundationstereo_ros d405_ffs.launch.py
```

若左右图像时间戳不是完全相同，可切换到近似同步：

```bash
ros2 launch fast_foundationstereo_ros d405_ffs.launch.py approximate_sync:=true
```

需要同时发布点云时：

```bash
ros2 launch fast_foundationstereo_ros d405_ffs.launch.py publish_point_cloud:=true
```

点云的序列化和带宽开销较大，实时控制只使用深度图时建议保持关闭。

开启点云后，消息包含标准 `x`、`y`、`z`、`rgb` 字段。桥接节点默认订阅 `/camera/camera/color/image_raw`，将时间戳最接近的 D405 彩色图与 FoundationStereo 深度匹配，并通过 sequence 缓存到对应的推理结果。若暂时没有匹配到彩色帧，则回退到左侧 `infra1` 灰度图，不会中断深度或点云发布。

## 4. 创建并复用持久化 `ffs` 容器

容器脚本使用 `--network=host`，ZMQ 可以直接通过 `127.0.0.1` 通信。工程挂载到 `/workspace/Fast-FoundationStereo`，源码修改会立刻反映到容器；通过 pip/uv 补装的包则保存在命名容器的可写层中。

首次创建：

```bash
cd /home/f1sh/DexHand/Fast-FoundationStereo
bash docker/1_create_container.sh
```

以后只需要唤醒并进入同一个容器：

```bash
bash docker/2_start_container.sh
```

兼容入口 `bash docker/run_container.sh` 会自动判断：容器不存在则首次创建，已经存在则直接启动。脚本不再执行 `docker rm -f ffs`，所以容器环境不会在每次启动时丢失。

首次创建会运行 `docker/3_prepare_python_env.sh`。它会检查 `torch==2.6.x`、匹配的 `torchvision` 和 `pyzmq`：若环境已经正确，不会下载任何内容；若使用旧版 `ffs:latest`（旧镜像可能被 `nvidia-modelopt` 意外升级到 Torch 2.13），则只在这个持久化容器中修复一次。后续启动不会重复下载。

更新后的 `docker/dockerfile` 已固定兼容的 Torch/TorchVision/XFormers，并移除了会导致 ABI 错配的非必要 `nvidia-modelopt[torch]`。重新构建镜像时不会再需要容器内修复：

```bash
docker build --network host -t ffs -f docker/dockerfile .
```

需要停止容器但保留全部环境时执行 `docker stop ffs`。只有手动执行 `docker rm ffs` 才会删除容器可写层。

进入容器后，本文所有“容器内执行”命令都默认先完成以下初始化：

```bash
cd /workspace/Fast-FoundationStereo
source /opt/conda/etc/profile.d/conda.sh
conda activate my
```

如果终端提示符已经带有 `(my)`，只需确认当前目录是
`/workspace/Fast-FoundationStereo`，不必重复激活环境。

## 5. 启动容器内推理服务

使用原生 PyTorch 权重启动（默认 4 次迭代以降低实时延迟）：

在宿主机执行：

```bash
cd /home/f1sh/DexHand/Fast-FoundationStereo
./docker/exec_zmq_server.sh \
  --model_file weights/23-36-37/model_best_bp2_serialize.pth \
  --valid_iters 4 \
  --max_disp 192 \
  --zmin 0.05 \
  --zmax 2.0
```

已经进入 `ffs` 容器时，直接执行 Python 服务：

```bash
cd /workspace/Fast-FoundationStereo
python scripts/zmq_stereo_server.py \
  --model_file weights/23-36-37/model_best_bp2_serialize.pth \
  --valid_iters 4 \
  --max_disp 192 \
  --zmin 0.05 \
  --zmax 2.0
```

服务启动时默认用 640x480 做一次 CUDA warmup，把模型编译/内核初始化延迟放在接收相机帧之前。若相机使用其他分辨率，可设置 `--warmup_width` 和 `--warmup_height`；设 `--warmup_runs 0` 可关闭。

如果容器内工程不在默认路径，可覆盖环境变量：

```bash
FFS_PROJECT_IN_CONTAINER=/实际/容器内路径 ./docker/exec_zmq_server.sh
```

若已经导出固定 640x480 的 TensorRT engine，可直接替换模型参数：

在宿主机执行：

```bash
./docker/exec_zmq_server.sh \
  --model_file /path/to/fast_foundationstereo.engine \
  --config_file /path/to/fast_foundationstereo.yaml \
  --zmax 2.0
```

已经进入 `ffs` 容器时执行：

```bash
cd /workspace/Fast-FoundationStereo
python scripts/zmq_stereo_server.py \
  --model_file /path/to/fast_foundationstereo.engine \
  --config_file /path/to/fast_foundationstereo.yaml \
  --zmax 2.0
```

`.pth`、`.onnx` 和 `.engine` 三种后端使用同一套 ZMQ/ROS2 接口。固定尺寸的 ONNX/TensorRT 输入会在容器内缩放，返回前再恢复到 D405 原始分辨率，并同步恢复视差的像素尺度。

### `max_disp` 如何选择

`max_disp` 是模型搜索的最大视差，单位是像素，不是最大深度。双目深度近似为：

```text
depth_m = fx_pixels * baseline_m / disparity_pixels
```

所以理论最近深度约为：

```text
z_near ≈ fx_pixels × baseline_m / max_disp
```

以 D405 常见的 `fx≈430 px`、`baseline≈0.018 m` 为例：

| `max_disp` | 理论最近深度 | 代价 |
|---:|---:|---|
| 192 | 约 4.0 cm | 推荐默认值，D405 约 7 cm 以上工作距离通常足够 |
| 256 | 约 3.0 cm | 更近目标，显存和计算量增加 |
| 384 | 约 2.0 cm | 仅近距离场景使用，速度明显下降 |

`max_disp` 越大，越能覆盖近处目标，但会增加显存、计算量和延迟；它不限制最远深度，最远范围由最小有效视差和服务端的 `--zmax` 决定。当前 `.pth` 服务建议：

```bash
--max_disp 192
```

使用 `--scale 0.5` 时，模型输入视差是在缩小后的图像上计算，服务端会在输出前恢复到原始图像的像素单位。使用 ONNX/TensorRT 时，`max_disp` 必须在导出阶段确定，运行时再改参数不会改变已经生成的 engine。

### TensorRT FP16 部署

TensorRT 适合固定分辨率的实时推理。下面命令在持久化 `ffs` 容器中执行一次即可；生成的 ONNX、YAML 和 engine 位于挂载的工程目录，之后直接复用。

#### 1. 导出 640×480 单模型 ONNX

从宿主机执行：

```bash
docker exec -w /workspace/Fast-FoundationStereo ffs \
  /opt/conda/envs/my/bin/python scripts/make_single_onnx.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --save_path output_single_onnx_d405 \
  --onnx_name fast_foundationstereo_d405 \
  --height 480 \
  --width 640 \
  --valid_iters 4 \
  --max_disp 192
```

已经进入 `ffs` 容器时执行：

```bash
cd /workspace/Fast-FoundationStereo
python scripts/make_single_onnx.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --save_path output_single_onnx_d405 \
  --onnx_name fast_foundationstereo_d405 \
  --height 480 \
  --width 640 \
  --valid_iters 4 \
  --max_disp 192
```

`height` 和 `width` 必须是 32 的倍数，并且建议与 `d405_stereo.launch.py` 的 640×480 输入保持一致。需要更近距离时，在这一步将 `--max_disp` 一起改成 256 或 384。

#### 2. 转换 FP16 ONNX 并构建 TensorRT engine

容器当前固定使用 TensorRT 10.11.0.33。下面仍先把 FP32 ONNX 显式转换成
FP16，再调用 `trtexec` 构建 engine。转换脚本会为 `GridSample`、`MatMul`、
`ConvTranspose` 等要求严格同类型的节点补齐 Cast。

从宿主机执行转换：

```bash
docker exec -w /workspace/Fast-FoundationStereo ffs \
  /opt/conda/envs/my/bin/python scripts/convert_onnx_fp16.py \
  --input output_single_onnx_d405/fast_foundationstereo_d405.onnx \
  --output output_single_onnx_d405/fast_foundationstereo_d405_fp16_trt10.onnx \
  --keep_io_types
```

已经进入 `ffs` 容器时，直接执行 Python 转换脚本：

```bash
cd /workspace/Fast-FoundationStereo
python scripts/convert_onnx_fp16.py \
  --input output_single_onnx_d405/fast_foundationstereo_d405.onnx \
  --output output_single_onnx_d405/fast_foundationstereo_d405_fp16_trt10.onnx \
  --keep_io_types
```

从宿主机执行 engine 构建：

```bash
docker exec -w /workspace/Fast-FoundationStereo ffs \
  /usr/src/tensorrt/bin/trtexec \
  --onnx=/workspace/Fast-FoundationStereo/output_single_onnx_d405/fast_foundationstereo_d405_fp16_trt10.onnx \
  --saveEngine=/workspace/Fast-FoundationStereo/output_single_onnx_d405/fast_foundationstereo_d405_fp16_trt10.engine \
  --memPoolSize=workspace:4096 \
  --skipInference
```

已经进入 `ffs` 容器时执行：

```bash
cd /workspace/Fast-FoundationStereo
/usr/src/tensorrt/bin/trtexec \
  --onnx=output_single_onnx_d405/fast_foundationstereo_d405_fp16_trt10.onnx \
  --saveEngine=output_single_onnx_d405/fast_foundationstereo_d405_fp16_trt10.engine \
  --memPoolSize=workspace:4096 \
  --skipInference
```

engine 与构建它的 GPU、CUDA、TensorRT 版本相关；更换机器或镜像后需要重新构建。若当前 TensorRT 版本不接受 `--memPoolSize`，可去掉该参数，或使用对应版本的 workspace 参数格式。

#### 3. 用 TensorRT engine 启动 ZMQ 服务

从宿主机执行：

```bash
cd /home/f1sh/DexHand/Fast-FoundationStereo
./docker/exec_zmq_server.sh \
  --model_file output_single_onnx_d405/fast_foundationstereo_d405_fp16_trt10.engine \
  --config_file output_single_onnx_d405/fast_foundationstereo_d405.yaml \
  --zmin 0.05 \
  --zmax 2.0 \
  --warmup_width 640 \
  --warmup_height 480
```

已经进入 `ffs` 容器时，直接运行 Python 服务：

```bash
cd /workspace/Fast-FoundationStereo
python scripts/zmq_stereo_server.py \
  --model_file output_single_onnx_d405/fast_foundationstereo_d405_fp16_trt10.engine \
  --config_file output_single_onnx_d405/fast_foundationstereo_d405.yaml \
  --zmin 0.05 \
  --zmax 2.0 \
  --warmup_width 640 \
  --warmup_height 480
```

TensorRT 后端的 `valid_iters` 和 `max_disp` 已经固化在 ONNX/engine 中；运行服务时的 `--zmin`、`--zmax` 仍然有效。固定 640×480、FP16、CUDA Graph 和关闭点云发布通常能获得最高帧率。可以继续用以下命令确认实际推理耗时：

```bash
ros2 topic echo /fast_foundationstereo/inference_ms
```

## 6. 验证 ROS2 输出

```bash
ros2 topic hz /fast_foundationstereo/depth
ros2 topic echo /fast_foundationstereo/inference_ms
ros2 topic info /fast_foundationstereo/disparity
```

RViz2 中可添加：

- `Image`：`/fast_foundationstereo/depth`
- `PointCloud2`：`/fast_foundationstereo/points`（启动时需开启）
- Fixed Frame：D405 左红外图像消息中的 frame，一般为 `camera_infra1_optical_frame`

深度定义为：

```text
depth_m = fx_pixels * baseline_m / disparity_pixels
```

无效、超出 `zmin/zmax`、以及在右图中没有可见对应点的像素输出为 `0.0 m`。

## 常见问题

### 桥接节点一直显示 `Waiting for both stereo CameraInfo topics`

确认实际 topic 名称并通过 launch 参数覆盖。示例：

```bash
ros2 launch fast_foundationstereo_ros d405_ffs.launch.py \
  left_camera_info_topic:=/your/left/camera_info \
  right_camera_info_topic:=/your/right/camera_info
```

### 有 CameraInfo，但报 baseline 无效

检查右相机投影矩阵：

```bash
ros2 topic echo --once /camera/camera/infra2/camera_info
```

正常校正后的右相机 `p[3]` 应包含非零水平平移。否则使用经过确认的 `baseline_override`。

### 容器显示等待图像，ROS2 端也没有报错

确认容器使用 host network，并检查 5560/5561 没有被其他进程占用：

```bash
ss -ltnp | grep -E ':5560|:5561'
```

### 推理帧率不足

依次尝试：

1. 使用 `--valid_iters 2` 或 `4`。
2. 原生 PyTorch 后端使用 `--scale 0.5`。
3. 导出 640x480 ONNX 并构建 FP16 TensorRT engine。
4. 固定输入尺寸并启用 `--useCudaGraph`。
5. 保持 `publish_point_cloud:=false`，只发布深度与视差；带颜色的点云会增加 CPU 序列化和 DDS 带宽开销。

缩放不会改变最终深度图尺寸；服务端会把视差转换回原始图像的像素单位后再计算深度。
