#!/usr/bin/env python3
"""ZMQ inference server for Fast FoundationStereo.

This process is intentionally ROS-free and is meant to run inside the GPU
container.  A ROS 2 bridge on the host publishes rectified stereo frames over
ZMQ; this server returns disparity (pixels) and metric depth (metres).
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
import zmq
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_logging_format, set_seed


def _as_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a mono/RGB/RGBA input into contiguous RGB uint8."""
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] >= 3:
        image = image[..., :3]
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _resize_disparity_to_input(
    disparity: np.ndarray, input_width: int, input_height: int
) -> np.ndarray:
    """Resize disparity and restore its horizontal pixel units."""
    infer_height, infer_width = disparity.shape
    if (infer_width, infer_height) == (input_width, input_height):
        return disparity.astype(np.float32, copy=False)
    scale_x = infer_width / float(input_width)
    resized = cv2.resize(
        disparity, (input_width, input_height), interpolation=cv2.INTER_LINEAR
    )
    return (resized / scale_x).astype(np.float32, copy=False)


class NativeTorchRunner:
    def __init__(self, args):
        cfg_path = Path(args.model_file).resolve().parent / "cfg.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Model config not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream) or {}

        logging.info("Loading PyTorch model: %s", args.model_file)
        self.model = torch.load(
            args.model_file, map_location="cpu", weights_only=False
        )
        self.model.args.valid_iters = args.valid_iters
        self.model.args.max_disp = args.max_disp
        self.model.args.mixed_precision = cfg.get("mixed_precision", True)
        self.model.cuda().eval()
        self.scale = args.scale
        self.max_disp = args.max_disp

    @torch.inference_mode()
    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if self.scale != 1.0:
            left = cv2.resize(
                left, dsize=None, fx=self.scale, fy=self.scale,
                interpolation=cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_LINEAR,
            )
            right = cv2.resize(
                right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR
            )

        height, width = left.shape[:2]
        left_tensor = (
            torch.as_tensor(left, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
        )
        right_tensor = (
            torch.as_tensor(right, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
        )
        padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
        with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
            disparity = self.model.forward(
                left_tensor,
                right_tensor,
                iters=self.model.args.valid_iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )
        disparity = padder.unpad(disparity.float())
        return (
            disparity.detach().cpu().numpy().reshape(height, width).clip(0, None)
        )


class ExportedModelRunner:
    def __init__(self, args):
        from scripts.run_demo_single_trt import (
            OnnxRuntimeRunner,
            SingleEngineTrtRunner,
            normalize_imagenet,
            resolve_config,
        )

        cfg_path = args.config_file or resolve_config(args.model_file)
        with open(cfg_path, "r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream) or {}
        self.target_height, self.target_width = map(int, cfg["image_size"])
        self.max_disp = int(cfg.get("max_disp", args.max_disp))
        self.normalize = normalize_imagenet
        if args.model_file.endswith(".onnx"):
            self.runner = OnnxRuntimeRunner(args.model_file)
        else:
            self.runner = SingleEngineTrtRunner(args.model_file)
        logging.info(
            "Loaded exported model at fixed resolution %dx%d",
            self.target_width,
            self.target_height,
        )

    @torch.inference_mode()
    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if left.shape[:2] != (self.target_height, self.target_width):
            left = cv2.resize(
                left, (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR,
            )
            right = cv2.resize(
                right, (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        left_tensor = (
            torch.as_tensor(self.normalize(left), device="cuda")
            .float()
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        right_tensor = (
            torch.as_tensor(self.normalize(right), device="cuda")
            .float()
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        outputs = self.runner(
            {"left_image": left_tensor, "right_image": right_tensor}
        )
        return (
            outputs["disparity"]
            .float()
            .detach()
            .cpu()
            .numpy()
            .reshape(self.target_height, self.target_width)
            .clip(0, None)
        )


class TwoEngineTrtRunner:
    """Run feature/post TensorRT engines with the Triton GWC kernel between."""

    def __init__(self, args):
        from core.foundation_stereo import TrtRunner

        engine_dir = Path(args.two_engine_dir).resolve()
        feature_engine = engine_dir / "feature_runner.engine"
        post_engine = engine_dir / "post_runner.engine"
        cfg_path = (
            Path(args.config_file).resolve()
            if args.config_file
            else engine_dir / "onnx.yaml"
        )
        for path in (feature_engine, post_engine, cfg_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        with cfg_path.open("r", encoding="utf-8") as stream:
            cfg = OmegaConf.create(yaml.safe_load(stream) or {})
        self.target_height, self.target_width = map(int, cfg.image_size)
        self.max_disp = int(cfg.max_disp)
        self.runner = TrtRunner(
            cfg, str(feature_engine), str(post_engine),
            use_cuda_graph=args.cuda_graph,
        )
        logging.info(
            "Loaded two-engine TensorRT model at fixed resolution %dx%d%s",
            self.target_width,
            self.target_height,
            " with CUDA Graph" if args.cuda_graph else "",
        )

    @torch.inference_mode()
    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if left.shape[:2] != (self.target_height, self.target_width):
            left = cv2.resize(
                left, (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR,
            )
            right = cv2.resize(
                right, (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        left_tensor = (
            torch.as_tensor(left, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        right_tensor = (
            torch.as_tensor(right, device="cuda", dtype=torch.float32)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        disparity = self.runner(left_tensor, right_tensor)
        return (
            disparity.float().detach().cpu().numpy()
            .reshape(self.target_height, self.target_width)
            .clip(0, None)
        )


def make_runner(args):
    if args.two_engine_dir:
        return TwoEngineTrtRunner(args)
    suffix = Path(args.model_file).suffix.lower()
    if suffix == ".pth":
        return NativeTorchRunner(args)
    if suffix in (".onnx", ".engine"):
        return ExportedModelRunner(args)
    raise ValueError(
        "use --two_engine_dir, or provide a --model_file ending in "
        ".pth, .onnx, or .engine"
    )


def recv_latest(socket):
    """Receive one request and drain queued frames, keeping the newest."""
    parts = socket.recv_multipart()
    while True:
        try:
            parts = socket.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
    if len(parts) != 3:
        raise ValueError(f"Expected 3 message parts, received {len(parts)}")
    metadata = json.loads(parts[0].decode("utf-8"))
    left = np.frombuffer(parts[1], dtype=np.dtype(metadata["left_dtype"]))
    right = np.frombuffer(parts[2], dtype=np.dtype(metadata["right_dtype"]))
    left = left.reshape(metadata["left_shape"]).copy()
    right = right.reshape(metadata["right_shape"]).copy()
    return metadata, left, right


def disparity_to_depth(disparity, fx, baseline, zmin, zmax, remove_invisible):
    disparity = disparity.astype(np.float32, copy=True)
    valid = np.isfinite(disparity) & (disparity > 1e-6)
    if remove_invisible:
        x_coordinates = np.arange(disparity.shape[1], dtype=np.float32)[None, :]
        valid &= (x_coordinates - disparity) >= 0

    depth = np.zeros_like(disparity, dtype=np.float32)
    depth[valid] = float(fx) * float(baseline) / disparity[valid]
    valid &= (depth >= zmin) & (depth <= zmax) & np.isfinite(depth)
    depth[~valid] = 0.0
    disparity[~valid] = 0.0
    return disparity, depth


def parse_args():
    parser = argparse.ArgumentParser(
        description="ROS-free Fast FoundationStereo ZMQ inference server"
    )
    parser.add_argument(
        "--model_file",
        default=str(ROOT / "weights/23-36-37/model_best_bp2_serialize.pth"),
        help="Serialized .pth model, exported .onnx, or TensorRT .engine",
    )
    parser.add_argument(
        "--config_file", default="", help="Config YAML for ONNX/TensorRT"
    )
    parser.add_argument(
        "--two_engine_dir",
        default="output_two_engine_d405",
        help=(
            "Directory containing feature_runner.engine, post_runner.engine, "
            "and onnx.yaml; takes precedence over --model_file"
        ),
    )
    parser.add_argument(
        "--cuda_graph", action="store_true",
        help="Capture and replay the fixed-shape two-engine GPU pipeline",
    )
    parser.add_argument("--host_ip", default="127.0.0.1")
    parser.add_argument("--image_port", type=int, default=5560)
    parser.add_argument("--result_port", type=int, default=5561)
    parser.add_argument("--valid_iters", type=int, default=4)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--zmin", type=float, default=0.05)
    parser.add_argument("--zmax", type=float, default=3.0)
    parser.add_argument("--warmup_width", type=int, default=640)
    parser.add_argument("--warmup_height", type=int, default=480)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument(
        "--keep_invisible", action="store_true",
        help="Keep pixels whose correspondence falls outside the right image",
    )
    parser.add_argument("--log_every", type=int, default=30)
    args = parser.parse_args()
    args.cuda_graph = bool(args.two_engine_dir)

    return args


def main():
    args = parse_args()
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the container")
    if not os.path.isfile(args.model_file):
        raise FileNotFoundError(args.model_file)
    if args.scale <= 0:
        raise ValueError("--scale must be positive")

    runner = make_runner(args)
    if args.warmup_runs > 0:
        if args.warmup_width <= 0 or args.warmup_height <= 0:
            raise ValueError("--warmup_width and --warmup_height must be positive")
        logging.info(
            "Warming up CUDA at %dx%d for %d run(s)...",
            args.warmup_width,
            args.warmup_height,
            args.warmup_runs,
        )
        warmup_image = np.zeros(
            (args.warmup_height, args.warmup_width, 3), dtype=np.uint8
        )
        warmup_started = time.perf_counter()
        for _ in range(args.warmup_runs):
            runner(warmup_image, warmup_image)
        torch.cuda.synchronize()
        logging.info(
            "CUDA warmup finished in %.1f ms",
            (time.perf_counter() - warmup_started) * 1000.0,
        )
    context = zmq.Context.instance()
    image_socket = context.socket(zmq.SUB)
    image_socket.setsockopt(zmq.RCVHWM, 2)
    image_socket.connect(f"tcp://{args.host_ip}:{args.image_port}")
    image_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    result_socket = context.socket(zmq.PUB)
    result_socket.setsockopt(zmq.SNDHWM, 2)
    result_socket.bind(f"tcp://0.0.0.0:{args.result_port}")
    logging.info(
        "Input tcp://%s:%d, result tcp://0.0.0.0:%d",
        args.host_ip,
        args.image_port,
        args.result_port,
    )
    logging.info("Waiting for rectified stereo frames...")

    frame_count = 0
    total_inference_ms = 0.0
    try:
        while True:
            try:
                metadata, left_raw, right_raw = recv_latest(image_socket)
                if left_raw.shape[:2] != right_raw.shape[:2]:
                    raise ValueError(
                        f"Stereo sizes differ: {left_raw.shape} vs {right_raw.shape}"
                    )
                fx = float(metadata["fx"])
                baseline = float(metadata["baseline"])
                if fx <= 0 or baseline <= 0:
                    raise ValueError(f"Invalid calibration fx={fx}, baseline={baseline}")

                input_height, input_width = left_raw.shape[:2]
                left = _as_rgb(left_raw)
                right = _as_rgb(right_raw)
                started = time.perf_counter()
                disparity = runner(left, right)
                torch.cuda.synchronize()
                inference_ms = (time.perf_counter() - started) * 1000.0
                disparity = _resize_disparity_to_input(
                    disparity, input_width, input_height
                )
                disparity, depth = disparity_to_depth(
                    disparity,
                    fx,
                    baseline,
                    args.zmin,
                    args.zmax,
                    not args.keep_invisible,
                )

                response = {
                    "stamp_sec": metadata.get("stamp_sec", 0),
                    "stamp_nsec": metadata.get("stamp_nsec", 0),
                    "sequence": metadata.get("sequence", 0),
                    "frame_id": metadata.get("frame_id", ""),
                    "shape": list(disparity.shape),
                    "dtype": str(disparity.dtype),
                    "fx": fx,
                    "fy": float(metadata["fy"]),
                    "cx": float(metadata["cx"]),
                    "cy": float(metadata["cy"]),
                    "baseline": baseline,
                    "min_disparity": 0.0,
                    "max_disparity": float(runner.max_disp),
                    "inference_ms": inference_ms,
                }
                try:
                    result_socket.send_multipart(
                        [
                            json.dumps(response).encode("utf-8"),
                            np.ascontiguousarray(disparity).tobytes(),
                            np.ascontiguousarray(depth).tobytes(),
                        ],
                        flags=zmq.NOBLOCK,
                    )
                except zmq.Again:
                    pass

                frame_count += 1
                total_inference_ms += inference_ms
                if args.log_every > 0 and frame_count % args.log_every == 0:
                    average_ms = total_inference_ms / frame_count
                    logging.info(
                        "frames=%d inference=%.1f ms average=%.1f ms (%.1f FPS)",
                        frame_count,
                        inference_ms,
                        average_ms,
                        1000.0 / average_ms,
                    )
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                logging.warning("Dropped malformed frame: %s", exc)
    except KeyboardInterrupt:
        pass
    finally:
        image_socket.close(linger=0)
        result_socket.close(linger=0)


if __name__ == "__main__":
    main()
