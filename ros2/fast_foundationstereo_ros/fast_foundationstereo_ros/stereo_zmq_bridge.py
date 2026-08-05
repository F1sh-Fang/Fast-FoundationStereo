#!/usr/bin/env python3
"""Bridge rectified ROS 2 stereo images to a ROS-free ZMQ GPU server."""

import copy
import json
import threading
from collections import OrderedDict

import cv2
import numpy as np
import zmq

import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber, TimeSynchronizer
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Float32
from stereo_msgs.msg import DisparityImage


def calibration_from_camera_info(left_info, right_info, baseline_override=0.0):
    """Return rectified intrinsics and baseline from a stereo CameraInfo pair."""
    left_p = np.asarray(left_info.p, dtype=np.float64).reshape(3, 4)
    right_p = np.asarray(right_info.p, dtype=np.float64).reshape(3, 4)
    if abs(left_p[0, 0]) > 1e-9 and abs(left_p[1, 1]) > 1e-9:
        fx, fy = left_p[0, 0], left_p[1, 1]
        cx, cy = left_p[0, 2], left_p[1, 2]
    else:
        left_k = np.asarray(left_info.k, dtype=np.float64).reshape(3, 3)
        fx, fy = left_k[0, 0], left_k[1, 1]
        cx, cy = left_k[0, 2], left_k[1, 2]

    if baseline_override > 0.0:
        baseline = float(baseline_override)
    else:
        if abs(left_p[0, 0]) <= 1e-9 or abs(right_p[0, 0]) <= 1e-9:
            raise ValueError(
                "CameraInfo.P does not contain valid stereo projection matrices; "
                "set baseline_override"
            )
        left_tx = left_p[0, 3] / left_p[0, 0]
        right_tx = right_p[0, 3] / right_p[0, 0]
        baseline = abs(right_tx - left_tx)

    if fx <= 0.0 or fy <= 0.0 or baseline <= 0.0:
        raise ValueError(
            f"Invalid stereo calibration: fx={fx}, fy={fy}, baseline={baseline}"
        )
    return float(fx), float(fy), float(cx), float(cy), baseline


def _to_rgb_uint8(image, encoding):
    """Convert a ROS image array to RGB uint8 for point-cloud coloring."""
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] >= 3:
        if encoding.lower() in ("bgr8", "bgra8"):
            image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
        else:
            image = image[..., :3]
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def make_point_cloud(depth, fx, fy, cx, cy, header, color=None):
    """Create an organized XYZRGB PointCloud2 from metric depth."""
    height, width = depth.shape
    v, u = np.indices((height, width), dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    xyz = np.empty((height, width, 3), dtype=np.float32)
    xyz[..., 2] = depth
    xyz[..., 0] = (u - cx) * depth / fx
    xyz[..., 1] = (v - cy) * depth / fy
    xyz[~valid] = np.nan

    if color is None:
        color = np.zeros((height, width, 3), dtype=np.uint8)
    if color.shape[:2] != (height, width):
        raise ValueError(
            f"Point-cloud color shape {color.shape} does not match depth {depth.shape}"
        )
    if color.ndim == 2:
        color = np.repeat(color[..., None], 3, axis=2)
    packed_rgb = (
        (color[..., 0].astype(np.uint32) << 16)
        | (color[..., 1].astype(np.uint32) << 8)
        | color[..., 2].astype(np.uint32)
    )

    point_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("rgb", "<f4"),
        ]
    )
    points = np.empty((height, width), dtype=point_dtype)
    points["x"] = xyz[..., 0]
    points["y"] = xyz[..., 1]
    points["z"] = xyz[..., 2]
    points["rgb"] = packed_rgb.view(np.float32)
    points["rgb"][~valid] = 0.0

    message = PointCloud2()
    message.header = header
    message.height = height
    message.width = width
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 16
    message.row_step = message.point_step * width
    message.data = np.ascontiguousarray(points).tobytes()
    message.is_dense = False
    return message


class StereoZmqBridge(Node):
    def __init__(self):
        super().__init__("fast_foundationstereo_zmq_bridge")
        self._declare_parameters()
        self.cv_bridge = CvBridge()
        self.sequence = 0
        self.left_info = None
        self.right_info = None
        self._warned_missing_info = False
        self._calibration_logged = False
        self._warned_missing_color = False
        self._left_color_cache = OrderedDict()
        self._color_cache_size = 8
        self._socket_lock = threading.Lock()

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            CameraInfo,
            self._parameter("left_camera_info_topic"),
            self._left_info_callback,
            sensor_qos,
        )
        self.create_subscription(
            CameraInfo,
            self._parameter("right_camera_info_topic"),
            self._right_info_callback,
            sensor_qos,
        )

        left_sub = Subscriber(
            self, Image, self._parameter("left_image_topic"), qos_profile=sensor_qos
        )
        right_sub = Subscriber(
            self, Image, self._parameter("right_image_topic"), qos_profile=sensor_qos
        )
        queue_size = int(self._parameter("sync_queue_size"))
        if self._parameter("approximate_sync"):
            self.synchronizer = ApproximateTimeSynchronizer(
                [left_sub, right_sub],
                queue_size=queue_size,
                slop=float(self._parameter("sync_slop")),
            )
        else:
            self.synchronizer = TimeSynchronizer(
                [left_sub, right_sub], queue_size=queue_size
            )
        self.synchronizer.registerCallback(self._stereo_callback)

        context = zmq.Context.instance()
        self.image_socket = context.socket(zmq.PUB)
        self.image_socket.setsockopt(zmq.SNDHWM, 2)
        self.image_socket.setsockopt(zmq.LINGER, 0)
        self.image_socket.bind(self._parameter("image_endpoint"))
        self.result_socket = context.socket(zmq.SUB)
        self.result_socket.setsockopt(zmq.RCVHWM, 2)
        self.result_socket.setsockopt(zmq.LINGER, 0)
        self.result_socket.connect(self._parameter("result_endpoint"))
        self.result_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.depth_publisher = self.create_publisher(
            Image, self._parameter("depth_topic"), 10
        )
        self.disparity_publisher = self.create_publisher(
            DisparityImage, self._parameter("disparity_topic"), 10
        )
        self.camera_info_publisher = self.create_publisher(
            CameraInfo, self._parameter("output_camera_info_topic"), 10
        )
        self.latency_publisher = self.create_publisher(
            Float32, self._parameter("inference_time_topic"), 10
        )
        self.publish_point_cloud = bool(self._parameter("publish_point_cloud"))
        self.point_cloud_publisher = None
        if self.publish_point_cloud:
            self.point_cloud_publisher = self.create_publisher(
                PointCloud2, self._parameter("point_cloud_topic"), 2
            )

        poll_period = float(self._parameter("result_poll_period"))
        self.create_timer(poll_period, self._poll_results)
        self.get_logger().info(
            f"ZMQ bridge ready: {self._parameter('image_endpoint')} -> Docker, "
            f"Docker -> {self._parameter('result_endpoint')}"
        )

    def _declare_parameters(self):
        defaults = {
            "left_image_topic": "/camera/camera/infra1/image_rect_raw",
            "right_image_topic": "/camera/camera/infra2/image_rect_raw",
            "left_camera_info_topic": "/camera/camera/infra1/camera_info",
            "right_camera_info_topic": "/camera/camera/infra2/camera_info",
            "image_endpoint": "tcp://0.0.0.0:5560",
            "result_endpoint": "tcp://127.0.0.1:5561",
            "depth_topic": "/fast_foundationstereo/depth",
            "disparity_topic": "/fast_foundationstereo/disparity",
            "output_camera_info_topic": "/fast_foundationstereo/camera_info",
            "point_cloud_topic": "/fast_foundationstereo/points",
            "inference_time_topic": "/fast_foundationstereo/inference_ms",
            "publish_point_cloud": False,
            "baseline_override": 0.0,
            "approximate_sync": False,
            "sync_queue_size": 5,
            "sync_slop": 0.01,
            "result_poll_period": 0.002,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _parameter(self, name):
        return self.get_parameter(name).value

    def _left_info_callback(self, message):
        self.left_info = message

    def _right_info_callback(self, message):
        self.right_info = message

    def _stereo_callback(self, left_message, right_message):
        if self.left_info is None or self.right_info is None:
            if not self._warned_missing_info:
                self.get_logger().warning("Waiting for both stereo CameraInfo topics")
                self._warned_missing_info = True
            return

        try:
            fx, fy, cx, cy, baseline = calibration_from_camera_info(
                self.left_info,
                self.right_info,
                float(self._parameter("baseline_override")),
            )
            left_raw = self.cv_bridge.imgmsg_to_cv2(
                left_message, desired_encoding="passthrough"
            )
            right_raw = self.cv_bridge.imgmsg_to_cv2(
                right_message, desired_encoding="passthrough"
            )
            left_color = _to_rgb_uint8(left_raw, left_message.encoding)
            right_color = _to_rgb_uint8(right_raw, right_message.encoding)
            left = cv2.cvtColor(left_color, cv2.COLOR_RGB2GRAY)
            right = cv2.cvtColor(right_color, cv2.COLOR_RGB2GRAY)
            if left.shape != right.shape:
                raise ValueError(f"Stereo image shapes differ: {left.shape} vs {right.shape}")
        except Exception as exc:  # cv_bridge exceptions vary by ROS distribution
            self.get_logger().error(f"Cannot prepare stereo frame: {exc}")
            return

        if not self._calibration_logged:
            self.get_logger().info(
                f"Stereo calibration: {left.shape[1]}x{left.shape[0]}, "
                f"fx={fx:.3f}, baseline={baseline:.6f} m, "
                f"frame={left_message.header.frame_id}"
            )
            self._calibration_logged = True

        metadata = {
            "stamp_sec": left_message.header.stamp.sec,
            "stamp_nsec": left_message.header.stamp.nanosec,
            "sequence": self.sequence,
            "frame_id": left_message.header.frame_id,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "baseline": baseline,
        }
        sequence = self.sequence
        self.sequence += 1
        if self.publish_point_cloud:
            self._left_color_cache[sequence] = left_color
            while len(self._left_color_cache) > self._color_cache_size:
                self._left_color_cache.popitem(last=False)
        try:
            with self._socket_lock:
                self.image_socket.send_multipart(
                    [
                        json.dumps(metadata).encode("utf-8"),
                        np.ascontiguousarray(left).tobytes(),
                        np.ascontiguousarray(right).tobytes(),
                    ],
                    flags=zmq.NOBLOCK,
                )
        except zmq.Again:
            pass

    def _poll_results(self):
        latest = None
        while True:
            try:
                latest = self.result_socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if latest is None:
            return
        try:
            if len(latest) != 3:
                raise ValueError(f"Expected 3 result parts, received {len(latest)}")
            metadata = json.loads(latest[0].decode("utf-8"))
            shape = tuple(metadata["shape"])
            dtype = np.dtype(metadata["dtype"])
            disparity = np.frombuffer(latest[1], dtype=dtype).reshape(shape).copy()
            depth = np.frombuffer(latest[2], dtype=dtype).reshape(shape).copy()
            color = self._left_color_cache.pop(
                int(metadata.get("sequence", -1)), None
            )
            self._publish_result(metadata, disparity, depth, color)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"Dropped malformed inference result: {exc}")

    def _publish_result(self, metadata, disparity, depth, color=None):
        stamp = rclpy.time.Time(
            seconds=int(metadata.get("stamp_sec", 0)),
            nanoseconds=int(metadata.get("stamp_nsec", 0)),
        ).to_msg()
        frame_id = metadata.get("frame_id") or (
            self.left_info.header.frame_id if self.left_info else ""
        )

        depth_message = self.cv_bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_message.header.stamp = stamp
        depth_message.header.frame_id = frame_id
        self.depth_publisher.publish(depth_message)

        disparity_image = self.cv_bridge.cv2_to_imgmsg(
            disparity, encoding="32FC1"
        )
        disparity_image.header = depth_message.header
        disparity_message = DisparityImage()
        disparity_message.header = depth_message.header
        disparity_message.image = disparity_image
        disparity_message.f = float(metadata["fx"])
        disparity_message.t = float(metadata["baseline"])
        disparity_message.min_disparity = float(metadata.get("min_disparity", 0.0))
        disparity_message.max_disparity = float(metadata.get("max_disparity", 0.0))
        disparity_message.delta_d = 0.0
        self.disparity_publisher.publish(disparity_message)

        if self.left_info is not None:
            info_message = copy.deepcopy(self.left_info)
            info_message.header = depth_message.header
            self.camera_info_publisher.publish(info_message)

        latency = Float32()
        latency.data = float(metadata.get("inference_ms", 0.0))
        self.latency_publisher.publish(latency)

        if self.point_cloud_publisher is not None:
            if color is None and not self._warned_missing_color:
                self.get_logger().warning(
                    "No matching left image cached for PointCloud2 color; "
                    "publishing black RGB values for this frame"
                )
                self._warned_missing_color = True
            cloud = make_point_cloud(
                depth,
                float(metadata["fx"]),
                float(metadata["fy"]),
                float(metadata["cx"]),
                float(metadata["cy"]),
                depth_message.header,
                color=color,
            )
            self.point_cloud_publisher.publish(cloud)

    def destroy_node(self):
        with self._socket_lock:
            self.image_socket.close(linger=0)
        self.result_socket.close(linger=0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StereoZmqBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
