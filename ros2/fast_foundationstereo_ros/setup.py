from glob import glob
import os

from setuptools import find_packages, setup


package_name = "fast_foundationstereo_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="f1sh",
    maintainer_email="1163981636@qq.com",
    description="ROS 2/ZMQ bridge for containerized Fast FoundationStereo inference",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "stereo_zmq_bridge = fast_foundationstereo_ros.stereo_zmq_bridge:main",
        ],
    },
)
