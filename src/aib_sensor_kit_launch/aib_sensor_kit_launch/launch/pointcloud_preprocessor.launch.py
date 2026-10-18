# Copyright 2020 Tier IV, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Isaac LiDAR Pointcloud Preprocessor Launch

For the Isaac ego with a single LiDAR, we use a passthrough filter instead of
the concatenation filter (which requires 2+ LiDARs). The passthrough filter relays
the single LiDAR input to the concatenated output topic used by downstream Autoware
perception modules.
"""

import launch
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.actions import SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def launch_setup(context, *args, **kwargs):
    # Use passthrough filter as a relay for single LiDAR
    # This relays /sensing/lidar/top/pointcloud_before_sync to concatenated/pointcloud
    passthrough_component = ComposableNode(
        package="autoware_pointcloud_preprocessor",
        plugin="autoware::pointcloud_preprocessor::PassThroughFilterComponent",
        name="passthrough_filter",
        remappings=[
            ("input", "/sensing/lidar/top/pointcloud_before_sync"),
            ("output", "concatenated/pointcloud"),
        ],
        parameters=[
            {
                # Output frame for coordinate transform
                "output_frame": LaunchConfiguration("base_frame"),
                # Passthrough filter parameters - wide range to pass all points
                "filter_field_name": "z",
                "filter_limit_min": -1000.0,
                "filter_limit_max": 1000.0,
                "filter_limit_negative": False,
            }
        ],
        extra_arguments=[{"use_intra_process_comms": LaunchConfiguration("use_intra_process")}],
    )

    # Load passthrough filter
    passthrough_loader = LoadComposableNodes(
        composable_node_descriptions=[passthrough_component],
        target_container=LaunchConfiguration("pointcloud_container_name"),
        condition=IfCondition(LaunchConfiguration("use_concat_filter")),
    )

    return [passthrough_loader]


def generate_launch_description():

    launch_arguments = []

    def add_launch_arg(name: str, default_value=None):
        launch_arguments.append(DeclareLaunchArgument(name, default_value=default_value))

    add_launch_arg("base_frame", "base_link")
    add_launch_arg("use_multithread", "False")
    add_launch_arg("use_intra_process", "False")
    add_launch_arg("pointcloud_container_name", "pointcloud_container")
    add_launch_arg("use_concat_filter", "true")  # Use this to enable/disable the relay

    set_container_executable = SetLaunchConfiguration(
        "container_executable",
        "component_container",
        condition=UnlessCondition(LaunchConfiguration("use_multithread")),
    )

    set_container_mt_executable = SetLaunchConfiguration(
        "container_executable",
        "component_container_mt",
        condition=IfCondition(LaunchConfiguration("use_multithread")),
    )

    return launch.LaunchDescription(
        launch_arguments
        + [set_container_executable, set_container_mt_executable]
        + [OpaqueFunction(function=launch_setup)]
    )
