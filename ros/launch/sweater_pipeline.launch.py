"""from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

VENV_PYTHON = ("/home/gianluca/ros2_ws/src/coordinate_converter/"
               "coordinate_converter/rete/.venv/bin/python")


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("venv_python", default_value=VENV_PYTHON),
        DeclareLaunchArgument("home_config", default_value="finale"),

        # 1) home (Python di sistema)
        Node(package="coordinate_converter", executable="home_joint_final",
             arguments=[LaunchConfiguration("home_config")], output="screen"),

        # 2) foto + inferenza (SOLO questo usa il venv)
        ExecuteProcess(
            cmd=[LaunchConfiguration("venv_python"), "-m",
                 "coordinate_converter.vision_node"],
            output="screen"),

        # 3) traiettoria (Python di sistema)
        Node(package="coordinate_converter", executable="spline_separata_finale",
             output="screen"),
    ])"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

RETE_DIR = ("/home/gianluca/ros2_ws/src/coordinate_converter/"
            "coordinate_converter/rete")
VENV_PYTHON = os.path.join(RETE_DIR, ".venv", "bin", "python")
# TEST_RGB e TEST_DEPTH sono immagini di test prese dal dataset "dataset_rgbd_maglioncino" e non sono da usare nel caso reale,
# serviva solo a testare il funzionamento della pipeline senza dover fare una foto reale ogni volta.
TEST_RGB = os.path.join(RETE_DIR, "dataset_rgbd_maglioncino",
                        "20260925_121440_623656_rgb.png")
TEST_DEPTH = os.path.join(RETE_DIR, "dataset_rgbd_maglioncino",
                          "20260925_121440_623656_depth.png")


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("venv_python", default_value=VENV_PYTHON),
        DeclareLaunchArgument("home_config", default_value="finale"), #finale è la configurazione della home_position, dal file home_joint_final_launcher.py si può cambiare
        DeclareLaunchArgument("rgb_path", default_value=TEST_RGB),
        DeclareLaunchArgument("depth_path", default_value=TEST_DEPTH),

        # 1) home
        Node(package="coordinate_converter", executable="home_joint_final_launcher",
             arguments=[LaunchConfiguration("home_config")], output="screen"),

        # 2) inferenza su immagine di test, c'è bisogno anche di caricare il venv perché la rete è stata addestrata 
        # con PyTorch e quindi bisogna usare il Python del venv per eseguire il modulo vision_node.py
        ExecuteProcess(
            cmd=[LaunchConfiguration("venv_python"), "-m",
                 "coordinate_converter.vision_node",
                 "--ros-args",
                 "-p", ["rgb_path:=", LaunchConfiguration("rgb_path")],
                 "-p", ["depth_path:=", LaunchConfiguration("depth_path")],
                 "-p", "settle_time:=0.0"],
            output="screen"),
          # TEST_RGB e TEST_DEPTH sono immagini di test prese dal dataset "dataset_rgbd_maglioncino" e non sono da 
          # usare nel caso reale, serviva solo a testare il funzionamento della pipeline senza dover fare una foto 
          # reale ogni volta.

        # 3) traiettoria (spline)
        Node(package="coordinate_converter", executable="spline_separata_finale_launcher",
             output="screen"),
    ])