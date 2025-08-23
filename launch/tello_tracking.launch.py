from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tello_ros2_object_tracking',
            executable='tello_driver',
            name='tello_driver',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='tello_ros2_object_tracking',
            executable='video_processor',
            name='video_processor',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='tello_ros2_object_tracking',
            executable='tello_controller',
            name='tello_controller',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='tello_ros2_object_tracking',
            executable='status_display',
            name='status_display',
            output='screen',
            emulate_tty=True,
        ),
    ])