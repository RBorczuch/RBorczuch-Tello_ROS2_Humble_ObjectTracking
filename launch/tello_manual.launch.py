from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """Generuje opis startowy do uruchomienia węzłów do manualnego sterowania Tello."""
    return LaunchDescription([
        # Węzeł sterownika - komunikuje się z dronem
        Node(
            package='tello_ros2_object_tracking',
            executable='tello_driver',
            name='tello_driver',
            output='screen',
            emulate_tty=True,
        ),
        # Węzeł wyświetlający obraz z kamery
        Node(
            package='tello_ros2_object_tracking',
            executable='video_processor',
            name='video_display',
            output='screen',
            emulate_tty=True,
        ),
        # PONIŻSZY WĘZEŁ ZOSTAŁ ZAKOMENTOWANY
        # # Węzeł kontrolera - odczytuje klawiaturę do sterowania
        # Node(
        #     package='tello_ros2_object_tracking',
        #     executable='tello_controller',
        #     name='tello_controller',
        #     output='screen',
        #     emulate_tty=True,
        # ),
        # Węzeł wyświetlający status i telemetrię w konsoli
        Node(
            package='tello_ros2_object_tracking',
            executable='status_display',
            name='status_display',
            output='screen',
            emulate_tty=True,
        ),
    ])