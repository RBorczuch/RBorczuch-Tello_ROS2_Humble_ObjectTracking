from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tello_ros2_object_tracking'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='A simplified ROS 2 package for manual control of a DJI Tello drone.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tello_driver = tello_ros2_object_tracking.tello_driver_node:main',
            'video_processor = tello_ros2_object_tracking.video_processor_node:main',
            'tello_controller = tello_ros2_object_tracking.tello_controller_node:main',
            'status_display = tello_ros2_object_tracking.status_display_node:main',
        ],
    },
)