from setuptools import find_packages, setup

package_name = 'anygrasp_ros2_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jairlab',
    maintainer_email='jairlab@todo.todo',
    description='ROS 2 bridge for running AnyGrasp detection on RealSense aligned RGB-D frames.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'anygrasp_detection_node = anygrasp_ros2_bridge.anygrasp_detection_node:main',
            'anygrasp_pose_to_base_node = anygrasp_ros2_bridge.anygrasp_pose_to_base_node:main',
            'anygrasp_hover_test_node = anygrasp_ros2_bridge.anygrasp_hover_test_node:main',
            'anygrasp_candidates_to_base_node = anygrasp_ros2_bridge.anygrasp_candidates_to_base_node:main',
        ],
    },
)
