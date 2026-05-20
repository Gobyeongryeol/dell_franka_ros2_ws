from setuptools import setup

package_name = 'yolo_webcam_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'google-genai', 'python-dotenv'],
    zip_safe=True,
    maintainer='lab',
    maintainer_email='lab@todo.com',
    description='Webcam YOLO pipeline for ROS2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'webcam_node = yolo_webcam_pkg.webcam_node:main',
            'yolo_node = yolo_webcam_pkg.yolo_node:main',
            'yolo_3d_to_base_node = yolo_webcam_pkg.yolo_3d_to_base_node:main',
            'yolo_depth_to_3d_node = yolo_webcam_pkg.yolo_depth_to_3d_node:main',
            'yolo_center_to_base_node = yolo_webcam_pkg.yolo_center_to_base_node:main',
            'center_error_node = yolo_webcam_pkg.center_error_node:main',
            'pixel_servo_node = yolo_webcam_pkg.pixel_servo_node:main',
            'llm_pick_bridge_node = yolo_webcam_pkg.llm_pick_bridge_node:main',
            'llm_target_hover_node = yolo_webcam_pkg.llm_target_hover_node:main',
            'llm_anygrasp_trigger_node = yolo_webcam_pkg.llm_anygrasp_trigger_node:main',
            'llm_anygrasp_pose_filter_node = yolo_webcam_pkg.llm_anygrasp_pose_filter_node:main',
            'voice_llm_pick_bridge_node = yolo_webcam_pkg.voice_llm_pick_bridge_node:main',
            'safe_home_node = yolo_webcam_pkg.safe_home_node:main',
        ],
    },
)
