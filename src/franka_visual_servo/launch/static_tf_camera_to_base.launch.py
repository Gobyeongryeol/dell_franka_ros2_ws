from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # base -> fr3_link0: MoveIt planning frame 'base'와 로봇 URDF root 'fr3_link0'를
    # 연결해 TF tree를 하나로 합친다. 이 transform이 없으면 camera frame과 base frame이
    # 별개의 tree로 분리되어 MoveIt planning_scene_monitor가 camera 객체를 base로
    # 변환하지 못한다.
    tf_base_to_link0 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_base_to_fr3_link0',
        output='screen',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--roll', '0.0', '--pitch', '0.0', '--yaw', '0.0',
            '--frame-id', 'base',
            '--child-frame-id', 'fr3_link0',
        ],
    )

    # fr3_link8 -> camera_link: RealSense TF tree의 루트(camera_link)를 child로 지정.
    # camera_color_optical_frame은 RealSense 드라이버가 이미 parent를 갖고 있으므로
    # child로 사용하면 TF 충돌이 발생한다. camera_link는 parent가 없으므로 안전하다.
    # 현재 identity transform (handeye calibration 미적용 상태 — TF tree 연결 전용).
    tf_link8_to_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_fr3_link8_to_camera_link',
        output='screen',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--roll', '0.0', '--pitch', '0.0', '--yaw', '0.0',
            '--frame-id', 'fr3_link8',
            '--child-frame-id', 'camera_link',
        ],
    )

    return LaunchDescription([tf_base_to_link0, tf_link8_to_camera])
