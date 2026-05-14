from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # ---------------------------------------------------------------
    # base -> fr3_link0 (identity)
    # MoveIt의 planning frame 'base'와 로봇 URDF 루트 'fr3_link0'를 연결.
    # FR3 URDF에서 base와 fr3_link0는 동일 위치이므로 identity.
    # ---------------------------------------------------------------
    tf_base_to_fr3_link0 = Node(
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

    # ---------------------------------------------------------------
    # fr3_hand_tcp -> camera_link  [UNCALIBRATED — eye-in-hand, TF tree 연결 전용]
    #
    # 카메라가 손목(fr3_hand_tcp)에 장착된 eye-in-hand 구성.
    # 저장된 handeye calibration 결과가 없어 identity 값 사용.
    # 실제 로봇 이동/grasp 실행 전에 반드시 eye-in-hand 캘리브레이션
    # (easy_handeye2)으로 정확한 값을 측정해야 한다.
    #
    # easy_handeye2 설정 참고:
    #   calibration_type:      eye_in_hand
    #   robot_base_frame:      fr3_link0
    #   robot_effector_frame:  fr3_hand_tcp
    #   tracking_base_frame:   camera_link       ← TF 충돌 방지
    #   tracking_marker_frame: checkerboard_frame
    #
    # child를 camera_link로 지정하는 이유:
    #   camera_depth_optical_frame / camera_color_optical_frame은
    #   RealSense 드라이버가 이미 부모를 갖고 있어 TF 충돌이 발생함.
    #   camera_link는 부모가 없으므로 이 transform의 child로 사용 가능.
    # ---------------------------------------------------------------
    tf_hand_tcp_to_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_fr3_hand_tcp_to_camera_link',
        output='screen',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--roll', '0.0', '--pitch', '0.0', '--yaw', '0.0',
            '--frame-id', 'fr3_hand_tcp',
            '--child-frame-id', 'camera_link',
        ],
    )

    return LaunchDescription([
        tf_base_to_fr3_link0,
        tf_hand_tcp_to_camera,
    ])
