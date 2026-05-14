from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = PathJoinSubstitution([
        FindPackageShare('franka_visual_servo'),
        'config',
        'yolo_hover_linear_final.yaml',
    ])

    node = Node(
        package='franka_visual_servo',
        executable='yolo_hover_linear_node',
        name='yolo_hover_linear_node',
        output='screen',
        parameters=[params_file],
    )

    return LaunchDescription([node])
