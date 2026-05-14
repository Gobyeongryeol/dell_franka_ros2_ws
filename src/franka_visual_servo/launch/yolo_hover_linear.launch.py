from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = PathJoinSubstitution([
        FindPackageShare('franka_visual_servo'),
        'config',
        'yolo_hover_linear_params.yaml',
    ])

    dry_run_arg = DeclareLaunchArgument(
        'dry_run', default_value='false',
        description='If true, plan but do not execute any motion')

    once_arg = DeclareLaunchArgument(
        'once', default_value='true',
        description='If true, process only the first YOLO target then stop')

    node = Node(
        package='franka_visual_servo',
        executable='yolo_hover_linear_node',
        name='yolo_hover_linear_node',
        output='screen',
        parameters=[
            params_file,
            {
                'dry_run': LaunchConfiguration('dry_run'),
                'once':    LaunchConfiguration('once'),
            },
        ],
    )

    return LaunchDescription([
        dry_run_arg,
        once_arg,
        node,
    ])
