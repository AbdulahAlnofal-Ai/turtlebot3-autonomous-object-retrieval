import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration, PythonExpression, Command
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


CELL_SIZE = 0.70


def cell_to_world_expr(value):
    return PythonExpression([
        '(',
        value,
        ' - 4) * ',
        str(CELL_SIZE)
    ])


def cell_to_world_expr_with_offset(value, offset):
    return PythonExpression([
        '(',
        value,
        ' - 4) * ',
        str(CELL_SIZE),
        ' + ',
        str(offset)
    ])


def generate_launch_description():
    pkg_project = get_package_share_directory('pkg')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    pkg_tb3_manip_desc = get_package_share_directory(
        'turtlebot3_manipulation_description'
    )

    pkg_tb3_manip_gazebo = get_package_share_directory(
        'turtlebot3_manipulation_gazebo'
    )

    pkg_tb3_moveit = get_package_share_directory(
        'turtlebot3_manipulation_moveit_config'
    )

    world_path = os.path.join(
        pkg_project,
        'worlds',
        'maze_world.world'
    )

    start_row_arg = DeclareLaunchArgument('start_row', default_value='1')
    start_col_arg = DeclareLaunchArgument('start_col', default_value='3')
    target_row_arg = DeclareLaunchArgument('target_row', default_value='1')
    target_col_arg = DeclareLaunchArgument('target_col', default_value='7')
    object_row_arg = DeclareLaunchArgument('object_row', default_value='5')
    object_col_arg = DeclareLaunchArgument('object_col', default_value='4')
    yaw_arg = DeclareLaunchArgument('yaw', default_value='1.5708')

    start_row = LaunchConfiguration('start_row')
    start_col = LaunchConfiguration('start_col')
    target_row = LaunchConfiguration('target_row')
    target_col = LaunchConfiguration('target_col')
    object_row = LaunchConfiguration('object_row')
    object_col = LaunchConfiguration('object_col')
    yaw = LaunchConfiguration('yaw')

    robot_x = cell_to_world_expr(start_col)
    robot_y = cell_to_world_expr(start_row)

    # Two objects in the same detected cell, separated slightly left/right.
    object1_x = cell_to_world_expr_with_offset(object_col, -0.05)
    object1_y = cell_to_world_expr_with_offset(object_row, 0.00)

    object2_x = cell_to_world_expr_with_offset(object_col, 0.05)
    object2_y = cell_to_world_expr_with_offset(object_row, 0.00)

    gazebo_model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=[
            os.path.expanduser('~/.gazebo/models'),
            ':',
            os.path.join(pkg_project, 'models'),
            ':',
            os.path.join(pkg_tb3_gazebo, 'models'),
            ':',
            os.path.join(pkg_tb3_manip_gazebo, 'models'),
            ':',
            os.path.dirname(pkg_tb3_manip_desc)
        ]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_gazebo_ros,
                'launch',
                'gazebo.launch.py'
            )
        ),
        launch_arguments={
            'world': world_path,
            'verbose': 'true'
        }.items()
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Use the SOURCE xacro file because this is where you added
    # the gazebo_grasp_fix / glue plugin.
    #
    # Do NOT use the installed path unless you also install the
    # modified urdf folder in turtlebot3_manipulation_gazebo.
    # ---------------------------------------------------------
    robot_xacro = os.path.expanduser(
        '~/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_manipulation_gazebo/urdf/turtlebot3_manipulation.urdf.xacro'
    )

    robot_description = ParameterValue(
        Command([
            'xacro ',
            robot_xacro,
            ' use_sim:=true'
        ]),
        value_type=str
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {
                'robot_description': robot_description,
                'use_sim_time': True
            }
        ]
    )

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'turtlebot3_manipulation',
            '-topic', 'robot_description',
            '-x', robot_x,
            '-y', robot_y,
            '-z', '0.01',
            '-Y', yaw
        ],
        output='screen'
    )

    object_model_file = os.path.join(
        pkg_project,
        'models',
        'object_block',
        'model.sdf'
    )

    spawn_object_1 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'object_1',
            '-file', object_model_file,
            '-x', object1_x,
            '-y', object1_y,
            '-z', '0.0',
            '-Y', '0.0'
        ],
        output='screen'
    )

    spawn_object_2 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'object_2',
            '-file', object_model_file,
            '-x', object2_x,
            '-y', object2_y,
            '-z', '0.0',
            '-Y', '0.0'
        ],
        output='screen'
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager',
            '/controller_manager'
        ],
        output='screen'
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'arm_controller',
            '--controller-manager',
            '/controller_manager'
        ],
        output='screen'
    )

    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'gripper_controller',
            '--controller-manager',
            '/controller_manager'
        ],
        output='screen'
    )

    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_tb3_moveit,
                'launch',
                'move_group.launch.py'
            )
        )
    )

    # Keep this only if you still need camera debug.
    # Your explorer now uses LiDAR + IK, so camera is not required for pickup.
    camera_object_detector = Node(
        package='pkg',
        executable='camera_object_detector',
        name='camera_object_detector',
        output='screen'
    )

    mission_manager = Node(
        package='pkg',
        executable='mission_manager',
        name='mission_manager',
        output='screen',
        parameters=[
            {
                'cell_size': CELL_SIZE,
                'start_row': start_row,
                'start_col': start_col,
                'target_row': target_row,
                'target_col': target_col,
            }
        ]
    )

    object_detector = Node(
        package='pkg',
        executable='object_detector',
        name='object_detector',
        output='screen'
    )

    return LaunchDescription([
        gazebo_model_path,

        start_row_arg,
        start_col_arg,
        target_row_arg,
        target_col_arg,
        object_row_arg,
        object_col_arg,
        yaw_arg,

        gazebo,

        robot_state_publisher,
        spawn_robot,

        spawn_object_1,
        spawn_object_2,

        TimerAction(
            period=4.0,
            actions=[
                joint_state_broadcaster_spawner
            ]
        ),

        TimerAction(
            period=5.0,
            actions=[
                arm_controller_spawner
            ]
        ),

        TimerAction(
            period=6.0,
            actions=[
                gripper_controller_spawner
            ]
        ),

        TimerAction(
            period=7.0,
            actions=[
                move_group
            ]
        ),

        # TimerAction(
        #     period=9.0,
        #     actions=[
        #         camera_object_detector
        #     ]
        # ),

        # TimerAction(
        #     period=10.0,
        #     actions=[
        #         mission_manager
        #     ]
        # ),

        # TimerAction(
        #     period=11.0,
        #     actions=[
        #         object_detector
        #     ]
        # ),
    ])