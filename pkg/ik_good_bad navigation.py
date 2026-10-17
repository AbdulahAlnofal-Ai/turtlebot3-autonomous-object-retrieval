import math
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand

from moveit_msgs.srv import GetPositionIK


class Explorer(Node):
    def __init__(self):
        super().__init__('explorer')

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.arm_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            '/gripper_controller/gripper_cmd'
        )

        self.ik_client = self.create_client(
            GetPositionIK,
            '/compute_ik'
        )

        self.declare_parameter('start_row', 1)
        self.declare_parameter('start_col', 3)
        self.declare_parameter('target_row', 1)
        self.declare_parameter('target_col', 7)

        self.start_cell = (
            int(self.get_parameter('start_row').value),
            int(self.get_parameter('start_col').value)
        )

        self.target_drop_cell = (
            int(self.get_parameter('target_row').value),
            int(self.get_parameter('target_col').value)
        )

        # ---------------- NAVIGATION CONSTANTS ----------------

        self.CELL_SIZE = 0.70
        self.MAZE_CELLS = 7

        self.CENTER_TOLERANCE = 0.10
        self.RETREAT_TOLERANCE = 0.11
        self.YAW_TOLERANCE = math.radians(1)

        self.FRONT_STOP = 0.35
        self.OPEN_DIST = 0.55

        self.SPEED_FWD = 0.25
        self.SPEED_BACK = -0.09
        self.SPEED_TURN = 0.25

        self.PAUSE_TIME = 0.6
        self.INIT_WAIT_TIME = 1.5

        # ---------------- PICK / DROP CONSTANTS ----------------

        # Stay slightly far so the arm does not crash from above.
        self.PICK_LIDAR_TARGET_DIST = 0.25
        self.PICK_LIDAR_TOLERANCE = 0.02
        self.PICK_APPROACH_SPEED = 0.035
        self.PICK_APPROACH_MAX_TIME = 14.0

        # Fallback poses only if IK fails.
        self.ARM_PICK_FALLBACK_1 = [0.0, 0.96, -1.70, 1.05]
        self.ARM_PICK_FALLBACK_2 = [0.0, 0.96, -1.70, 1.05]

        self.ARM_DEFAULT = [0.0, 0.25, -0.65, 0.45]
        self.ARM_DROP = [0.0, 0.65, -1.05, 0.40]

        # Not too wide. Wide opening was hitting/pushing the nearby second object.
        self.GRIPPER_OPEN = 0.012

        # Softer close. Do not close aggressively on a light/tall object.
        self.GRIPPER_CLOSE_STEP1 = 0.010
        self.GRIPPER_CLOSE_STEP2 = 0.006
        self.GRIPPER_CLOSE_STEP3 = 0.002
        self.GRIPPER_CLOSE_STEP4 = -0.004
        self.GRIPPER_EFFORT = 0.5

        # ---------------- IK CONSTANTS ----------------

        self.IK_GROUP_NAME = "arm"
        self.IK_LINK_NAME = "end_effector_link"
        self.IK_FRAME_ID = "link1"

        # From your TF test.
        self.BASE_TO_LINK1_X = 0.092

        # IK was good; this pushes object slightly deeper into the middle of gripper.
        self.IK_FORWARD_OFFSET = 0.020

        # Keep slightly high to avoid hitting the object from above.
        self.IK_PICK_Z = 0.070

        # Remove side offset. It was pushing the gripper toward the object edge.
        self.IK_OBJECT2_Y_OFFSET = 0.0

        self.last_pick_distance = 0.25
        self.last_pick_angle = 0.0
        self.ik_future = None
        self.ik_request_start_time = 0.0
        self.current_pick_pose = None

        # ---------------- ROBOT STATE ----------------

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.odom_ready = False
        self.scan_ready = False
        self.latest_scan = None

        self.current_cell = None
        self.target_cell = None
        self.target_yaw = 0.0

        self.visited = set()
        self.graph = {}
        self.backtrack_stack = []
        self.blocked_cells = set()

        self.object_cell = None
        self.object_approach_cell = None
        self.target_approach_cell = None

        self.current_path = []
        self.path_index = 0
        self.following_path = False
        self.after_path_state = None
        self.after_face_state = None

        self.mission_phase = "SEARCH_OBJECT"

        self.current_object_number = 1
        self.manip_mode = None
        self.manip_after_state = None
        self.manip_step = 0
        self.manip_step_start = time.time()
        self.manip_command_sent = False

        self.state = "INIT_WAIT"
        self.state_start = time.time()

        self.get_logger().info("DFS + BFS + MoveIt IK pickup explorer started ✓")
        self.get_logger().info(f"Start cell: {self.start_cell}")
        self.get_logger().info(f"Target cell: {self.target_drop_cell}")

    # ---------------- CALLBACKS ----------------

    def scan_callback(self, msg):
        self.latest_scan = msg
        self.scan_ready = True

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = self.get_yaw(msg)
        self.odom_ready = True

    # ---------------- BASIC HELPERS ----------------

    def get_yaw(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    def set_state(self, new_state):
        self.state = new_state
        self.state_start = time.time()

    def elapsed(self):
        return time.time() - self.state_start

    def stop_robot(self):
        self.pub.publish(Twist())

    def world_to_cell(self, x, y):
        col = round(x / self.CELL_SIZE) + 4
        row = round(y / self.CELL_SIZE) + 4

        row = max(1, min(self.MAZE_CELLS, row))
        col = max(1, min(self.MAZE_CELLS, col))

        return row, col

    def cell_to_world(self, row, col):
        x = (col - 4) * self.CELL_SIZE
        y = (row - 4) * self.CELL_SIZE
        return x, y

    def distance_to_cell_center(self, cell):
        cx, cy = self.cell_to_world(cell[0], cell[1])
        dx = self.x - cx
        dy = self.y - cy
        return math.sqrt(dx * dx + dy * dy)

    def confirmed_cell(self):
        raw_cell = self.world_to_cell(self.x, self.y)

        if self.distance_to_cell_center(raw_cell) <= self.CENTER_TOLERANCE:
            return raw_cell

        return None

    def get_zone(self, center_deg, span_deg):
        if self.latest_scan is None:
            return float('inf')

        ranges = list(self.latest_scan.ranges)
        n = len(ranges)

        def idx(deg):
            return int((deg % 360) * n / 360)

        half = span_deg // 2
        start = idx(center_deg - half)
        end = idx(center_deg + half)

        if start <= end:
            selected = ranges[start:end]
        else:
            selected = ranges[start:] + ranges[:end]

        valid = [
            r for r in selected
            if not math.isnan(r)
            and not math.isinf(r)
            and r > 0.01
        ]

        return min(valid) if valid else float('inf')

    def get_front_object_distance_angle(self):
        if self.latest_scan is None:
            return None

        ranges = list(self.latest_scan.ranges)
        n = len(ranges)

        best_dist = float('inf')
        best_deg = 0.0

        for deg in range(-35, 36):
            idx = int((deg % 360) * n / 360)
            r = ranges[idx]

            if math.isnan(r) or math.isinf(r):
                continue

            if r < 0.04 or r > 0.90:
                continue

            if r < best_dist:
                best_dist = r
                best_deg = deg

        if best_dist == float('inf'):
            return None

        return best_dist, math.radians(best_deg)

    # ---------------- IK HELPERS ----------------

    def get_fallback_pick_pose(self):
        if self.current_object_number == 1:
            return self.ARM_PICK_FALLBACK_1

        return self.ARM_PICK_FALLBACK_2

    def build_ik_request(self):
        distance = self.last_pick_distance
        angle = self.last_pick_angle

        x_scan = distance * math.cos(angle)
        y_scan = distance * math.sin(angle)

        x_link1 = x_scan + self.BASE_TO_LINK1_X + self.IK_FORWARD_OFFSET
        y_link1 = y_scan

        if self.current_object_number == 2:
            y_link1 += self.IK_OBJECT2_Y_OFFSET

        z_link1 = self.IK_PICK_Z

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.IK_GROUP_NAME
        req.ik_request.ik_link_name = self.IK_LINK_NAME
        req.ik_request.avoid_collisions = False

        req.ik_request.pose_stamped = PoseStamped()
        req.ik_request.pose_stamped.header.frame_id = self.IK_FRAME_ID
        req.ik_request.pose_stamped.pose.position.x = float(x_link1)
        req.ik_request.pose_stamped.pose.position.y = float(y_link1)
        req.ik_request.pose_stamped.pose.position.z = float(z_link1)

        req.ik_request.pose_stamped.pose.orientation.x = 0.0
        req.ik_request.pose_stamped.pose.orientation.y = 0.0
        req.ik_request.pose_stamped.pose.orientation.z = 0.0
        req.ik_request.pose_stamped.pose.orientation.w = 1.0

        req.ik_request.timeout.sec = 2
        req.ik_request.timeout.nanosec = 0

        self.get_logger().info(
            f"IK target in {self.IK_FRAME_ID}: "
            f"x={x_link1:.3f}, y={y_link1:.3f}, z={z_link1:.3f}, "
            f"distance={distance:.3f}, angle={math.degrees(angle):.1f}"
        )

        return req

    def extract_arm_joints_from_ik(self, response):
        if response is None:
            return None

        if response.error_code.val != 1:
            self.get_logger().warn(
                f"MoveIt IK failed with error_code={response.error_code.val}"
            )
            return None

        names = list(response.solution.joint_state.name)
        positions = list(response.solution.joint_state.position)

        joint_map = {}

        for name, pos in zip(names, positions):
            joint_map[name] = pos

        required = ['joint1', 'joint2', 'joint3', 'joint4']

        for joint in required:
            if joint not in joint_map:
                self.get_logger().warn(
                    f"IK result missing {joint}. Available joints: {names}"
                )
                return None

        pose = [
            joint_map['joint1'],
            joint_map['joint2'],
            joint_map['joint3'],
            joint_map['joint4']
        ]

        self.get_logger().info(
            f"IK solution: {[round(v, 4) for v in pose]}"
        )

        return pose

    def request_ik_async(self):
        if not self.ik_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warn("/compute_ik service not available, using fallback pose")
            self.current_pick_pose = self.get_fallback_pick_pose()
            return False

        req = self.build_ik_request()
        self.ik_future = self.ik_client.call_async(req)
        self.ik_request_start_time = time.time()
        return True

    # ---------------- GRID HELPERS ----------------

    def yaw_to_heading(self):
        yaw = self.normalize_angle(self.yaw)

        headings = {
            "E": 0.0,
            "N": math.pi / 2,
            "W": math.pi,
            "S": -math.pi / 2
        }

        best_heading = "E"
        best_error = 999.0

        for heading, angle in headings.items():
            error = abs(self.normalize_angle(yaw - angle))

            if error < best_error:
                best_error = error
                best_heading = heading

        return best_heading

    def left_of(self, heading):
        return {
            "N": "W",
            "W": "S",
            "S": "E",
            "E": "N"
        }[heading]

    def right_of(self, heading):
        return {
            "N": "E",
            "E": "S",
            "S": "W",
            "W": "N"
        }[heading]

    def back_of(self, heading):
        return {
            "N": "S",
            "S": "N",
            "E": "W",
            "W": "E"
        }[heading]

    def neighbor_cell(self, cell, direction):
        row, col = cell

        if direction == "N":
            return row + 1, col
        if direction == "S":
            return row - 1, col
        if direction == "E":
            return row, col + 1
        if direction == "W":
            return row, col - 1

        return cell

    def inside_maze(self, cell):
        row, col = cell
        return 1 <= row <= 7 and 1 <= col <= 7

    def direction_to_yaw(self, direction):
        if direction == "E":
            return 0.0
        if direction == "N":
            return math.pi / 2
        if direction == "W":
            return math.pi
        if direction == "S":
            return -math.pi / 2

        return 0.0

    def direction_between_cells(self, start, goal):
        sr, sc = start
        gr, gc = goal

        if gr > sr:
            return "N"
        if gr < sr:
            return "S"
        if gc > sc:
            return "E"
        if gc < sc:
            return "W"

        return None

    # ---------------- GRAPH / SEARCH ----------------

    def add_edge(self, a, b):
        if a in self.blocked_cells or b in self.blocked_cells:
            return

        if a not in self.graph:
            self.graph[a] = set()

        if b not in self.graph:
            self.graph[b] = set()

        self.graph[a].add(b)
        self.graph[b].add(a)

    def remove_edge(self, a, b):
        if a in self.graph and b in self.graph[a]:
            self.graph[a].remove(b)

        if b in self.graph and a in self.graph[b]:
            self.graph[b].remove(a)

    def detect_open_neighbors(self, cell):
        heading = self.yaw_to_heading()

        front_dist = self.get_zone(0, 40)
        left_dist = self.get_zone(90, 40)
        right_dist = self.get_zone(270, 40)
        back_dist = self.get_zone(180, 40)

        checks = [
            (heading, front_dist),
            (self.left_of(heading), left_dist),
            (self.right_of(heading), right_dist),
            (self.back_of(heading), back_dist)
        ]

        neighbors = []

        for direction, dist in checks:
            if dist > self.OPEN_DIST:
                neighbor = self.neighbor_cell(cell, direction)

                if self.inside_maze(neighbor) and neighbor not in self.blocked_cells:
                    neighbors.append(neighbor)
                    self.add_edge(cell, neighbor)

        return neighbors

    def choose_next_cell(self):
        cell = self.current_cell
        self.visited.add(cell)

        open_neighbors = self.detect_open_neighbors(cell)

        for neighbor in open_neighbors:
            if neighbor not in self.visited and neighbor not in self.blocked_cells:
                self.backtrack_stack.append(cell)

                self.get_logger().info(
                    f"DFS: {cell} → {neighbor} | visited={len(self.visited)}"
                )

                return neighbor

        while self.backtrack_stack:
            back_cell = self.backtrack_stack.pop()

            if back_cell == self.current_cell:
                continue

            if back_cell not in self.blocked_cells:
                self.get_logger().info(
                    f"Backtrack: {cell} → {back_cell}"
                )

                return back_cell

        self.get_logger().warn("DFS complete: no more unvisited cells")
        return None

    def bfs(self, start, goal, extra_blocked=None):
        if extra_blocked is None:
            extra_blocked = set()

        if start not in self.graph or goal not in self.graph:
            return None

        queue = deque([start])
        came_from = {start: None}

        while queue:
            current = queue.popleft()

            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for neighbor in self.graph.get(current, []):
                if neighbor in came_from:
                    continue

                if neighbor in self.blocked_cells:
                    continue

                if neighbor in extra_blocked and neighbor != goal:
                    continue

                came_from[neighbor] = current
                queue.append(neighbor)

        return None

    def astar(self, start, goal, extra_blocked=None):
        return self.bfs(start, goal, extra_blocked)

    # ---------------- NAVIGATION MOTION ----------------

    def start_move_to_cell(self, target):
        if target == self.target_drop_cell and self.mission_phase in ["GO_TARGET_1", "GO_TARGET_2"]:
            self.get_logger().warn(
                f"Refusing to drive into target cell {self.target_drop_cell}; stopping at approach cell."
            )

            if self.mission_phase == "GO_TARGET_1":
                self.set_state("FACE_TARGET_1")
            else:
                self.set_state("FACE_TARGET_2")

            return

        direction = self.direction_between_cells(self.current_cell, target)

        if direction is None:
            self.set_state("PAUSE_AT_CELL")
            return

        self.target_cell = target
        self.target_yaw = self.direction_to_yaw(direction)

        self.set_state("ROTATE_TO_TARGET")

    def rotate_to_target(self, cmd):
        error = self.normalize_angle(self.target_yaw - self.yaw)

        if abs(error) <= self.YAW_TOLERANCE:
            cmd.angular.z = 0.0
            self.stop_robot()
            self.set_state("MOVE_TO_TARGET")
            return

        if error > 0:
            cmd.angular.z = self.SPEED_TURN
        else:
            cmd.angular.z = -self.SPEED_TURN

    def rotate_in_place_to_target(self, cmd):
        error = self.normalize_angle(self.target_yaw - self.yaw)

        if abs(error) <= self.YAW_TOLERANCE:
            cmd.angular.z = 0.0
            self.stop_robot()
            self.set_state(self.after_face_state)
            return

        if error > 0:
            cmd.angular.z = self.SPEED_TURN
        else:
            cmd.angular.z = -self.SPEED_TURN

    def start_face_cell(self, cell_to_face, after_state):
        direction = self.direction_between_cells(self.current_cell, cell_to_face)

        if direction is None:
            self.set_state(after_state)
            return

        self.target_yaw = self.direction_to_yaw(direction)
        self.after_face_state = after_state
        self.set_state("ROTATE_IN_PLACE_TO_FACE")

    def handle_blocked_path(self):
        self.get_logger().warn(
            f"Blocked edge: {self.current_cell} → {self.target_cell}. Removing edge."
        )

        self.remove_edge(self.current_cell, self.target_cell)

        if self.following_path:
            self.following_path = False
            self.current_path = []
            self.path_index = 0

            if self.mission_phase == "GO_TARGET_1":
                self.set_state("PLAN_TO_TARGET_1")
            elif self.mission_phase == "BACK_TO_OBJECT":
                self.set_state("PLAN_BACK_TO_OBJECT")
            elif self.mission_phase == "GO_TARGET_2":
                self.set_state("PLAN_TO_TARGET_2")
            elif self.mission_phase == "RETURN_START":
                self.set_state("PLAN_BACK_TO_START")
            else:
                self.set_state("PAUSE_AT_CELL")
        else:
            self.set_state("PAUSE_AT_CELL")

    def move_to_target(self, cmd):
        if self.target_cell is None:
            self.set_state("PAUSE_AT_CELL")
            return

        front = self.get_zone(0, 40)

        if front < self.FRONT_STOP:
            if self.mission_phase == "SEARCH_OBJECT":
                self.object_cell = self.target_cell
                self.object_approach_cell = self.current_cell

                self.blocked_cells.add(self.object_cell)
                self.remove_edge(self.object_approach_cell, self.object_cell)

                self.get_logger().warn(
                    f"Object found at {self.object_cell}"
                )

                self.get_logger().info(
                    f"Pickup approach cell: {self.object_approach_cell}"
                )

                self.get_logger().info(
                    "Retreating to approach cell center before continuing"
                )

                self.stop_robot()
                self.set_state("RETREAT_TO_APPROACH_CENTER")
                return

            self.stop_robot()
            self.handle_blocked_path()
            return

        tx, ty = self.cell_to_world(self.target_cell[0], self.target_cell[1])

        dx = tx - self.x
        dy = ty - self.y

        dist = math.sqrt(dx * dx + dy * dy)

        if dist <= self.CENTER_TOLERANCE:
            self.current_cell = self.target_cell
            self.visited.add(self.current_cell)

            if self.current_cell not in self.graph:
                self.graph[self.current_cell] = set()

            self.stop_robot()

            self.get_logger().info(
                f"Reached {self.current_cell}"
            )

            if self.following_path:
                self.advance_path()
            else:
                self.set_state("PAUSE_AT_CELL")

            return

        cmd.linear.x = self.SPEED_FWD
        cmd.angular.z = 0.0

    def start_follow_path(self, path, after_state):
        self.current_path = path
        self.path_index = 1
        self.after_path_state = after_state
        self.following_path = True

        self.get_logger().info(f"BFS path: {path}")

        if len(path) <= 1:
            self.following_path = False
            self.set_state(after_state)
            return

        self.start_move_to_cell(path[self.path_index])

    def advance_path(self):
        self.path_index += 1

        if self.path_index >= len(self.current_path):
            self.following_path = False
            self.set_state(self.after_path_state)
            return

        self.start_move_to_cell(self.current_path[self.path_index])

    def plan_to_goal(self, goal, after_state):
        extra_blocked = set()

        if goal != self.target_drop_cell:
            extra_blocked.add(self.target_drop_cell)

        path = self.bfs(self.current_cell, goal, extra_blocked)

        if path is None:
            return False

        self.start_follow_path(path, after_state)
        return True

    def plan_to_target_approach(self, face_state):
        path_to_target = self.bfs(self.current_cell, self.target_drop_cell)

        if path_to_target is None:
            return False

        if len(path_to_target) < 2:
            self.get_logger().warn(
                "Robot is already at target cell; cannot choose approach cell"
            )
            return False

        self.target_approach_cell = path_to_target[-2]
        approach_path = path_to_target[:-1]

        self.get_logger().info(
            f"Target approach cell: {self.target_approach_cell}, "
            f"target cell: {self.target_drop_cell}"
        )

        self.start_follow_path(approach_path, face_state)
        return True

    def continue_dfs_until_target_known(self):
        self.visited.add(self.current_cell)
        self.detect_open_neighbors(self.current_cell)

        if self.bfs(self.current_cell, self.target_drop_cell) is not None:
            self.get_logger().warn(
                "Target discovered in graph → plan to target approach"
            )
            self.set_state("PLAN_TO_TARGET_1")
            return

        next_cell = self.choose_next_cell()

        if next_cell is None:
            self.get_logger().warn(
                "Could not discover target with current graph"
            )
            self.set_state("DONE")
        else:
            self.start_move_to_cell(next_cell)

    # ---------------- ARM HELPERS ----------------

    def send_arm_pose(self, positions, seconds=3):
        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = seconds
        point.time_from_start.nanosec = 0

        msg.points.append(point)
        self.arm_pub.publish(msg)

    def send_gripper_goal(self, position):
        if not self.gripper_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warn("Gripper action server not available")
            return

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = self.GRIPPER_EFFORT

        self.gripper_client.send_goal_async(goal)

    def start_manipulation(self, mode, after_state):
        self.manip_mode = mode
        self.manip_after_state = after_state
        self.manip_step = 0
        self.manip_step_start = time.time()
        self.manip_command_sent = False
        self.ik_future = None
        self.current_pick_pose = None

        self.stop_robot()

        self.get_logger().warn(
            f"Starting {mode.upper()} sequence for object {self.current_object_number}"
        )

        self.set_state("MANIPULATE")

    def next_manip_step(self):
        self.manip_step += 1
        self.manip_step_start = time.time()
        self.manip_command_sent = False

    def manip_elapsed(self):
        return time.time() - self.manip_step_start

    def run_pick_sequence(self, cmd):
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        if self.manip_step == 0:
            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_OPEN)
                self.get_logger().info("Pick: gripper open")
                self.manip_command_sent = True

            if self.manip_elapsed() >= 1.0:
                self.next_manip_step()

        elif self.manip_step == 1:
            result = self.get_front_object_distance_angle()

            if result is None:
                self.get_logger().warn("Pick: waiting for LiDAR object point")
                return

            distance, angle = result
            angle_deg = math.degrees(angle)

            self.last_pick_distance = distance
            self.last_pick_angle = angle

            if distance <= self.PICK_LIDAR_TARGET_DIST + self.PICK_LIDAR_TOLERANCE:
                self.stop_robot()
                self.get_logger().info(
                    f"Pick: distance ready d={distance:.3f}, angle={angle_deg:.1f}"
                )
                self.next_manip_step()

            elif distance > self.PICK_LIDAR_TARGET_DIST:
                cmd.linear.x = self.PICK_APPROACH_SPEED
                cmd.angular.z = 0.0

                if self.manip_elapsed() >= self.PICK_APPROACH_MAX_TIME:
                    self.stop_robot()
                    self.get_logger().warn(
                        f"Pick: max approach time reached d={distance:.3f}, angle={angle_deg:.1f}"
                    )
                    self.next_manip_step()

        elif self.manip_step == 2:
            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_OPEN)
                self.get_logger().info("Pick: gripper open before IK move")

                ik_started = self.request_ik_async()
                self.manip_command_sent = True

                if not ik_started:
                    self.send_arm_pose(self.current_pick_pose, 3)
                    self.get_logger().warn(
                        f"Pick: using fallback pose {self.current_pick_pose}"
                    )
                    self.next_manip_step()

                return

            if self.ik_future is not None and self.ik_future.done():
                try:
                    response = self.ik_future.result()
                    ik_pose = self.extract_arm_joints_from_ik(response)
                except Exception as e:
                    self.get_logger().warn(f"MoveIt IK exception: {e}")
                    ik_pose = None

                if ik_pose is None:
                    self.current_pick_pose = self.get_fallback_pick_pose()
                    self.get_logger().warn(
                        f"Pick: IK failed, using fallback pose {self.current_pick_pose}"
                    )
                else:
                    self.current_pick_pose = ik_pose

                self.send_arm_pose(self.current_pick_pose, 3)
                self.get_logger().info(
                    f"Pick: arm to IK/fallback pickup pose {self.current_pick_pose}"
                )
                self.next_manip_step()
                return

            if time.time() - self.ik_request_start_time > 3.0:
                self.current_pick_pose = self.get_fallback_pick_pose()
                self.get_logger().warn(
                    f"Pick: IK timeout, using fallback pose {self.current_pick_pose}"
                )
                self.send_arm_pose(self.current_pick_pose, 3)
                self.next_manip_step()

        elif self.manip_step == 3:
            if self.manip_elapsed() >= 3.2:
                self.next_manip_step()

        elif self.manip_step == 4:
            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_CLOSE_STEP1)
                self.get_logger().info("Pick: gripper slow close step 1")
                self.manip_command_sent = True

            if self.manip_elapsed() >= 1.2:
                self.next_manip_step()

        elif self.manip_step == 5:
            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_CLOSE_STEP2)
                self.get_logger().info("Pick: gripper slow close step 2")
                self.manip_command_sent = True

            if self.manip_elapsed() >= 1.2:
                self.next_manip_step()

        elif self.manip_step == 6:
            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_CLOSE_STEP3)
                self.get_logger().info("Pick: gripper slow close step 3")
                self.manip_command_sent = True

            if self.manip_elapsed() >= 1.2:
                self.next_manip_step()

        elif self.manip_step == 7:
            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_CLOSE_STEP4)
                self.get_logger().info("Pick: gripper slow close final step")
                self.manip_command_sent = True

            if self.manip_elapsed() >= 1.8:
                self.next_manip_step()

        elif self.manip_step == 8:
            if not self.manip_command_sent:
                self.send_arm_pose(self.ARM_DEFAULT, 3)
                self.get_logger().info("Pick: arm to gentle carry pose")
                self.manip_command_sent = True

            if self.manip_elapsed() >= 3.2:
                self.next_manip_step()

        elif self.manip_step == 9:
            dist = self.distance_to_cell_center(self.object_approach_cell)

            if dist <= self.RETREAT_TOLERANCE:
                self.current_cell = self.object_approach_cell
                self.stop_robot()
                self.get_logger().info("Pick: returned to object approach center")
                self.next_manip_step()
            else:
                cmd.linear.x = self.SPEED_BACK
                cmd.angular.z = 0.0

        elif self.manip_step == 10:
            self.get_logger().warn(
                f"Pick sequence complete for object {self.current_object_number}"
            )
            self.set_state(self.manip_after_state)

    def run_drop_sequence(self, cmd):
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        if self.manip_step == 0:
            if not self.manip_command_sent:
                self.send_arm_pose(self.ARM_DROP, 3)
                self.get_logger().info("Drop: arm to drop pose")
                self.manip_command_sent = True

            if self.manip_elapsed() >= 3.2:
                self.next_manip_step()

        elif self.manip_step == 1:
            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_OPEN)
                self.get_logger().info("Drop: gripper open")
                self.manip_command_sent = True

            if self.manip_elapsed() >= 1.5:
                self.next_manip_step()

        elif self.manip_step == 2:
            if not self.manip_command_sent:
                self.send_arm_pose(self.ARM_DEFAULT, 3)
                self.get_logger().info("Drop: arm to carry/default")
                self.manip_command_sent = True

            if self.manip_elapsed() >= 3.2:
                self.next_manip_step()

        elif self.manip_step == 3:
            self.get_logger().warn(
                f"Drop sequence complete for object {self.current_object_number}"
            )
            self.set_state(self.manip_after_state)

    def run_manipulation(self, cmd):
        if self.manip_mode == "pick":
            self.run_pick_sequence(cmd)
        elif self.manip_mode == "drop":
            self.run_drop_sequence(cmd)
        else:
            self.set_state(self.manip_after_state)

    # ---------------- MISSION ----------------

    def after_object_retreat(self):
        self.start_face_cell(self.object_cell, "PICK_OBJECT_1")

    def control_loop(self):
        cmd = Twist()

        if self.state == "INIT_WAIT":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if self.odom_ready and self.scan_ready and self.elapsed() >= self.INIT_WAIT_TIME:
                cell = self.confirmed_cell()

                if cell is not None:
                    self.current_cell = cell
                    self.visited.add(cell)

                    if cell not in self.graph:
                        self.graph[cell] = set()

                    self.get_logger().info(f"Start DFS at {cell}")
                    self.set_state("PAUSE_AT_CELL")

        elif self.state == "PAUSE_AT_CELL":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if self.elapsed() >= self.PAUSE_TIME:
                if self.mission_phase == "SEARCH_TARGET":
                    self.set_state("CONTINUE_SEARCH_TARGET")
                else:
                    self.set_state("CHOOSE_NEXT")

        elif self.state == "CHOOSE_NEXT":
            next_cell = self.choose_next_cell()

            if next_cell is None:
                self.stop_robot()
                self.set_state("DONE")
            else:
                self.start_move_to_cell(next_cell)

        elif self.state == "CONTINUE_SEARCH_TARGET":
            self.continue_dfs_until_target_known()

        elif self.state == "RETREAT_TO_APPROACH_CENTER":
            cmd.angular.z = 0.0

            dist = self.distance_to_cell_center(self.object_approach_cell)

            if dist <= self.RETREAT_TOLERANCE:
                self.current_cell = self.object_approach_cell
                self.stop_robot()

                self.get_logger().info(
                    f"Back at safe approach center {self.object_approach_cell}"
                )

                self.after_object_retreat()
                return

            cmd.linear.x = self.SPEED_BACK

        elif self.state == "PICK_OBJECT_1":
            self.current_object_number = 1
            self.start_manipulation("pick", "PLAN_TO_TARGET_1")

        elif self.state == "PLAN_TO_TARGET_1":
            self.mission_phase = "GO_TARGET_1"

            if self.plan_to_target_approach("FACE_TARGET_1"):
                self.get_logger().info("BFS: going to target approach first time")
            else:
                self.get_logger().warn(
                    "No known path to target yet → continue DFS"
                )
                self.mission_phase = "SEARCH_TARGET"
                self.set_state("PAUSE_AT_CELL")

        elif self.state == "FACE_TARGET_1":
            self.start_face_cell(self.target_drop_cell, "DROP_OBJECT_1")

        elif self.state == "DROP_OBJECT_1":
            self.current_object_number = 1
            self.start_manipulation("drop", "PLAN_BACK_TO_OBJECT")

        elif self.state == "PLAN_BACK_TO_OBJECT":
            self.mission_phase = "BACK_TO_OBJECT"

            if self.plan_to_goal(self.object_approach_cell, "AT_OBJECT_AGAIN"):
                self.get_logger().info("BFS: returning to object approach cell")
            else:
                self.get_logger().warn("No path back to object approach cell")
                self.set_state("DONE")

        elif self.state == "AT_OBJECT_AGAIN":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if self.elapsed() >= self.PAUSE_TIME:
                self.get_logger().warn("Returned to object approach cell")
                self.start_face_cell(self.object_cell, "PICK_OBJECT_2")

        elif self.state == "PICK_OBJECT_2":
            self.current_object_number = 2
            self.start_manipulation("pick", "PLAN_TO_TARGET_2")

        elif self.state == "PLAN_TO_TARGET_2":
            self.mission_phase = "GO_TARGET_2"

            if self.plan_to_target_approach("FACE_TARGET_2"):
                self.get_logger().info("BFS: going to target approach second time")
            else:
                self.get_logger().warn("No path to target second time")
                self.set_state("DONE")

        elif self.state == "FACE_TARGET_2":
            self.start_face_cell(self.target_drop_cell, "DROP_OBJECT_2")

        elif self.state == "DROP_OBJECT_2":
            self.current_object_number = 2
            self.start_manipulation("drop", "PLAN_BACK_TO_START")

        elif self.state == "PLAN_BACK_TO_START":
            self.mission_phase = "RETURN_START"

            if self.plan_to_goal(self.start_cell, "RETURNED_START"):
                self.get_logger().info("BFS: returning to start cell")
            else:
                self.get_logger().warn("No path back to start")
                self.set_state("DONE")

        elif self.state == "RETURNED_START":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if self.elapsed() >= self.PAUSE_TIME:
                self.get_logger().warn(
                    f"Mission complete. Returned to {self.start_cell}"
                )
                self.set_state("DONE")

        elif self.state == "ROTATE_TO_TARGET":
            self.rotate_to_target(cmd)

        elif self.state == "ROTATE_IN_PLACE_TO_FACE":
            self.rotate_in_place_to_target(cmd)

        elif self.state == "MOVE_TO_TARGET":
            self.move_to_target(cmd)

        elif self.state == "MANIPULATE":
            self.run_manipulation(cmd)

        elif self.state == "DONE":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.pub.publish(cmd)

    def run_timer(self):
        self.control_loop()


def main(args=None):
    rclpy.init(args=args)

    node = Explorer()
    node.create_timer(0.1, node.run_timer)

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        try:
            node.pub.publish(Twist())
        except Exception:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()