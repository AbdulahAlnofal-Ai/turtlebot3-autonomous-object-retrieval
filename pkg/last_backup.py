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

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

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

        # ============================================================
        # NAVIGATION CONSTANTS
        # ============================================================

        self.CELL_SIZE = 0.70
        self.MAZE_CELLS = 7

        self.CENTER_TOLERANCE = 0.10
        self.RETREAT_TOLERANCE = 0.11
        self.YAW_TOLERANCE = math.radians(1.0)

        self.FRONT_STOP = 0.35
        self.OPEN_DIST = 0.55

        self.SPEED_FWD = 0.25
        self.SPEED_BACK = -0.09
        self.SPEED_TURN = 0.25

        self.PAUSE_TIME = 0.6
        self.INIT_WAIT_TIME = 1.5

        # ============================================================
        # PICKUP CONSTANTS
        # ============================================================

        # Slightly closer than before so the object enters deeper into the gripper.
        self.PICK_LIDAR_TARGET_DIST = 0.225
        self.PICK_LIDAR_TOLERANCE = 0.015
        self.PICK_TOO_CLOSE_DIST = 0.145

        self.PICK_APPROACH_SPEED = 0.035
        self.PICK_APPROACH_MAX_TIME = 25.0

        # Gentle alignment only; keeps navigation stable.
        self.PICK_ALIGN_KP = 0.12
        self.PICK_ALIGN_MAX_W = 0.018
        self.PICK_ALIGN_DEADBAND = math.radians(4.0)

        # Wider LiDAR cone for second pickup.
        self.LIDAR_FRONT_MIN_DEG = -45
        self.LIDAR_FRONT_MAX_DEG = 45
        self.LIDAR_MAX_OBJECT_DIST = 0.85
        self.LIDAR_CLOSE_OBJECT_DIST = 0.45
        self.last_lidar_log_time = 0.0

        # ============================================================
        # ARM / GRIPPER CONSTANTS
        # ============================================================

        self.ARM_DEFAULT = [0.0, 0.15, -0.45, 0.65]
        self.ARM_DROP = [0.0, 0.65, -1.05, 0.40]

                # Small base nudge during drop to place object deeper into target cell
        self.DROP_FORWARD_SPEED = 0.055
        self.DROP_BACK_SPEED = -0.055
        self.DROP_NUDGE_TIME = 0.9

        # Open enough to release glue.
        self.GRIPPER_OPEN = 0.016

        # Glue plugin needs real contact. Final close is slightly negative.
        self.GRIPPER_CLOSE_STEP1 = 0.008
        self.GRIPPER_CLOSE_STEP2 = 0.004
        self.GRIPPER_CLOSE_STEP3 = 0.000
        self.GRIPPER_CLOSE_STEP4 = -0.008
        self.GRIPPER_EFFORT = 0.55

        # ============================================================
        # MOVEIT IK CONSTANTS
        # ============================================================

        self.IK_GROUP_NAME = "arm"
        self.IK_LINK_NAME = "end_effector_link"
        self.IK_FRAME_ID = "link1"

        self.BASE_TO_LINK1_X = 0.092

        # Slightly more forward than 0.010 so the object is deeper in gripper.
        # With d=0.235: x ≈ 0.235 + 0.092 + 0.018 = 0.345, still reachable.
        self.IK_FORWARD_OFFSET = 0.024
        self.IK_PICK_Z = 0.055
        self.IK_MAX_LATERAL_ANGLE = math.radians(12.0)

        self.last_pick_distance = self.PICK_LIDAR_TARGET_DIST
        self.last_pick_angle = 0.0

        self.ik_future = None
        self.ik_request_start_time = 0.0
        self.current_pick_pose = None

        # ============================================================
        # ROBOT STATE
        # ============================================================

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

        self.get_logger().info("Explorer started: DFS/BFS + MoveIt IK + glue grasp (LAST)")
        self.get_logger().info(f"Start cell: {self.start_cell}")
        self.get_logger().info(f"Target cell: {self.target_drop_cell}")

    # ============================================================
    # CALLBACKS
    # ============================================================

    def scan_callback(self, msg):
        self.latest_scan = msg
        self.scan_ready = True

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = self.get_yaw_from_odom(msg)
        self.odom_ready = True

    # ============================================================
    # BASIC HELPERS
    # ============================================================

    def get_yaw_from_odom(self, msg):
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

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def set_state(self, state):
        self.state = state
        self.state_start = time.time()

    def elapsed(self):
        return time.time() - self.state_start

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

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
        return math.hypot(self.x - cx, self.y - cy)

    def confirmed_cell(self):
        cell = self.world_to_cell(self.x, self.y)

        if self.distance_to_cell_center(cell) <= self.CENTER_TOLERANCE:
            return cell

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

    # ============================================================
    # OBJECT DETECTION FOR PICKUP
    # ============================================================

    def get_front_object_distance_angle(self):
        if self.latest_scan is None:
            return None

        ranges = list(self.latest_scan.ranges)
        n = len(ranges)

        points = []

        for deg in range(self.LIDAR_FRONT_MIN_DEG, self.LIDAR_FRONT_MAX_DEG + 1):
            idx = int((deg % 360) * n / 360)
            r = ranges[idx]

            if math.isnan(r) or math.isinf(r):
                continue

            if r < 0.04 or r > self.LIDAR_MAX_OBJECT_DIST:
                continue

            points.append((deg, r))

        if not points:
            return None

        clusters = []
        current = [points[0]]

        for p in points[1:]:
            prev_deg, prev_r = current[-1]
            deg, r = p

            same_cluster = (
                abs(deg - prev_deg) <= 2
                and abs(r - prev_r) <= 0.08
            )

            if same_cluster:
                current.append(p)
            else:
                if len(current) >= 2:
                    clusters.append(current)
                current = [p]

        if len(current) >= 2:
            clusters.append(current)

        if not clusters:
            best_deg, best_dist = min(points, key=lambda p: p[1])
            return best_dist, math.radians(best_deg)

        candidates = []

        for cluster in clusters:
            degs = [p[0] for p in cluster]
            dists = [p[1] for p in cluster]

            candidates.append({
                "angle_deg": sum(degs) / len(degs),
                "distance": min(dists),
                "size": len(cluster)
            })

        filtered = [c for c in candidates if c["size"] >= 3]
        if filtered:
            candidates = filtered

        close_candidates = [
            c for c in candidates
            if c["distance"] <= self.LIDAR_CLOSE_OBJECT_DIST
        ]

        if close_candidates:
            candidates = close_candidates

        chosen = min(candidates, key=lambda c: c["distance"])

        now = time.time()
        if now - self.last_lidar_log_time >= 0.7:
            self.last_lidar_log_time = now
            debug_list = [
                (round(c["angle_deg"], 1), round(c["distance"], 3), c["size"])
                for c in candidates
            ]

            self.get_logger().info(
                f"LiDAR objects: {debug_list} | "
                f"selected nearest: angle={chosen['angle_deg']:.1f}, "
                f"dist={chosen['distance']:.3f}"
            )

        return chosen["distance"], math.radians(chosen["angle_deg"])

    # ============================================================
    # MOVEIT IK
    # ============================================================

    def build_ik_request(self):
        distance = self.last_pick_distance
        raw_angle = self.last_pick_angle

        angle = self.clamp(
            raw_angle,
            -self.IK_MAX_LATERAL_ANGLE,
            self.IK_MAX_LATERAL_ANGLE
        )

        x_scan = distance * math.cos(angle)
        y_scan = distance * math.sin(angle)

        x_link1 = x_scan + self.BASE_TO_LINK1_X + self.IK_FORWARD_OFFSET
        y_link1 = y_scan
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
            f"distance={distance:.3f}, "
            f"raw_angle={math.degrees(raw_angle):.1f}, "
            f"used_angle={math.degrees(angle):.1f}"
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

        joint_map = dict(zip(names, positions))
        required = ['joint1', 'joint2', 'joint3', 'joint4']

        for joint in required:
            if joint not in joint_map:
                self.get_logger().warn(
                    f"IK result missing {joint}. Available joints: {names}"
                )
                return None

        pose = [joint_map[j] for j in required]

        self.get_logger().info(
            f"IK solution: {[round(v, 4) for v in pose]}"
        )

        return pose

    def request_ik_async(self):
        if not self.ik_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().error("/compute_ik service not available")
            return False

        req = self.build_ik_request()
        self.ik_future = self.ik_client.call_async(req)
        self.ik_request_start_time = time.time()

        return True

    # ============================================================
    # GRID / GRAPH
    # ============================================================

    def yaw_to_heading(self):
        yaw = self.normalize_angle(self.yaw)

        options = {
            "E": 0.0,
            "N": math.pi / 2,
            "W": math.pi,
            "S": -math.pi / 2
        }

        return min(
            options,
            key=lambda h: abs(self.normalize_angle(yaw - options[h]))
        )

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

        return (
            1 <= row <= self.MAZE_CELLS
            and 1 <= col <= self.MAZE_CELLS
        )

    def direction_to_yaw(self, direction):
        return {
            "E": 0.0,
            "N": math.pi / 2,
            "W": math.pi,
            "S": -math.pi / 2
        }.get(direction, 0.0)

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

    def add_edge(self, a, b):
        if a in self.blocked_cells or b in self.blocked_cells:
            return

        self.graph.setdefault(a, set())
        self.graph.setdefault(b, set())

        self.graph[a].add(b)
        self.graph[b].add(a)

    def remove_edge(self, a, b):
        if a in self.graph:
            self.graph[a].discard(b)

        if b in self.graph:
            self.graph[b].discard(a)

    def detect_open_neighbors(self, cell):
        heading = self.yaw_to_heading()

        checks = [
            (heading, self.get_zone(0, 40)),
            (self.left_of(heading), self.get_zone(90, 40)),
            (self.right_of(heading), self.get_zone(270, 40)),
            (self.back_of(heading), self.get_zone(180, 40)),
        ]

        neighbors = []

        for direction, dist in checks:
            if dist > self.OPEN_DIST:
                neighbor = self.neighbor_cell(cell, direction)

                if (
                    self.inside_maze(neighbor)
                    and neighbor not in self.blocked_cells
                ):
                    neighbors.append(neighbor)
                    self.add_edge(cell, neighbor)

        return neighbors

    def choose_next_cell(self):
        cell = self.current_cell
        self.visited.add(cell)

        for neighbor in self.detect_open_neighbors(cell):
            if (
                neighbor not in self.visited
                and neighbor not in self.blocked_cells
            ):
                self.backtrack_stack.append(cell)

                self.get_logger().info(
                    f"DFS: {cell} → {neighbor} | visited={len(self.visited)}"
                )

                return neighbor

        while self.backtrack_stack:
            back_cell = self.backtrack_stack.pop()

            if (
                back_cell != self.current_cell
                and back_cell not in self.blocked_cells
            ):
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

    # ============================================================
    # MOVEMENT
    # ============================================================

    def start_move_to_cell(self, target):
        if (
            target == self.target_drop_cell
            and self.mission_phase in ["GO_TARGET_1", "GO_TARGET_2"]
        ):
            self.get_logger().warn(
                f"Refusing to drive into target cell {self.target_drop_cell}"
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

        tx, ty = self.cell_to_world(
            self.target_cell[0],
            self.target_cell[1]
        )

        dist = math.hypot(tx - self.x, ty - self.y)

        if dist <= self.CENTER_TOLERANCE:
            self.current_cell = self.target_cell
            self.visited.add(self.current_cell)
            self.graph.setdefault(self.current_cell, set())

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

    def start_follow_path(self, path, after_state):
        self.current_path = path
        self.path_index = 1
        self.after_path_state = after_state
        self.following_path = True

        self.get_logger().info(
            f"BFS path: {path}"
        )

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
        path_to_target = self.bfs(
            self.current_cell,
            self.target_drop_cell
        )

        if path_to_target is None or len(path_to_target) < 2:
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

    # ============================================================
    # ARM / GRIPPER
    # ============================================================

    def send_arm_pose(self, positions, seconds=3):
        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(seconds)
        point.time_from_start.nanosec = int(
            (seconds - int(seconds)) * 1e9
        )

        msg.points.append(point)
        self.arm_pub.publish(msg)

    def send_gripper_goal(self, position):
        if not self.gripper_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warn(
                "Gripper action server not available"
            )
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

                self.get_logger().info(
                    "Pick: gripper open"
                )

                self.manip_command_sent = True

            if self.manip_elapsed() >= 1.0:
                self.next_manip_step()

        elif self.manip_step == 1:
            result = self.get_front_object_distance_angle()

            if result is None:
                self.get_logger().warn(
                    "Pick: waiting for LiDAR object point"
                )

                # Search gently in place; useful for second pickup.
                if self.manip_elapsed() < 5.0:
                    cmd.angular.z = 0.08

                elif self.manip_elapsed() < 10.0:
                    cmd.angular.z = -0.08

                else:
                    self.stop_robot()
                    self.get_logger().error(
                        "Pick failed: no LiDAR object point found."
                    )
                    self.set_state("DONE")

                return

            distance, angle = result
            angle_deg = math.degrees(angle)

            self.last_pick_distance = distance
            self.last_pick_angle = angle

            if distance <= self.PICK_TOO_CLOSE_DIST:
                self.stop_robot()

                self.get_logger().warn(
                    f"Pick: TOO CLOSE d={distance:.3f}, "
                    f"angle={angle_deg:.1f}; using current pose"
                )

                self.next_manip_step()
                return

            if distance <= self.PICK_LIDAR_TARGET_DIST + self.PICK_LIDAR_TOLERANCE:
                self.stop_robot()

                self.get_logger().info(
                    f"Pick: distance ready d={distance:.3f}, "
                    f"angle={angle_deg:.1f}"
                )

                self.next_manip_step()
                return

            cmd.linear.x = self.PICK_APPROACH_SPEED

            if abs(angle) < self.PICK_ALIGN_DEADBAND:
                cmd.angular.z = 0.0

            else:
                cmd.angular.z = self.clamp(
                    self.PICK_ALIGN_KP * angle,
                    -self.PICK_ALIGN_MAX_W,
                    self.PICK_ALIGN_MAX_W
                )

            if self.manip_elapsed() >= self.PICK_APPROACH_MAX_TIME:
                self.stop_robot()

                self.get_logger().error(
                    f"Pick failed: could not reach pickup distance. "
                    f"d={distance:.3f}, angle={angle_deg:.1f}"
                )

                self.set_state("DONE")
                return

        elif self.manip_step == 2:
            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_OPEN)

                self.get_logger().info(
                    "Pick: gripper open before IK"
                )

                if not self.request_ik_async():
                    self.get_logger().error(
                        "Pick failed: IK service unavailable"
                    )

                    self.set_state("DONE")
                    return

                self.manip_command_sent = True
                return

            if self.ik_future is not None and self.ik_future.done():
                try:
                    response = self.ik_future.result()
                    ik_pose = self.extract_arm_joints_from_ik(response)

                except Exception as e:
                    self.get_logger().error(
                        f"MoveIt IK exception: {e}"
                    )
                    ik_pose = None

                if ik_pose is None:
                    self.get_logger().error(
                        "Pick failed: IK failed. Stopping, no fallback pose."
                    )

                    self.stop_robot()
                    self.set_state("DONE")
                    return

                self.current_pick_pose = ik_pose
                self.send_arm_pose(self.current_pick_pose, 3)

                self.get_logger().info(
                    f"Pick: arm to IK pose {self.current_pick_pose}"
                )

                self.next_manip_step()
                return

            if time.time() - self.ik_request_start_time > 3.0:
                self.get_logger().error(
                    "Pick failed: IK timeout. Stopping, no fallback pose."
                )

                self.stop_robot()
                self.set_state("DONE")
                return

        elif self.manip_step == 3:
            if self.manip_elapsed() >= 3.2:
                self.next_manip_step()

        elif self.manip_step == 4:
            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_CLOSE_STEP1)

                self.get_logger().info(
                    "Pick: gripper close step 1"
                )

                self.manip_command_sent = True

            if self.manip_elapsed() >= 1.0:
                self.next_manip_step()

        elif self.manip_step == 5:
            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_CLOSE_STEP2)

                self.get_logger().info(
                    "Pick: gripper close step 2"
                )

                self.manip_command_sent = True

            if self.manip_elapsed() >= 1.0:
                self.next_manip_step()

        elif self.manip_step == 6:
            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_CLOSE_STEP3)

                self.get_logger().info(
                    "Pick: gripper close step 3"
                )

                self.manip_command_sent = True

            if self.manip_elapsed() >= 1.0:
                self.next_manip_step()

        elif self.manip_step == 7:
            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_CLOSE_STEP4)

                self.get_logger().info(
                    f"Pick: gripper final close at {self.GRIPPER_CLOSE_STEP4}"
                )

                self.manip_command_sent = True

            # Give gazebo_grasp_fix time to attach before lifting.
            if self.manip_elapsed() >= 3.5:
                self.next_manip_step()

        elif self.manip_step == 8:
            if not self.manip_command_sent:
                self.send_arm_pose(self.ARM_DEFAULT, 5)

                self.get_logger().info(
                    "Pick: arm to high carry pose"
                )

                self.manip_command_sent = True

            if self.manip_elapsed() >= 5.2:
                self.next_manip_step()

        elif self.manip_step == 9:
            dist = self.distance_to_cell_center(
                self.object_approach_cell
            )

            if dist <= self.RETREAT_TOLERANCE:
                self.current_cell = self.object_approach_cell
                self.stop_robot()

                self.get_logger().info(
                    "Pick: returned to object approach center"
                )

                self.next_manip_step()

            else:
                cmd.linear.x = self.SPEED_BACK

        elif self.manip_step == 10:
            self.get_logger().warn(
                f"Pick sequence complete for object {self.current_object_number}"
            )

            self.set_state(self.manip_after_state)

    def run_drop_sequence(self, cmd):
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        # Step 0: move arm to drop pose
        if self.manip_step == 0:
            if not self.manip_command_sent:
                self.send_arm_pose(self.ARM_DROP, 3)

                self.get_logger().info(
                    "Drop: arm to drop pose"
                )

                self.manip_command_sent = True

            if self.manip_elapsed() >= 3.2:
                self.next_manip_step()

        # Step 1: small forward nudge while still holding object
        # This places the object deeper into the target cell.
        elif self.manip_step == 1:
            if self.manip_elapsed() < self.DROP_NUDGE_TIME:
                cmd.linear.x = self.DROP_FORWARD_SPEED
                cmd.angular.z = 0.0
            else:
                self.stop_robot()
                self.next_manip_step()

        # Step 2: open gripper and release glue
        elif self.manip_step == 2:
            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_OPEN)

                self.get_logger().info(
                    "Drop: gripper open"
                )

                self.manip_command_sent = True

            if self.manip_elapsed() >= 2.2:
                self.next_manip_step()

        # Step 3: reverse back to approach cell center
        elif self.manip_step == 3:
            if self.manip_elapsed() < self.DROP_NUDGE_TIME:
                cmd.linear.x = self.DROP_BACK_SPEED
                cmd.angular.z = 0.0
            else:
                self.stop_robot()
                self.next_manip_step()

        # Step 4: return arm to carry/default pose
        elif self.manip_step == 4:
            if not self.manip_command_sent:
                self.send_arm_pose(self.ARM_DEFAULT, 3)

                self.get_logger().info(
                    "Drop: arm to carry/default"
                )

                self.manip_command_sent = True

            if self.manip_elapsed() >= 3.2:
                self.next_manip_step()

        # Step 5: done
        elif self.manip_step == 5:
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

    # ============================================================
    # MISSION STATE MACHINE
    # ============================================================

    def after_object_retreat(self):
        self.start_face_cell(self.object_cell, "PICK_OBJECT_1")

    def control_loop(self):
        cmd = Twist()

        if self.state == "INIT_WAIT":
            if (
                self.odom_ready
                and self.scan_ready
                and self.elapsed() >= self.INIT_WAIT_TIME
            ):
                cell = self.confirmed_cell()

                if cell is not None:
                    self.current_cell = cell
                    self.visited.add(cell)
                    self.graph.setdefault(cell, set())

                    self.get_logger().info(
                        f"Start DFS at {cell}"
                    )

                    self.set_state("PAUSE_AT_CELL")

        elif self.state == "PAUSE_AT_CELL":
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
            dist = self.distance_to_cell_center(
                self.object_approach_cell
            )

            if dist <= self.RETREAT_TOLERANCE:
                self.current_cell = self.object_approach_cell
                self.stop_robot()

                self.get_logger().info(
                    f"Back at safe approach center {self.object_approach_cell}"
                )

                self.after_object_retreat()
                self.cmd_pub.publish(cmd)
                return

            cmd.linear.x = self.SPEED_BACK

        elif self.state == "PICK_OBJECT_1":
            self.current_object_number = 1
            self.start_manipulation("pick", "PLAN_TO_TARGET_1")

        elif self.state == "PLAN_TO_TARGET_1":
            self.mission_phase = "GO_TARGET_1"

            if self.plan_to_target_approach("FACE_TARGET_1"):
                self.get_logger().info(
                    "BFS: going to target approach first time"
                )

            else:
                self.get_logger().warn(
                    "No known path to target yet → continue DFS"
                )

                self.mission_phase = "SEARCH_TARGET"
                self.set_state("PAUSE_AT_CELL")

        elif self.state == "FACE_TARGET_1":
            self.stop_robot()
            self.set_state("DROP_OBJECT_1")

        elif self.state == "DROP_OBJECT_1":
            self.current_object_number = 1
            self.start_manipulation(
                "drop",
                "PLAN_BACK_TO_OBJECT"
            )

        elif self.state == "PLAN_BACK_TO_OBJECT":
            self.mission_phase = "BACK_TO_OBJECT"

            if self.plan_to_goal(
                self.object_approach_cell,
                "AT_OBJECT_AGAIN"
            ):
                self.get_logger().info(
                    "BFS: returning to object approach cell"
                )

            else:
                self.get_logger().warn(
                    "No path back to object approach cell"
                )

                self.set_state("DONE")

        elif self.state == "AT_OBJECT_AGAIN":
            if self.elapsed() >= self.PAUSE_TIME:
                self.get_logger().warn(
                    "Returned to object approach cell"
                )

                self.start_face_cell(
                    self.object_cell,
                    "PICK_OBJECT_2"
                )

        elif self.state == "PICK_OBJECT_2":
            self.current_object_number = 2
            self.start_manipulation(
                "pick",
                "PLAN_TO_TARGET_2"
            )

        elif self.state == "PLAN_TO_TARGET_2":
            self.mission_phase = "GO_TARGET_2"

            if self.plan_to_target_approach("FACE_TARGET_2"):
                self.get_logger().info(
                    "BFS: going to target approach second time"
                )

            else:
                self.get_logger().warn(
                    "No path to target second time"
                )

                self.set_state("DONE")

        elif self.state == "FACE_TARGET_2":
            self.stop_robot()
            self.set_state("DROP_OBJECT_2")

        elif self.state == "DROP_OBJECT_2":
            self.current_object_number = 2
            self.start_manipulation(
                "drop",
                "PLAN_BACK_TO_START"
            )

        elif self.state == "PLAN_BACK_TO_START":
            self.mission_phase = "RETURN_START"

            if self.plan_to_goal(
                self.start_cell,
                "RETURNED_START"
            ):
                self.get_logger().info(
                    "BFS: returning to start cell"
                )

            else:
                self.get_logger().warn(
                    "No path back to start"
                )

                self.set_state("DONE")

        elif self.state == "RETURNED_START":
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

        self.cmd_pub.publish(cmd)

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
            node.stop_robot()
        except Exception:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()



    #######################3

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

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

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

        # ============================================================
        # NAVIGATION CONSTANTS
        # ============================================================

        self.CELL_SIZE = 0.70
        self.MAZE_CELLS = 7

        self.CENTER_TOLERANCE = 0.10
        self.RETREAT_TOLERANCE = 0.11
        self.YAW_TOLERANCE = math.radians(1.0)

        self.FRONT_STOP = 0.35
        self.OPEN_DIST = 0.55

        self.SPEED_FWD = 0.25
        self.SPEED_BACK = -0.09
        self.SPEED_TURN = 0.25

        self.PAUSE_TIME = 0.6
        self.INIT_WAIT_TIME = 1.5

        # ============================================================
        # PICKUP CONSTANTS
        # ============================================================
        # KEEPING THE WORKING PICKUP VALUES

        self.PICK_LIDAR_TARGET_DIST = 0.225
        self.PICK_LIDAR_TOLERANCE = 0.015
        self.PICK_TOO_CLOSE_DIST = 0.145

        self.PICK_APPROACH_SPEED = 0.035
        self.PICK_APPROACH_MAX_TIME = 25.0

        self.PICK_ALIGN_KP = 0.16
        self.PICK_ALIGN_MAX_W = 0.020
        self.PICK_ALIGN_DEADBAND = math.radians(3.0)

        self.LIDAR_FRONT_MIN_DEG = -45
        self.LIDAR_FRONT_MAX_DEG = 45
        self.LIDAR_MAX_OBJECT_DIST = 0.85
        self.LIDAR_CLOSE_OBJECT_DIST = 0.45
        self.last_lidar_log_time = 0.0

        # ============================================================
        # ARM / GRIPPER CONSTANTS
        # ============================================================

        self.ARM_DEFAULT = [0.0, 0.15, -0.45, 0.65]
        self.ARM_DROP = [0.0, 0.65, -1.05, 0.40]

        self.ARM_START_DEFAULT = [0.0, 0.0, 0.0, 0.0]
        self.GRIPPER_START_DEFAULT = 0.010

        # Drop deeper into target cell, then reverse back to approach cell.
        self.DROP_FORWARD_SPEED = 0.055
        self.DROP_BACK_SPEED = -0.055
        self.DROP_NUDGE_TIME = 2.15

        self.GRIPPER_OPEN = 0.016

        self.GRIPPER_CLOSE_STEP1 = 0.008
        self.GRIPPER_CLOSE_STEP2 = 0.004
        self.GRIPPER_CLOSE_STEP3 = 0.000
        self.GRIPPER_CLOSE_STEP4 = -0.008
        self.GRIPPER_EFFORT = 0.55

        # ============================================================
        # MOVEIT IK CONSTANTS
        # ============================================================

        self.IK_GROUP_NAME = "arm"
        self.IK_LINK_NAME = "end_effector_link"
        self.IK_FRAME_ID = "link1"

        self.BASE_TO_LINK1_X = 0.092
        self.IK_FORWARD_OFFSET = 0.024
        self.IK_PICK_Z = 0.055
        self.IK_MAX_LATERAL_ANGLE = math.radians(12.0)

        self.last_pick_distance = self.PICK_LIDAR_TARGET_DIST
        self.last_pick_angle = 0.0

        self.ik_future = None
        self.ik_request_start_time = 0.0
        self.current_pick_pose = None

        # ============================================================
        # ROBOT STATE
        # ============================================================

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

        self.get_logger().info("Explorer started: DFS/BFS + MoveIt IK + glue grasp (PICKUP RESTORED)")
        self.get_logger().info(f"Start cell: {self.start_cell}")
        self.get_logger().info(f"Target cell: {self.target_drop_cell}")

    # ============================================================
    # CALLBACKS
    # ============================================================

    def scan_callback(self, msg):
        self.latest_scan = msg
        self.scan_ready = True

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = self.get_yaw_from_odom(msg)
        self.odom_ready = True

    # ============================================================
    # BASIC HELPERS
    # ============================================================

    def get_yaw_from_odom(self, msg):
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

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def set_state(self, state):
        self.state = state
        self.state_start = time.time()

    def elapsed(self):
        return time.time() - self.state_start

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

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
        return math.hypot(self.x - cx, self.y - cy)

    def confirmed_cell(self):
        cell = self.world_to_cell(self.x, self.y)

        if self.distance_to_cell_center(cell) <= self.CENTER_TOLERANCE:
            return cell

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

    # ============================================================
    # OBJECT DETECTION FOR PICKUP
    # ============================================================

    def get_front_object_distance_angle(self):
        if self.latest_scan is None:
            return None

        ranges = list(self.latest_scan.ranges)
        n = len(ranges)

        points = []

        for deg in range(self.LIDAR_FRONT_MIN_DEG, self.LIDAR_FRONT_MAX_DEG + 1):
            idx = int((deg % 360) * n / 360)
            r = ranges[idx]

            if math.isnan(r) or math.isinf(r):
                continue

            if r < 0.04 or r > self.LIDAR_MAX_OBJECT_DIST:
                continue

            points.append((deg, r))

        if not points:
            return None

        clusters = []
        current = [points[0]]

        for p in points[1:]:
            prev_deg, prev_r = current[-1]
            deg, r = p

            same_cluster = (
                abs(deg - prev_deg) <= 2
                and abs(r - prev_r) <= 0.08
            )

            if same_cluster:
                current.append(p)
            else:
                if len(current) >= 2:
                    clusters.append(current)
                current = [p]

        if len(current) >= 2:
            clusters.append(current)

        if not clusters:
            best_deg, best_dist = min(points, key=lambda p: p[1])
            return best_dist, math.radians(best_deg)

        candidates = []

        for cluster in clusters:
            degs = [p[0] for p in cluster]
            dists = [p[1] for p in cluster]

            candidates.append({
                "angle_deg": sum(degs) / len(degs),
                "distance": min(dists),
                "size": len(cluster)
            })

        object_like = []

        for c in candidates:
            # Object cluster should not be huge.
            # Large clusters are usually wall / corridor / opening edge.
            if c["size"] > 12:
                continue

            # Tiny 1-point noise was already mostly removed, but keep this safe.
            if c["size"] < 2:
                continue

            object_like.append(c)

        if object_like:
            candidates = object_like
        else:
            # Fallback: if all clusters are large, keep the smallest cluster,
            # because a stick is more likely to be the smallest visible thing.
            min_size = min(c["size"] for c in candidates)
            candidates = [c for c in candidates if c["size"] == min_size]

        # Prefer close object-like clusters only after wall rejection.
        close_candidates = [
            c for c in candidates
            if c["distance"] <= self.LIDAR_CLOSE_OBJECT_DIST
        ]

        if close_candidates:
            candidates = close_candidates

        # Choose nearest object-like cluster.
        # Do NOT center using wall/opening clusters.
        chosen = min(candidates, key=lambda c: c["distance"])

        now = time.time()
        if now - self.last_lidar_log_time >= 0.7:
            self.last_lidar_log_time = now
            debug_list = [
                (round(c["angle_deg"], 1), round(c["distance"], 3), c["size"])
                for c in candidates
            ]

            self.get_logger().info(
            f"LiDAR object-like clusters: {debug_list} | "
            f"selected nearest object: angle={chosen['angle_deg']:.1f}, "
            f"dist={chosen['distance']:.3f}, size={chosen['size']}"
        )

        return chosen["distance"], math.radians(chosen["angle_deg"])

    # ============================================================
    # MOVEIT IK
    # ============================================================

    def build_ik_request(self):
        distance = self.last_pick_distance
        raw_angle = self.last_pick_angle

        angle = self.clamp(
            raw_angle,
            -self.IK_MAX_LATERAL_ANGLE,
            self.IK_MAX_LATERAL_ANGLE
        )

        x_scan = distance * math.cos(angle)
        y_scan = distance * math.sin(angle)

        x_link1 = x_scan + self.BASE_TO_LINK1_X + self.IK_FORWARD_OFFSET
        y_link1 = y_scan
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
            f"distance={distance:.3f}, "
            f"raw_angle={math.degrees(raw_angle):.1f}, "
            f"used_angle={math.degrees(angle):.1f}"
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

        joint_map = dict(zip(names, positions))
        required = ['joint1', 'joint2', 'joint3', 'joint4']

        for joint in required:
            if joint not in joint_map:
                self.get_logger().warn(
                    f"IK result missing {joint}. Available joints: {names}"
                )
                return None

        pose = [joint_map[j] for j in required]

        self.get_logger().info(
            f"IK solution: {[round(v, 4) for v in pose]}"
        )

        return pose

    def request_ik_async(self):
        if not self.ik_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().error("/compute_ik service not available")
            return False

        req = self.build_ik_request()
        self.ik_future = self.ik_client.call_async(req)
        self.ik_request_start_time = time.time()

        return True

    # ============================================================
    # GRID / GRAPH
    # ============================================================

    def yaw_to_heading(self):
        yaw = self.normalize_angle(self.yaw)

        options = {
            "E": 0.0,
            "N": math.pi / 2,
            "W": math.pi,
            "S": -math.pi / 2
        }

        return min(
            options,
            key=lambda h: abs(self.normalize_angle(yaw - options[h]))
        )

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

        return (
            1 <= row <= self.MAZE_CELLS
            and 1 <= col <= self.MAZE_CELLS
        )

    def direction_to_yaw(self, direction):
        return {
            "E": 0.0,
            "N": math.pi / 2,
            "W": math.pi,
            "S": -math.pi / 2
        }.get(direction, 0.0)

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

    def add_edge(self, a, b):
        if a in self.blocked_cells or b in self.blocked_cells:
            return

        self.graph.setdefault(a, set())
        self.graph.setdefault(b, set())

        self.graph[a].add(b)
        self.graph[b].add(a)

    def remove_edge(self, a, b):
        if a in self.graph:
            self.graph[a].discard(b)

        if b in self.graph:
            self.graph[b].discard(a)

    def detect_open_neighbors(self, cell):
        heading = self.yaw_to_heading()

        checks = [
            (heading, self.get_zone(0, 40)),
            (self.left_of(heading), self.get_zone(90, 40)),
            (self.right_of(heading), self.get_zone(270, 40)),
            (self.back_of(heading), self.get_zone(180, 40)),
        ]

        neighbors = []

        for direction, dist in checks:
            if dist > self.OPEN_DIST:
                neighbor = self.neighbor_cell(cell, direction)

                if (
                    self.inside_maze(neighbor)
                    and neighbor not in self.blocked_cells
                ):
                    neighbors.append(neighbor)
                    self.add_edge(cell, neighbor)

        return neighbors

    def choose_next_cell(self):
        cell = self.current_cell
        self.visited.add(cell)

        for neighbor in self.detect_open_neighbors(cell):
            if (
                neighbor not in self.visited
                and neighbor not in self.blocked_cells
            ):
                self.backtrack_stack.append(cell)

                self.get_logger().info(
                    f"DFS: {cell} → {neighbor} | visited={len(self.visited)}"
                )

                return neighbor

        while self.backtrack_stack:
            back_cell = self.backtrack_stack.pop()

            if (
                back_cell != self.current_cell
                and back_cell not in self.blocked_cells
            ):
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

    # ============================================================
    # MOVEMENT
    # ============================================================

    def start_move_to_cell(self, target):
        if (
            target == self.target_drop_cell
            and self.mission_phase in ["GO_TARGET_1", "GO_TARGET_2"]
        ):
            self.get_logger().warn(
                f"Refusing to drive into target cell {self.target_drop_cell}"
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

        if self.mission_phase == "GO_TARGET_2":
            self.get_logger().warn(
                "Second delivery blocked near target. Dropping object 2 here instead of failing."
            )

            self.following_path = False
            self.current_path = []
            self.path_index = 0

            self.stop_robot()
            self.set_state("DROP_OBJECT_2")
            return

        self.remove_edge(self.current_cell, self.target_cell)

        self.following_path = False
        self.current_path = []
        self.path_index = 0

        if self.mission_phase == "GO_TARGET_1":
            self.set_state("PLAN_TO_TARGET_1")

        elif self.mission_phase == "BACK_TO_OBJECT":
            self.set_state("PLAN_BACK_TO_OBJECT")

        elif self.mission_phase == "RETURN_START":
            self.set_state("PLAN_BACK_TO_START")

        else:
            self.set_state("PAUSE_AT_CELL")

    def move_to_target(self, cmd):
        if self.target_cell is None:
            self.set_state("PAUSE_AT_CELL")
            return

        # Less sensitive while carrying, because arm/object can appear in LiDAR.
        if self.mission_phase in ["GO_TARGET_1", "GO_TARGET_2", "BACK_TO_OBJECT", "RETURN_START"]:
            front = self.get_zone(0, 30)
            front_stop = 0.24
        else:
            front = self.get_zone(0, 40)
            front_stop = self.FRONT_STOP

        if front < front_stop:
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

        tx, ty = self.cell_to_world(
            self.target_cell[0],
            self.target_cell[1]
        )

        dist = math.hypot(tx - self.x, ty - self.y)

        if dist <= self.CENTER_TOLERANCE:
            self.current_cell = self.target_cell
            self.visited.add(self.current_cell)
            self.graph.setdefault(self.current_cell, set())

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

    def start_follow_path(self, path, after_state):
        self.current_path = path
        self.path_index = 1
        self.after_path_state = after_state
        self.following_path = True

        self.get_logger().info(
            f"BFS path: {path}"
        )

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
        path_to_target = self.bfs(
            self.current_cell,
            self.target_drop_cell
        )

        if path_to_target is None or len(path_to_target) < 2:
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

    # ============================================================
    # ARM / GRIPPER
    # ============================================================

    def send_arm_pose(self, positions, seconds=3):
        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(seconds)
        point.time_from_start.nanosec = int(
            (seconds - int(seconds)) * 1e9
        )

        msg.points.append(point)
        self.arm_pub.publish(msg)

    def send_gripper_goal(self, position):
        if not self.gripper_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warn(
                "Gripper action server not available"
            )
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

                self.get_logger().info(
                    "Pick: gripper open"
                )

                self.manip_command_sent = True

            if self.manip_elapsed() >= 1.0:
                self.next_manip_step()

        elif self.manip_step == 1:
            result = self.get_front_object_distance_angle()

            if result is None:
                self.get_logger().warn(
                    "Pick: waiting for LiDAR object point"
                )

                if self.manip_elapsed() < 5.0:
                    cmd.angular.z = 0.08

                elif self.manip_elapsed() < 10.0:
                    cmd.angular.z = -0.08

                else:
                    self.stop_robot()
                    self.get_logger().error(
                        "Pick failed: no LiDAR object point found."
                    )
                    self.set_state("DONE")

                return

            distance, angle = result
            angle_deg = math.degrees(angle)

            self.last_pick_distance = distance
            self.last_pick_angle = angle

            if distance <= self.PICK_TOO_CLOSE_DIST:
                self.stop_robot()

                self.get_logger().warn(
                    f"Pick: TOO CLOSE d={distance:.3f}, "
                    f"angle={angle_deg:.1f}; using current pose"
                )

                self.next_manip_step()
                return

            if distance <= self.PICK_LIDAR_TARGET_DIST + self.PICK_LIDAR_TOLERANCE:
                self.stop_robot()

                self.get_logger().info(
                    f"Pick: distance ready d={distance:.3f}, "
                    f"angle={angle_deg:.1f}"
                )

                self.next_manip_step()
                return

            cmd.linear.x = self.PICK_APPROACH_SPEED

            if abs(angle) < self.PICK_ALIGN_DEADBAND:
                cmd.angular.z = 0.0

            else:
                cmd.angular.z = self.clamp(
                    self.PICK_ALIGN_KP * angle,
                    -self.PICK_ALIGN_MAX_W,
                    self.PICK_ALIGN_MAX_W
                )

            if self.manip_elapsed() >= self.PICK_APPROACH_MAX_TIME:
                self.stop_robot()

                self.get_logger().error(
                    f"Pick failed: could not reach pickup distance. "
                    f"d={distance:.3f}, angle={angle_deg:.1f}"
                )

                self.set_state("DONE")
                return

        elif self.manip_step == 2:
            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_OPEN)

                self.get_logger().info(
                    "Pick: gripper open before IK"
                )

                if not self.request_ik_async():
                    self.get_logger().error(
                        "Pick failed: IK service unavailable"
                    )

                    self.set_state("DONE")
                    return

                self.manip_command_sent = True
                return

            if self.ik_future is not None and self.ik_future.done():
                try:
                    response = self.ik_future.result()
                    ik_pose = self.extract_arm_joints_from_ik(response)

                except Exception as e:
                    self.get_logger().error(
                        f"MoveIt IK exception: {e}"
                    )
                    ik_pose = None

                if ik_pose is None:
                    self.get_logger().error(
                        "Pick failed: IK failed. Stopping, no fallback pose."
                    )

                    self.stop_robot()
                    self.set_state("DONE")
                    return

                self.current_pick_pose = ik_pose
                self.send_arm_pose(self.current_pick_pose, 3)

                self.get_logger().info(
                    f"Pick: arm to IK pose {self.current_pick_pose}"
                )

                self.next_manip_step()
                return

            if time.time() - self.ik_request_start_time > 3.0:
                self.get_logger().error(
                    "Pick failed: IK timeout. Stopping, no fallback pose."
                )

                self.stop_robot()
                self.set_state("DONE")
                return

        elif self.manip_step == 3:
            self.stop_robot()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if self.manip_elapsed() >= 3.2:
                self.next_manip_step()

        elif self.manip_step == 4:
            self.stop_robot()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_CLOSE_STEP1)

                self.get_logger().info(
                    "Pick: gripper close step 1"
                )

                self.manip_command_sent = True

            if self.manip_elapsed() >= 1.0:
                self.next_manip_step()

        elif self.manip_step == 5:
            self.stop_robot()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_CLOSE_STEP2)

                self.get_logger().info(
                    "Pick: gripper close step 2"
                )

                self.manip_command_sent = True

            if self.manip_elapsed() >= 1.0:
                self.next_manip_step()

        elif self.manip_step == 6:
            self.stop_robot()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_CLOSE_STEP3)

                self.get_logger().info(
                    "Pick: gripper close step 3"
                )

                self.manip_command_sent = True

            if self.manip_elapsed() >= 1.0:
                self.next_manip_step()

        elif self.manip_step == 7:
            self.stop_robot()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_CLOSE_STEP4)

                self.get_logger().info(
                    f"Pick: gripper final close at {self.GRIPPER_CLOSE_STEP4}"
                )

                self.manip_command_sent = True

            if self.manip_elapsed() >= 3.5:
                self.next_manip_step()

        elif self.manip_step == 8:
            self.stop_robot()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if not self.manip_command_sent:
                self.send_arm_pose(self.ARM_DEFAULT, 5)

                self.get_logger().info(
                    "Pick: arm to high carry pose"
                )

                self.manip_command_sent = True

            if self.manip_elapsed() >= 5.2:
                self.next_manip_step()

        elif self.manip_step == 9:
            dist = self.distance_to_cell_center(
                self.object_approach_cell
            )

            if dist <= self.RETREAT_TOLERANCE:
                self.current_cell = self.object_approach_cell
                self.stop_robot()

                self.get_logger().info(
                    "Pick: returned to object approach center"
                )

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
            if self.manip_elapsed() < self.DROP_NUDGE_TIME:
                cmd.linear.x = self.DROP_FORWARD_SPEED
                cmd.angular.z = 0.0
            else:
                self.stop_robot()
                self.get_logger().info("Drop: forward nudge complete")
                self.next_manip_step()

        elif self.manip_step == 2:
            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_OPEN)
                self.get_logger().info("Drop: gripper open")
                self.manip_command_sent = True

            if self.manip_elapsed() >= 2.2:
                self.next_manip_step()

        elif self.manip_step == 3:
            if self.manip_elapsed() < self.DROP_NUDGE_TIME:
                cmd.linear.x = self.DROP_BACK_SPEED
                cmd.angular.z = 0.0
            else:
                self.stop_robot()
                self.get_logger().info("Drop: reversed back to approach")
                self.next_manip_step()

        elif self.manip_step == 4:
            if not self.manip_command_sent:
                self.send_arm_pose(self.ARM_START_DEFAULT, 3)
                self.get_logger().info("Drop: arm to exact startup/default pose")
                self.manip_command_sent = True

            if self.manip_elapsed() >= 3.2:
                self.next_manip_step()

        elif self.manip_step == 5:
            if not self.manip_command_sent:
                self.send_gripper_goal(self.GRIPPER_START_DEFAULT)
                self.get_logger().info("Drop: gripper to startup/default")
                self.manip_command_sent = True

            if self.manip_elapsed() >= 1.2:
                self.next_manip_step()

        elif self.manip_step == 6:
            if self.target_approach_cell is not None:
                self.current_cell = self.target_approach_cell

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

    # ============================================================
    # MISSION STATE MACHINE
    # ============================================================

    def after_object_retreat(self):
        self.start_face_cell(self.object_cell, "PICK_OBJECT_1")

    def control_loop(self):
        cmd = Twist()

        if self.state == "INIT_WAIT":
            if (
                self.odom_ready
                and self.scan_ready
                and self.elapsed() >= self.INIT_WAIT_TIME
            ):
                cell = self.confirmed_cell()

                if cell is not None:
                    self.current_cell = cell
                    self.visited.add(cell)
                    self.graph.setdefault(cell, set())

                    self.get_logger().info(
                        f"Start DFS at {cell}"
                    )

                    self.set_state("PAUSE_AT_CELL")

        elif self.state == "PAUSE_AT_CELL":
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
            dist = self.distance_to_cell_center(
                self.object_approach_cell
            )

            if dist <= self.RETREAT_TOLERANCE:
                self.current_cell = self.object_approach_cell
                self.stop_robot()

                self.get_logger().info(
                    f"Back at safe approach center {self.object_approach_cell}"
                )

                self.after_object_retreat()
                self.cmd_pub.publish(cmd)
                return

            cmd.linear.x = self.SPEED_BACK
            cmd.angular.z = 0.0

        elif self.state == "PICK_OBJECT_1":
            self.current_object_number = 1
            self.start_manipulation("pick", "PLAN_TO_TARGET_1")

        elif self.state == "PLAN_TO_TARGET_1":
            self.mission_phase = "GO_TARGET_1"

            if self.plan_to_target_approach("FACE_TARGET_1"):
                self.get_logger().info(
                    "BFS: going to target approach first time"
                )

            else:
                self.get_logger().warn(
                    "No known path to target yet → continue DFS"
                )

                self.mission_phase = "SEARCH_TARGET"
                self.set_state("PAUSE_AT_CELL")

        elif self.state == "FACE_TARGET_1":
            self.stop_robot()
            self.start_face_cell(self.target_drop_cell, "DROP_OBJECT_1")

        elif self.state == "DROP_OBJECT_1":
            self.current_object_number = 1
            self.start_manipulation(
                "drop",
                "PLAN_BACK_TO_OBJECT"
            )

        elif self.state == "PLAN_BACK_TO_OBJECT":
            self.mission_phase = "BACK_TO_OBJECT"

            if self.plan_to_goal(
                self.object_approach_cell,
                "AT_OBJECT_AGAIN"
            ):
                self.get_logger().info(
                    "BFS: returning to object approach cell"
                )

            else:
                self.get_logger().warn(
                    "No path back to object approach cell"
                )

                self.set_state("DONE")

        elif self.state == "AT_OBJECT_AGAIN":
            if self.elapsed() >= self.PAUSE_TIME:
                self.get_logger().warn(
                    "Returned to object approach cell"
                )

                self.start_face_cell(
                    self.object_cell,
                    "PICK_OBJECT_2"
                )

        elif self.state == "PICK_OBJECT_2":
            self.current_object_number = 2
            self.start_manipulation(
                "pick",
                "PLAN_TO_TARGET_2"
            )

        elif self.state == "PLAN_TO_TARGET_2":
            self.mission_phase = "GO_TARGET_2"

            if self.plan_to_target_approach("FACE_TARGET_2"):
                self.get_logger().info(
                    "BFS: going to target approach second time"
                )

            else:
                self.get_logger().warn(
                    "No path to target second time"
                )

                self.set_state("DONE")

        elif self.state == "FACE_TARGET_2":
            self.stop_robot()
            self.start_face_cell(self.target_drop_cell, "DROP_OBJECT_2")

        elif self.state == "DROP_OBJECT_2":
            self.current_object_number = 2
            self.start_manipulation(
                "drop",
                "PLAN_BACK_TO_START"
            )

        elif self.state == "PLAN_BACK_TO_START":
            self.mission_phase = "RETURN_START"

            if self.plan_to_goal(
                self.start_cell,
                "RETURNED_START"
            ):
                self.get_logger().info(
                    "BFS: returning to start cell"
                )

            else:
                self.get_logger().warn(
                    "No path back to start"
                )

                self.set_state("DONE")

        elif self.state == "RETURNED_START":
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

        self.cmd_pub.publish(cmd)

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
            node.stop_robot()
        except Exception:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()