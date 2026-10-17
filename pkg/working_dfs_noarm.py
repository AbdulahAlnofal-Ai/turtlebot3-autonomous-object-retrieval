import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math
import time


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
        self.PLACE_TIME = 1.0
        self.INIT_WAIT_TIME = 1.5

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

        self.state = "INIT_WAIT"
        self.state_start = time.time()

        self.get_logger().info("DFS + A* explorer started ✓")
        self.get_logger().info(f"Start cell: {self.start_cell}")
        self.get_logger().info(f"Target cell: {self.target_drop_cell}")

    def scan_callback(self, msg):
        self.latest_scan = msg
        self.scan_ready = True

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = self.get_yaw(msg)
        self.odom_ready = True

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
            slice_ = ranges[start:end]
        else:
            slice_ = ranges[start:] + ranges[:end]

        valid = [
            r for r in slice_
            if not math.isnan(r) and not math.isinf(r) and r > 0.01
        ]

        return min(valid) if valid else float('inf')

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

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reconstruct_path(self, came_from, current):
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    def astar(self, start, goal, extra_blocked=None):
        if start not in self.graph:
            return None

        if goal not in self.graph:
            return None

        if extra_blocked is None:
            extra_blocked = set()

        open_set = [start]
        came_from = {}

        g_score = {}
        f_score = {}

        for cell in self.graph:
            g_score[cell] = float('inf')
            f_score[cell] = float('inf')

        g_score[start] = 0
        f_score[start] = self.heuristic(start, goal)

        while open_set:
            current = min(
                open_set,
                key=lambda cell: f_score.get(cell, float('inf'))
            )

            if current == goal:
                return self.reconstruct_path(came_from, current)

            open_set.remove(current)

            for neighbor in self.graph.get(current, []):
                if neighbor in self.blocked_cells:
                    continue

                if neighbor in extra_blocked and neighbor != goal:
                    continue

                tentative_g = g_score[current] + 1

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)

                    if neighbor not in open_set:
                        open_set.append(neighbor)

        return None

    def start_move_to_cell(self, target):
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

        self.get_logger().info(f"A* path: {path}")

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

        path = self.astar(self.current_cell, goal, extra_blocked)

        if path is None:
            return False

        self.start_follow_path(path, after_state)
        return True

    def plan_to_target_approach(self, face_state):
        path_to_target = self.astar(self.current_cell, self.target_drop_cell)

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

        if self.astar(self.current_cell, self.target_drop_cell) is not None:
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

    def after_object_retreat(self):
        if self.astar(self.current_cell, self.target_drop_cell) is not None:
            self.set_state("PLAN_TO_TARGET_1")
        else:
            self.get_logger().warn(
                "Target not in discovered graph yet → continue DFS safely"
            )
            self.mission_phase = "SEARCH_TARGET"
            self.set_state("PAUSE_AT_CELL")

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

        elif self.state == "PLAN_TO_TARGET_1":
            self.mission_phase = "GO_TARGET_1"

            if self.plan_to_target_approach("FACE_TARGET_1"):
                self.get_logger().info("A*: going to target approach first time")
            else:
                self.get_logger().warn(
                    "No known path to target yet → continue DFS"
                )
                self.mission_phase = "SEARCH_TARGET"
                self.set_state("PAUSE_AT_CELL")

        elif self.state == "FACE_TARGET_1":
            self.start_face_cell(self.target_drop_cell, "AT_TARGET_1")

        elif self.state == "AT_TARGET_1":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if self.elapsed() >= self.PLACE_TIME:
                self.get_logger().warn(
                    f"Placed object 1 from approach cell {self.current_cell}, "
                    f"facing target {self.target_drop_cell}"
                )
                self.set_state("PLAN_BACK_TO_OBJECT")

        elif self.state == "PLAN_BACK_TO_OBJECT":
            self.mission_phase = "BACK_TO_OBJECT"

            if self.plan_to_goal(self.object_approach_cell, "AT_OBJECT_AGAIN"):
                self.get_logger().info("A*: returning to object approach cell")
            else:
                self.get_logger().warn("No path back to object approach cell")
                self.set_state("DONE")

        elif self.state == "AT_OBJECT_AGAIN":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if self.elapsed() >= self.PAUSE_TIME:
                self.get_logger().warn("Returned to object approach cell")
                self.set_state("PLAN_TO_TARGET_2")

        elif self.state == "PLAN_TO_TARGET_2":
            self.mission_phase = "GO_TARGET_2"

            if self.plan_to_target_approach("FACE_TARGET_2"):
                self.get_logger().info("A*: going to target approach second time")
            else:
                self.get_logger().warn("No path to target second time")
                self.set_state("DONE")

        elif self.state == "FACE_TARGET_2":
            self.start_face_cell(self.target_drop_cell, "AT_TARGET_2")

        elif self.state == "AT_TARGET_2":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if self.elapsed() >= self.PLACE_TIME:
                self.get_logger().warn(
                    f"Placed object 2 from approach cell {self.current_cell}, "
                    f"facing target {self.target_drop_cell}"
                )
                self.set_state("PLAN_BACK_TO_START")

        elif self.state == "PLAN_BACK_TO_START":
            self.mission_phase = "RETURN_START"

            if self.plan_to_goal(self.start_cell, "RETURNED_START"):
                self.get_logger().info("A*: returning to start cell")
            else:
                self.get_logger().warn("No path back to start")
                self.set_state("DONE")

        elif self.state == "RETURNED_START":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if self.elapsed() >= self.PAUSE_TIME:
                self.get_logger().warn(
                    f"Mission test complete. Returned to {self.start_cell}"
                )
                self.set_state("DONE")

        elif self.state == "ROTATE_TO_TARGET":
            self.rotate_to_target(cmd)

        elif self.state == "ROTATE_IN_PLACE_TO_FACE":
            self.rotate_in_place_to_target(cmd)

        elif self.state == "MOVE_TO_TARGET":
            self.move_to_target(cmd)

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
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()