import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import time


class WallAvoider(Node):

    def __init__(self):
        super().__init__('wall_avoider')

        self.sub_scan = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.sub_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # ======================================================
        # DISTANCES
        # ======================================================
        self.FRONT_STOP = 0.35
        self.SIDE_BLOCK = 0.45
        self.SIDE_TOO_CLOSE = 0.20

        # ======================================================
        # OPENING DETECTION
        # ======================================================
        self.OPEN_DIST = 1.4
        self.OPENING_COUNT_LIMIT = 8
        self.opening_count = 0

        # ======================================================
        # SPEEDS
        # ======================================================
        self.SPEED_FWD = 0.16
        self.SPEED_BACK = -0.08
        self.SPEED_TURN = 0.25

        # ======================================================
        # SMOOTH SIDE SPACING
        # ======================================================
        self.SIDE_SPACE_SPEED = 0.09

        # Slightly increased
        self.SIDE_SPACE_TURN = 0.055

        # Slightly increased
        self.SIDE_SPACE_TIME = 0.62

        self.space_dir = 0

        self.side_space_cooldown = 0
        self.SIDE_SPACE_COOLDOWN_LIMIT = 35

        # ======================================================
        # TIMING
        # ======================================================
        self.STOP_TIME = 0.6
        self.BACK_TIME = 0.50

        # Initial wait to let odom settle
        self.INIT_WAIT_TIME = 1.5

        # ======================================================
        # ODOM ROTATION
        # ======================================================
        self.current_yaw = 0.0
        self.start_yaw = 0.0
        self.odom_ready = False

        self.target_angle = math.pi / 2
        self.turn_dir = 1

        self.ANGLE_TOLERANCE = math.radians(1)

        # ======================================================
        # OPENING IGNORE AFTER TURN
        # ======================================================
        self.ignore_opening_count = 0
        self.IGNORE_OPENING_AFTER_TURN = 20

        # ======================================================
        # STATE MACHINE
        # ======================================================
        self.state = "INIT_WAIT"
        self.state_start = time.time()

        self.get_logger().info('Wall avoider started ✓')

    # ==========================================================
    # ODOM CALLBACK
    # ==========================================================
    def odom_callback(self, msg):
        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (
            q.w * q.z +
            q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y * q.y +
            q.z * q.z
        )

        self.current_yaw = math.atan2(
            siny_cosp,
            cosy_cosp
        )

        self.odom_ready = True

    # ==========================================================
    # ANGLE HELPERS
    # ==========================================================
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    def turned_angle(self):
        diff = self.normalize_angle(
            self.current_yaw - self.start_yaw
        )

        return abs(diff)

    # ==========================================================
    # STATE HELPERS
    # ==========================================================
    def set_state(self, new_state):
        self.state = new_state
        self.state_start = time.time()

    def elapsed(self):
        return time.time() - self.state_start

    def stop_robot(self):
        self.pub.publish(Twist())

    def start_rotation(self, angle, direction):
        self.start_yaw = self.current_yaw
        self.target_angle = angle
        self.turn_dir = direction

        self.set_state("ROTATE")

    # ==========================================================
    # LIDAR ZONE
    # ==========================================================
    def get_zone(self, ranges, center_deg, span_deg):
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

        valid = []

        for r in slice_:
            if math.isnan(r):
                continue

            elif math.isinf(r):
                valid.append(10.0)

            elif r > 0.01:
                valid.append(r)

        return min(valid) if valid else 10.0

    # ==========================================================
    # TURN DECISION FOR WALLS
    # ==========================================================
    def choose_turn_for_wall(self, left, right):
        left_opening = left > self.OPEN_DIST
        right_opening = right > self.OPEN_DIST

        if left_opening and not right_opening:
            self.get_logger().warn(
                "Wall ahead + LEFT opening → turn RIGHT safe side"
            )

            return math.pi / 2, -1

        elif right_opening and not left_opening:
            self.get_logger().warn(
                "Wall ahead + RIGHT opening → turn LEFT safe side"
            )

            return math.pi / 2, 1

        elif left_opening and right_opening:
            self.get_logger().warn(
                "Wall ahead + both sides openings → turn 180"
            )

            return math.pi, 1

        left_blocked = left < self.SIDE_BLOCK
        right_blocked = right < self.SIDE_BLOCK

        if left_blocked and right_blocked:
            self.get_logger().warn(
                "Front + both sides blocked → 180 turn"
            )

            return math.pi, 1

        if right > left:
            self.get_logger().warn(
                "Wall ahead → RIGHT 90"
            )

            return math.pi / 2, -1

        else:
            self.get_logger().warn(
                "Wall ahead → LEFT 90"
            )

            return math.pi / 2, 1

    # ==========================================================
    # TURN DECISION FOR OPENINGS
    # ==========================================================
    def choose_turn_for_opening(self, left, right):
        left_opening = left > self.OPEN_DIST
        right_opening = right > self.OPEN_DIST

        if left_opening and not right_opening:
            self.get_logger().warn(
                "Opening ahead + LEFT opening → RIGHT safe side"
            )

            return math.pi / 2, -1

        elif right_opening and not left_opening:
            self.get_logger().warn(
                "Opening ahead + RIGHT opening → LEFT safe side"
            )

            return math.pi / 2, 1

        elif left_opening and right_opening:
            self.get_logger().warn(
                "Opening ahead + both sides openings → 180"
            )

            return math.pi, 1

        else:
            if right > left:
                self.get_logger().warn(
                    "Opening ahead → RIGHT"
                )

                return math.pi / 2, -1

            else:
                self.get_logger().warn(
                    "Opening ahead → LEFT"
                )

                return math.pi / 2, 1

    # ==========================================================
    # MAIN CALLBACK
    # ==========================================================
    def scan_callback(self, msg):
        ranges = list(msg.ranges)

        # ======================================================
        # ZONES
        # ======================================================
        front = self.get_zone(
            ranges,
            0,
            60
        )

        left = self.get_zone(
            ranges,
            90,
            70
        )

        right = self.get_zone(
            ranges,
            270,
            70
        )

        left_side = self.get_zone(
            ranges,
            90,
            35
        )

        right_side = self.get_zone(
            ranges,
            270,
            35
        )

        front_center = self.get_zone(
            ranges,
            0,
            20
        )

        front_left = self.get_zone(
            ranges,
            25,
            20
        )

        front_right = self.get_zone(
            ranges,
            335,
            20
        )

        front_is_really_open = (
            front_center > self.OPEN_DIST and
            front_left > self.OPEN_DIST and
            front_right > self.OPEN_DIST
        )

        cmd = Twist()

        # ======================================================
        # INITIAL WAIT
        # ======================================================
        if self.state == "INIT_WAIT":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if self.odom_ready and self.elapsed() >= self.INIT_WAIT_TIME:
                self.get_logger().info("Odom ready → starting navigation")
                self.set_state("FORWARD")

        # ======================================================
        # FORWARD STATE
        # ======================================================
        elif self.state == "FORWARD":

            if self.ignore_opening_count > 0:
                self.ignore_opening_count -= 1

            if self.side_space_cooldown > 0:
                self.side_space_cooldown -= 1

            # --------------------------------------------------
            # FRONT WALL
            # --------------------------------------------------
            if front < self.FRONT_STOP:
                self.opening_count = 0
                self.stop_robot()

                angle, direction = self.choose_turn_for_wall(
                    left,
                    right
                )

                self.target_angle = angle
                self.turn_dir = direction

                self.set_state("STOP_BEFORE_BACK")
                return

            # --------------------------------------------------
            # FRONT OPENING
            # --------------------------------------------------
            elif (
                self.ignore_opening_count == 0 and
                front_is_really_open
            ):
                self.opening_count += 1

                if self.opening_count >= self.OPENING_COUNT_LIMIT:
                    self.opening_count = 0
                    self.stop_robot()

                    angle, direction = self.choose_turn_for_opening(
                        left,
                        right
                    )

                    self.target_angle = angle
                    self.turn_dir = direction

                    self.set_state(
                        "STOP_AFTER_OPENING_DETECTED"
                    )

                    return

                cmd.linear.x = self.SPEED_FWD * 0.3
                cmd.angular.z = 0.0

            # --------------------------------------------------
            # TOO CLOSE LEFT
            # --------------------------------------------------
            elif (
                left_side < self.SIDE_TOO_CLOSE and
                self.side_space_cooldown == 0
            ):
                self.opening_count = 0
                self.space_dir = -1

                self.get_logger().info(
                    "Too close LEFT → smooth spacing RIGHT"
                )

                self.set_state("SIDE_SPACE")
                return

            # --------------------------------------------------
            # TOO CLOSE RIGHT
            # --------------------------------------------------
            elif (
                right_side < self.SIDE_TOO_CLOSE and
                self.side_space_cooldown == 0
            ):
                self.opening_count = 0
                self.space_dir = 1

                self.get_logger().info(
                    "Too close RIGHT → smooth spacing LEFT"
                )

                self.set_state("SIDE_SPACE")
                return

            # --------------------------------------------------
            # NORMAL FORWARD
            # --------------------------------------------------
            else:
                self.opening_count = 0

                cmd.linear.x = self.SPEED_FWD
                cmd.angular.z = 0.0

        # ======================================================
        # SIDE SPACING
        # ======================================================
        elif self.state == "SIDE_SPACE":
            cmd.linear.x = self.SIDE_SPACE_SPEED
            cmd.angular.z = (
                self.space_dir *
                self.SIDE_SPACE_TURN
            )

            if self.elapsed() >= self.SIDE_SPACE_TIME:
                self.side_space_cooldown = (
                    self.SIDE_SPACE_COOLDOWN_LIMIT
                )

                self.set_state(
                    "STOP_AFTER_SIDE_SPACE"
                )

        # ======================================================
        # STOP AFTER SIDE SPACE
        # ======================================================
        elif self.state == "STOP_AFTER_SIDE_SPACE":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if self.elapsed() >= 0.25:
                self.set_state("FORWARD")

        # ======================================================
        # STOP AFTER OPENING DETECTED
        # ======================================================
        elif self.state == "STOP_AFTER_OPENING_DETECTED":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if self.elapsed() >= self.STOP_TIME:
                self.start_rotation(
                    self.target_angle,
                    self.turn_dir
                )

        # ======================================================
        # STOP BEFORE BACK
        # ======================================================
        elif self.state == "STOP_BEFORE_BACK":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if self.elapsed() >= self.STOP_TIME:
                self.set_state("BACKWARD")

        # ======================================================
        # BACKWARD
        # ======================================================
        elif self.state == "BACKWARD":
            cmd.linear.x = self.SPEED_BACK
            cmd.angular.z = 0.0

            if self.elapsed() >= self.BACK_TIME:
                self.set_state("STOP_AFTER_BACK")

        # ======================================================
        # STOP AFTER BACK
        # ======================================================
        elif self.state == "STOP_AFTER_BACK":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if self.elapsed() >= self.STOP_TIME:
                self.start_rotation(
                    self.target_angle,
                    self.turn_dir
                )

        # ======================================================
        # ROTATE
        # ======================================================
        elif self.state == "ROTATE":
            cmd.linear.x = 0.0
            cmd.angular.z = (
                self.turn_dir *
                self.SPEED_TURN
            )

            if (
                self.turned_angle() >=
                self.target_angle -
                self.ANGLE_TOLERANCE
            ):
                cmd.angular.z = 0.0
                self.set_state("STOP_AFTER_TURN")

        # ======================================================
        # STOP AFTER TURN
        # ======================================================
        elif self.state == "STOP_AFTER_TURN":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            if self.elapsed() >= self.STOP_TIME:
                self.ignore_opening_count = (
                    self.IGNORE_OPENING_AFTER_TURN
                )

                self.opening_count = 0

                self.side_space_cooldown = (
                    self.SIDE_SPACE_COOLDOWN_LIMIT
                )

                self.set_state("FORWARD")

        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)

    node = WallAvoider()

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