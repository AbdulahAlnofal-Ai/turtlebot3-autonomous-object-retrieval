import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math


class MissionManager(Node):
    def __init__(self):
        super().__init__('mission_manager')

        # ======================================================
        # PARAMETERS
        # ======================================================
        self.declare_parameter('cell_size', 0.70)

        self.declare_parameter('start_row', 1)
        self.declare_parameter('start_col', 3)

        self.declare_parameter('target_row', 1)
        self.declare_parameter('target_col', 7)

        self.cell_size = float(
            self.get_parameter('cell_size').value
        )

        self.start_cell = (
            int(self.get_parameter('start_row').value),
            int(self.get_parameter('start_col').value)
        )

        self.target_cell = (
            int(self.get_parameter('target_row').value),
            int(self.get_parameter('target_col').value)
        )

        # ======================================================
        # CELL CONFIRMATION SETTINGS
        # ======================================================
        # Confirm cell only when robot is close to cell center.
        self.CENTER_TOLERANCE = 0.12

        # Warning only for unusually large center error.
        # 0.35m is normal at a cell boundary because cell size is 0.70m.
        self.DRIFT_WARNING_DIST = 0.38

        # ======================================================
        # CELL TRACKING
        # ======================================================
        self.raw_cell = None
        self.confirmed_cell = None

        self.last_confirmed_cell = None
        self.last_drift_warning_cell = None

        # ======================================================
        # MISSION MEMORY
        # ======================================================
        self.object_cell = None

        self.total_objects = 2
        self.delivered_objects = 0

        # ======================================================
        # SUBSCRIBERS
        # ======================================================
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # ======================================================
        # STARTUP LOGS
        # ======================================================
        self.get_logger().info("Mission manager started ✓")
        self.get_logger().info(f"Parking/start cell: {self.start_cell}")
        self.get_logger().info(f"Target/drop-off cell: {self.target_cell}")
        self.get_logger().info("Object cell: UNKNOWN")
        self.get_logger().info(
            f"Cell center tolerance: {self.CENTER_TOLERANCE:.2f} m"
        )

    # ==========================================================
    # CONVERT WORLD POSITION TO NEAREST CELL
    # ==========================================================
    def world_to_cell(self, x, y):
        """
        Convert Gazebo world position to nearest maze cell.

        Maze:
        7x7 cells
        cell size = 0.70 m
        center cell (4,4) = world (0,0)

        Formula:
        col = round(x / cell_size) + 4
        row = round(y / cell_size) + 4
        """

        col = round(x / self.cell_size) + 4
        row = round(y / self.cell_size) + 4

        row = max(1, min(7, row))
        col = max(1, min(7, col))

        return row, col

    # ==========================================================
    # CONVERT CELL TO WORLD CENTER
    # ==========================================================
    def cell_to_world(self, row, col):
        """
        Convert maze cell to Gazebo world center position.

        Formula:
        x = (col - 4) * cell_size
        y = (row - 4) * cell_size
        """

        x = (col - 4) * self.cell_size
        y = (row - 4) * self.cell_size

        return x, y

    # ==========================================================
    # DISTANCE FROM ROBOT TO CELL CENTER
    # ==========================================================
    def distance_to_cell_center(self, x, y, cell):
        row, col = cell

        center_x, center_y = self.cell_to_world(row, col)

        dx = x - center_x
        dy = y - center_y

        dist = math.sqrt(dx * dx + dy * dy)

        return dist, dx, dy, center_x, center_y

    # ==========================================================
    # YAW FROM ODOM
    # ==========================================================
    def get_yaw_from_odom(self, msg):
        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (
            q.w * q.z +
            q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y * q.y +
            q.z * q.z
        )

        yaw = math.atan2(siny_cosp, cosy_cosp)

        return yaw

    # ==========================================================
    # ODOM CALLBACK
    # ==========================================================
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.get_yaw_from_odom(msg)

        self.raw_cell = self.world_to_cell(x, y)

        dist, dx, dy, center_x, center_y = self.distance_to_cell_center(
            x,
            y,
            self.raw_cell
        )

        # ======================================================
        # WARN ONLY IF DRIFT IS BIGGER THAN NORMAL BOUNDARY SWITCH
        # ======================================================
        if (
            dist >= self.DRIFT_WARNING_DIST and
            self.raw_cell != self.last_drift_warning_cell
        ):
            self.last_drift_warning_cell = self.raw_cell

            self.get_logger().warn(
                f"Large drift near {self.raw_cell}: "
                f"error={dist:.2f} m | "
                f"robot=({x:.2f}, {y:.2f}) | "
                f"center=({center_x:.2f}, {center_y:.2f}) | "
                f"yaw={yaw:.2f}"
            )

        # ======================================================
        # CONFIRM CELL ONLY IF ROBOT IS CLOSE TO CENTER
        # ======================================================
        if dist <= self.CENTER_TOLERANCE:
            if self.raw_cell != self.confirmed_cell:
                self.confirmed_cell = self.raw_cell

                if self.confirmed_cell != self.last_confirmed_cell:
                    self.last_confirmed_cell = self.confirmed_cell

                    self.get_logger().info(
                        f"Confirmed cell: {self.confirmed_cell} | "
                        f"center error={dist:.2f} m | "
                        f"yaw={yaw:.2f}"
                    )

    # ==========================================================
    # OBJECT FOUND
    # ==========================================================
    def mark_object_found(self):
        """
        Later, object detection will call this.

        Important:
        Use confirmed_cell, not raw_cell.
        """

        if self.object_cell is None and self.confirmed_cell is not None:
            self.object_cell = self.confirmed_cell

            self.get_logger().warn(
                f"Found {self.object_cell}"
            )

    # ==========================================================
    # OBJECT DELIVERED
    # ==========================================================
    def mark_object_delivered(self):
        """
        Later, manipulator/drop-off logic will call this.
        """

        self.delivered_objects += 1

        self.get_logger().info(
            f"Delivered objects: "
            f"{self.delivered_objects}/{self.total_objects}"
        )

        if self.delivered_objects >= self.total_objects:
            self.get_logger().warn(
                f"All objects delivered. "
                f"Return to parking cell {self.start_cell}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = MissionManager()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()