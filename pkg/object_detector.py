import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math


class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.CELL_SIZE = 0.70
        self.MAZE_CELLS = 7

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_cell = None

        self.object_found = False
        self.detect_count = 0
        self.DETECT_LIMIT = 5

        # Only detect objects reasonably close.
        # 0.55 is allowed because object may be in adjacent cell.
        self.OBJECT_MIN_DIST = 0.10
        self.OBJECT_MAX_DIST = 0.65

        # Search wider area so object can be slightly left/right.
        self.SEARCH_ARC_DEG = 80

        # Object is narrow. Walls/corners are often wider or unstable.
        self.MIN_CLUSTER_POINTS = 2
        self.MAX_CLUSTER_POINTS = 10

        self.MIN_OBJECT_WIDTH = 0.005
        self.MAX_OBJECT_WIDTH = 0.08

        # Reject objects estimated outside maze.
        self.last_detected_cell = None

        self.get_logger().info("Object detector started ✓")
        self.get_logger().info("LiDAR object cell estimation enabled")

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_yaw = self.get_yaw_from_odom(msg)
        self.robot_cell = self.world_to_cell(self.robot_x, self.robot_y)

    def get_yaw_from_odom(self, msg):
        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        return math.atan2(siny_cosp, cosy_cosp)

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

    def distance_to_cell_center(self, x, y, cell):
        row, col = cell
        cx, cy = self.cell_to_world(row, col)

        dx = x - cx
        dy = y - cy

        return math.sqrt(dx * dx + dy * dy)

    def estimate_object_world_position(self, distance, lidar_angle):
        """
        LiDAR angle is relative to robot heading.

        Local robot frame:
        x forward
        y left

        Convert detected LiDAR point to world coordinates.
        """

        local_x = distance * math.cos(lidar_angle)
        local_y = distance * math.sin(lidar_angle)

        world_x = (
            self.robot_x +
            local_x * math.cos(self.robot_yaw) -
            local_y * math.sin(self.robot_yaw)
        )

        world_y = (
            self.robot_y +
            local_x * math.sin(self.robot_yaw) +
            local_y * math.cos(self.robot_yaw)
        )

        return world_x, world_y

    def get_lidar_points(self, msg):
        points = []
        angle = msg.angle_min

        for i, r in enumerate(msg.ranges):
            deg = math.degrees(angle)

            if deg > 180:
                deg -= 360
            elif deg < -180:
                deg += 360

            in_search_area = abs(deg) <= self.SEARCH_ARC_DEG

            valid_distance = (
                not math.isnan(r) and
                not math.isinf(r) and
                self.OBJECT_MIN_DIST <= r <= self.OBJECT_MAX_DIST
            )

            if in_search_area and valid_distance:
                points.append((i, angle, deg, r))

            angle += msg.angle_increment

        return points

    def find_clusters(self, points):
        if not points:
            return []

        clusters = []
        cluster = [points[0]]

        for i in range(1, len(points)):
            prev = points[i - 1]
            curr = points[i]

            index_gap = curr[0] - prev[0]
            dist_gap = abs(curr[3] - prev[3])

            if index_gap <= 2 and dist_gap <= 0.05:
                cluster.append(curr)
            else:
                clusters.append(cluster)
                cluster = [curr]

        clusters.append(cluster)
        return clusters

    def cluster_width(self, cluster):
        if len(cluster) < 2:
            return 0.0

        first_angle = cluster[0][1]
        last_angle = cluster[-1][1]

        angular_width = abs(last_angle - first_angle)
        avg_dist = self.cluster_avg_distance(cluster)

        return avg_dist * angular_width

    def cluster_avg_distance(self, cluster):
        return sum(p[3] for p in cluster) / len(cluster)

    def cluster_center_angle(self, cluster):
        mid = len(cluster) // 2
        return cluster[mid][1]

    def cluster_center_deg(self, cluster):
        mid = len(cluster) // 2
        return cluster[mid][2]

    def is_reasonable_object_cell(self, object_cell):
        if object_cell is None:
            return False

        row, col = object_cell

        if row < 1 or row > 7 or col < 1 or col > 7:
            return False

        return True

    def scan_callback(self, msg):
        if self.object_found:
            return

        if self.robot_cell is None:
            return

        points = self.get_lidar_points(msg)
        clusters = self.find_clusters(points)

        detected = False
        best_cell = None
        best_distance = None
        best_width = None
        best_angle_deg = None
        best_center_error = None
        best_world_x = None
        best_world_y = None

        for cluster in clusters:
            point_count = len(cluster)

            if point_count < self.MIN_CLUSTER_POINTS:
                continue

            if point_count > self.MAX_CLUSTER_POINTS:
                continue

            width = self.cluster_width(cluster)

            if not (self.MIN_OBJECT_WIDTH <= width <= self.MAX_OBJECT_WIDTH):
                continue

            avg_dist = self.cluster_avg_distance(cluster)
            center_angle = self.cluster_center_angle(cluster)
            center_deg = abs(self.cluster_center_deg(cluster))

            object_x, object_y = self.estimate_object_world_position(
                avg_dist,
                center_angle
            )

            object_cell = self.world_to_cell(object_x, object_y)

            if not self.is_reasonable_object_cell(object_cell):
                continue

            center_error = self.distance_to_cell_center(
                object_x,
                object_y,
                object_cell
            )

            # Object should be somewhere inside a cell, not very far from a cell center.
            # This helps reject wall edges/corners.
            if center_error > 0.28:
                continue

            detected = True
            best_cell = object_cell
            best_distance = avg_dist
            best_width = width
            best_angle_deg = center_deg
            best_center_error = center_error
            best_world_x = object_x
            best_world_y = object_y
            break

        if detected:
            if best_cell == self.last_detected_cell:
                self.detect_count += 1
            else:
                self.last_detected_cell = best_cell
                self.detect_count = 1
        else:
            self.detect_count = 0
            self.last_detected_cell = None

        if self.detect_count >= self.DETECT_LIMIT:
            self.object_found = True

            self.get_logger().warn(
                f"Object detected! Found {best_cell} | "
                f"robot_cell={self.robot_cell} | "
                f"distance={best_distance:.2f} m | "
                f"width={best_width:.3f} m | "
                f"angle={best_angle_deg:.1f} deg | "
                f"object_xy=({best_world_x:.2f}, {best_world_y:.2f}) | "
                f"cell_error={best_center_error:.2f} m"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()