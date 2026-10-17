import math
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class IKPickTest(Node):
    def __init__(self):
        super().__init__('ik_pick_test')

        self.arm_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        self.declare_parameter('x', 0.22)
        self.declare_parameter('y', 0.00)
        self.declare_parameter('z', 0.05)

        self.x = float(self.get_parameter('x').value)
        self.y = float(self.get_parameter('y').value)
        self.z = float(self.get_parameter('z').value)

        # OpenManipulator-X link lengths from Lab 8 DH table.
        self.L1 = 0.077
        self.L2 = 0.130
        self.L3 = 0.124
        self.L4 = 0.126

        self.timer = self.create_timer(1.0, self.run_once)
        self.sent = False

        self.get_logger().info('IK pick test started ✓')
        self.get_logger().info(
            f'Target in arm frame: x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}'
        )

    def calculate_ik(self, x, y, z):
        # joint1 rotates the arm left/right.
        joint1 = math.atan2(y, x)

        # Planar distance from arm base to target.
        r_total = math.sqrt(x * x + y * y)

        # Remove gripper/tool length from the reach.
        r = r_total - self.L4

        # Height relative to shoulder/base vertical offset.
        z_eff = z - self.L1

        if r <= 0:
            raise ValueError(
                f'Target too close after removing gripper length. r={r:.3f}'
            )

        # Cosine law for elbow.
        D = (
            r * r + z_eff * z_eff - self.L2 * self.L2 - self.L3 * self.L3
        ) / (2.0 * self.L2 * self.L3)

        if D > 1.0 or D < -1.0:
            raise ValueError(
                f'Target unreachable. D={D:.3f}. Try different x/y/z.'
            )

        # Elbow-down solution.
        joint3 = math.atan2(-math.sqrt(1.0 - D * D), D)

        # Shoulder angle.
        joint2 = math.atan2(z_eff, r) - math.atan2(
            self.L3 * math.sin(joint3),
            self.L2 + self.L3 * math.cos(joint3)
        )

        # Keep gripper roughly level.
        joint4 = -(joint2 + joint3)

        return [joint1, joint2, joint3, joint4]

    def forward_kinematics_simple(self, joints):
        joint1, joint2, joint3, joint4 = joints

        # This is simplified FK for checking position only.
        # It matches the geometric IK model:
        # L2 and L3 form the main planar arm, L4 is the gripper/tool extension.
        theta23 = joint2 + joint3
        theta234 = joint2 + joint3 + joint4

        r = (
            self.L2 * math.cos(joint2)
            + self.L3 * math.cos(theta23)
            + self.L4 * math.cos(theta234)
        )

        z = (
            self.L1
            + self.L2 * math.sin(joint2)
            + self.L3 * math.sin(theta23)
            + self.L4 * math.sin(theta234)
        )

        x = r * math.cos(joint1)
        y = r * math.sin(joint1)

        return x, y, z

    def send_arm_pose(self, joints):
        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']

        point = JointTrajectoryPoint()
        point.positions = joints
        point.time_from_start.sec = 3
        point.time_from_start.nanosec = 0

        msg.points.append(point)
        self.arm_pub.publish(msg)

    def run_once(self):
        if self.sent:
            return

        try:
            joints = self.calculate_ik(self.x, self.y, self.z)
        except ValueError as e:
            self.get_logger().error(str(e))
            self.sent = True
            return

        fk_x, fk_y, fk_z = self.forward_kinematics_simple(joints)

        self.get_logger().info(
            'IK joints rad: '
            f'[{joints[0]:.3f}, {joints[1]:.3f}, {joints[2]:.3f}, {joints[3]:.3f}]'
        )

        self.get_logger().info(
            'IK joints deg: '
            f'[{math.degrees(joints[0]):.1f}, '
            f'{math.degrees(joints[1]):.1f}, '
            f'{math.degrees(joints[2]):.1f}, '
            f'{math.degrees(joints[3]):.1f}]'
        )

        self.get_logger().info(
            f'FK check: x={fk_x:.3f}, y={fk_y:.3f}, z={fk_z:.3f}'
        )

        self.send_arm_pose(joints)
        self.get_logger().info('Sent IK pose to arm_controller')
        self.sent = True


def main(args=None):
    rclpy.init(args=args)

    node = IKPickTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()