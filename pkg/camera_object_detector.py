import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Bool

from cv_bridge import CvBridge
import cv2
import numpy as np
import json


class CameraObjectDetector(Node):
    def __init__(self):
        super().__init__('camera_object_detector')

        self.enabled_sub = self.create_subscription(
            Bool,
            '/camera_detector_enabled',
            self.enabled_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/pi_camera/image_raw',
            self.image_callback,
            10
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/pi_camera/camera_info',
            self.camera_info_callback,
            10
        )

        self.pub = self.create_publisher(
            String,
            '/selected_red_object',
            10
        )

        self.bridge = CvBridge()

        self.enabled = False

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Project object dimensions: 2 cm x 3 cm x 20 cm.
        self.OBJECT_WIDTH = 0.03
        self.OBJECT_HEIGHT = 0.20

        self.MIN_AREA = 100
        self.CENTER_WEIGHT = 0.65
        self.SIZE_WEIGHT = 0.35

        self.last_print_time = 0.0
        self.PRINT_INTERVAL = 0.5

        self.get_logger().info('Camera red-object detector started ✓')
        self.get_logger().info('Detector is OFF until /camera_detector_enabled is true')

    def enabled_callback(self, msg):
        if msg.data != self.enabled:
            self.enabled = msg.data

            if self.enabled:
                self.get_logger().warn('Camera detector ENABLED')
            else:
                self.get_logger().warn('Camera detector DISABLED')

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def estimate_position(self, u, v, pixel_w, pixel_h):
        if pixel_h <= 0:
            return None

        # Use object height because object is upright.
        z = self.OBJECT_HEIGHT * self.fy / float(pixel_h)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        return x, y, z

    def choose_one_object(self, detections, image_width):
        if len(detections) == 0:
            return None

        max_area = max(d['area'] for d in detections)

        best = None
        best_score = -999.0

        for d in detections:
            center_error = abs(d['u'] - image_width / 2.0) / (image_width / 2.0)
            center_score = 1.0 - center_error
            size_score = d['area'] / max_area

            score = self.CENTER_WEIGHT * center_score + self.SIZE_WEIGHT * size_score
            d['score'] = score

            if score > best_score:
                best_score = score
                best = d

        return best

    def image_callback(self, msg):
        if not self.enabled:
            return

        if self.fx is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        height, width, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red_1 = np.array([0, 80, 50])
        upper_red_1 = np.array([10, 255, 255])

        lower_red_2 = np.array([170, 80, 50])
        upper_red_2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if w <= 0 or h <= 0:
                continue

            u = x + w / 2.0
            v = y + h / 2.0

            pos = self.estimate_position(u, v, w, h)

            if pos is None:
                continue

            x_cam, y_cam, z_cam = pos
            offset = (u - width / 2.0) / (width / 2.0)

            if offset < -0.15:
                direction = 'LEFT'
            elif offset > 0.15:
                direction = 'RIGHT'
            else:
                direction = 'CENTER'

            detections.append({
                'u': u,
                'v': v,
                'bbox_x': x,
                'bbox_y': y,
                'bbox_w': w,
                'bbox_h': h,
                'area': area,
                'x_cam': x_cam,
                'y_cam': y_cam,
                'z_cam': z_cam,
                'offset': offset,
                'direction': direction
            })

        selected = self.choose_one_object(detections, width)

        if selected is None:
            return

        out = {
            'found': True,
            'u': round(selected['u'], 2),
            'v': round(selected['v'], 2),
            'bbox_w': int(selected['bbox_w']),
            'bbox_h': int(selected['bbox_h']),
            'area': round(selected['area'], 2),
            'x_cam': round(selected['x_cam'], 3),
            'y_cam': round(selected['y_cam'], 3),
            'z_cam': round(selected['z_cam'], 3),
            'offset': round(selected['offset'], 3),
            'direction': selected['direction'],
            'score': round(selected['score'], 3),
            'num_red_objects_seen': len(detections)
        }

        msg_out = String()
        msg_out.data = json.dumps(out)
        self.pub.publish(msg_out)

        now = self.get_clock().now().nanoseconds / 1e9

        if now - self.last_print_time >= self.PRINT_INTERVAL:
            self.get_logger().info(
                f"Selected red object: {selected['direction']} | "
                f"seen={len(detections)} | "
                f"u={selected['u']:.1f} | "
                f"bbox=({selected['bbox_w']}x{selected['bbox_h']}) | "
                f"X={selected['x_cam']:.3f}m | "
                f"Z={selected['z_cam']:.3f}m | "
                f"score={selected['score']:.2f}"
            )

            self.last_print_time = now


def main(args=None):
    rclpy.init(args=args)

    node = CameraObjectDetector()

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