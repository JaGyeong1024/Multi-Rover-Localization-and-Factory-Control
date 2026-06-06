#!/usr/bin/env python3
"""anchor_publisher_node — visualize UWB anchor positions in RViz.

Publishes a MarkerArray on /uwb/anchors at 1 Hz. Each anchor becomes a cube
plus a text label "anchor_N" (N = 1-based index in `anchor_ids`, so the
yaml order must match the physical anchor numbering). Positions are read
from the parameters `anchor_ids` (list[str]) and `anchor_positions` (flat
list[float], 3 values per anchor in x,y,z order). The short ID is logged
on startup for cross-reference.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


class AnchorPublisherNode(Node):
    def __init__(self):
        super().__init__('anchor_publisher_node')

        self.declare_parameter('frame_id', 'uwb_global')
        self.declare_parameter('anchor_ids', [''])
        self.declare_parameter('anchor_positions', [0.0])
        self.declare_parameter('cube_size', 0.08)
        self.declare_parameter('label_height', 0.20)
        # ROI = axis-aligned anchor bbox shrunk inward by this many meters
        # on each side. Set to 0 to disable. Drawn at z = roi_z (floor plane).
        self.declare_parameter('roi_buffer', 0.5)
        self.declare_parameter('roi_z', 0.0)
        # Waypoint grid: rows x cols points uniformly spaced inside the ROI.
        # Numbered in boustrophedon (ㄹ-shape) order starting at row 0 col 0.
        # Set rows or cols < 2 to disable.
        self.declare_parameter('waypoint_rows', 6)
        self.declare_parameter('waypoint_cols', 6)
        self.declare_parameter('waypoint_z', 0.0)

        self.frame_id = self.get_parameter('frame_id').value
        ids = list(self.get_parameter('anchor_ids').value)
        flat = list(self.get_parameter('anchor_positions').value)
        self.cube_size = float(self.get_parameter('cube_size').value)
        self.label_height = float(self.get_parameter('label_height').value)
        self.roi_buffer = float(self.get_parameter('roi_buffer').value)
        self.roi_z = float(self.get_parameter('roi_z').value)
        self.wp_rows = int(self.get_parameter('waypoint_rows').value)
        self.wp_cols = int(self.get_parameter('waypoint_cols').value)
        self.wp_z = float(self.get_parameter('waypoint_z').value)

        if len(flat) != 3 * len(ids):
            self.get_logger().error(
                f'anchor_positions length ({len(flat)}) must equal 3 * len(anchor_ids) ({3*len(ids)})')
            raise ValueError('anchor_positions/anchor_ids size mismatch')

        self.anchors = [
            (ids[i], flat[3*i], flat[3*i+1], flat[3*i+2])
            for i in range(len(ids))
            if ids[i]
        ]
        for n, (aid, x, y, z) in enumerate(self.anchors, start=1):
            self.get_logger().debug(
                f'anchor_{n}: id={aid} pos=({x:.3f}, {y:.3f}, {z:.3f})')

        self.roi_bbox = None
        if self.roi_buffer > 0 and len(self.anchors) >= 2:
            xs = [a[1] for a in self.anchors]
            ys = [a[2] for a in self.anchors]
            x_min = min(xs) + self.roi_buffer
            x_max = max(xs) - self.roi_buffer
            y_min = min(ys) + self.roi_buffer
            y_max = max(ys) - self.roi_buffer
            if x_min < x_max and y_min < y_max:
                self.roi_bbox = (x_min, x_max, y_min, y_max)
                self.get_logger().info(
                    f'ROI ({self.roi_buffer:.2f} m buffer): '
                    f'x=[{x_min:.3f}, {x_max:.3f}] y=[{y_min:.3f}, {y_max:.3f}]')
            else:
                self.get_logger().warn(
                    f'roi_buffer {self.roi_buffer} m collapses the anchor bbox; '
                    'ROI not drawn.')

        # Waypoint grid (boustrophedon numbering, sorted by visit order).
        self.waypoints = []  # list of (n, x, y)
        if self.wp_rows >= 2 and self.wp_cols >= 2 and self.roi_bbox is not None:
            x_min, x_max, y_min, y_max = self.roi_bbox
            dx = (x_max - x_min) / (self.wp_cols - 1)
            dy = (y_max - y_min) / (self.wp_rows - 1)
            pts = []
            for r in range(self.wp_rows):
                y = y_min + r * dy
                for c in range(self.wp_cols):
                    x = x_min + c * dx
                    if r % 2 == 0:
                        n = r * self.wp_cols + (c + 1)
                    else:
                        n = r * self.wp_cols + (self.wp_cols - c)
                    pts.append((n, x, y))
            self.waypoints = sorted(pts, key=lambda w: w[0])
            self.get_logger().debug(
                f'Waypoint grid {self.wp_rows}x{self.wp_cols}={len(self.waypoints)} '
                f'points, dx={dx:.3f} m, dy={dy:.3f} m.')

        self.pub = self.create_publisher(MarkerArray, '/uwb/anchors', 10)
        self.timer = self.create_timer(1.0, self._publish)

    def _publish(self):
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        for idx, (aid, x, y, z) in enumerate(self.anchors):
            cube = Marker()
            cube.header.stamp = now
            cube.header.frame_id = self.frame_id
            cube.ns = 'uwb_anchor_cube'
            cube.id = idx
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position.x = x
            cube.pose.position.y = y
            cube.pose.position.z = z
            cube.pose.orientation.w = 1.0
            cube.scale.x = self.cube_size
            cube.scale.y = self.cube_size
            cube.scale.z = self.cube_size
            cube.color.r = 0.2
            cube.color.g = 0.5
            cube.color.b = 1.0
            cube.color.a = 1.0
            arr.markers.append(cube)

            label = Marker()
            label.header.stamp = now
            label.header.frame_id = self.frame_id
            label.ns = 'uwb_anchor_label'
            label.id = idx
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = x
            label.pose.position.y = y
            label.pose.position.z = z + self.label_height
            label.pose.orientation.w = 1.0
            label.scale.z = 0.12
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 1.0
            label.text = f'anchor_{idx + 1}'
            arr.markers.append(label)

        if self.roi_bbox is not None:
            x_min, x_max, y_min, y_max = self.roi_bbox
            roi = Marker()
            roi.header.stamp = now
            roi.header.frame_id = self.frame_id
            roi.ns = 'uwb_roi'
            roi.id = 0
            roi.type = Marker.LINE_STRIP
            roi.action = Marker.ADD
            roi.pose.orientation.w = 1.0
            roi.scale.x = 0.02  # line width (m)
            roi.color.r = 1.0
            roi.color.g = 0.85
            roi.color.b = 0.0
            roi.color.a = 0.9
            corners = [
                (x_min, y_min), (x_max, y_min),
                (x_max, y_max), (x_min, y_max),
                (x_min, y_min),  # close the loop
            ]
            for px, py in corners:
                p = Point()
                p.x = px
                p.y = py
                p.z = self.roi_z
                roi.points.append(p)
            arr.markers.append(roi)

        if self.waypoints:
            # Path line connecting waypoints in numeric (boustrophedon) order.
            path = Marker()
            path.header.stamp = now
            path.header.frame_id = self.frame_id
            path.ns = 'uwb_waypoint_path'
            path.id = 0
            path.type = Marker.LINE_STRIP
            path.action = Marker.ADD
            path.pose.orientation.w = 1.0
            path.scale.x = 0.01
            path.color.r = 0.4
            path.color.g = 0.8
            path.color.b = 1.0
            path.color.a = 0.6
            for n, x, y in self.waypoints:
                p = Point()
                p.x = x
                p.y = y
                p.z = self.wp_z
                path.points.append(p)
            arr.markers.append(path)

            # Per-waypoint sphere + number label.
            for n, x, y in self.waypoints:
                sphere = Marker()
                sphere.header.stamp = now
                sphere.header.frame_id = self.frame_id
                sphere.ns = 'uwb_waypoint_dot'
                sphere.id = n
                sphere.type = Marker.SPHERE
                sphere.action = Marker.ADD
                sphere.pose.position.x = x
                sphere.pose.position.y = y
                sphere.pose.position.z = self.wp_z
                sphere.pose.orientation.w = 1.0
                sphere.scale.x = 0.06
                sphere.scale.y = 0.06
                sphere.scale.z = 0.06
                sphere.color.r = 0.4
                sphere.color.g = 0.8
                sphere.color.b = 1.0
                sphere.color.a = 1.0
                arr.markers.append(sphere)

                wp_label = Marker()
                wp_label.header.stamp = now
                wp_label.header.frame_id = self.frame_id
                wp_label.ns = 'uwb_waypoint_label'
                wp_label.id = n
                wp_label.type = Marker.TEXT_VIEW_FACING
                wp_label.action = Marker.ADD
                wp_label.pose.position.x = x
                wp_label.pose.position.y = y
                wp_label.pose.position.z = self.wp_z + 0.10
                wp_label.pose.orientation.w = 1.0
                wp_label.scale.z = 0.10
                wp_label.color.r = 1.0
                wp_label.color.g = 1.0
                wp_label.color.b = 1.0
                wp_label.color.a = 1.0
                wp_label.text = str(n)
                arr.markers.append(wp_label)

        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = AnchorPublisherNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
