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
        self._declare_parameters()
        self._read_parameters()
        self._load_anchors()
        self._build_roi()
        self._build_waypoints()

        self.pub = self.create_publisher(MarkerArray, '/uwb/anchors', 10)
        self.timer = self.create_timer(1.0, self._publish)

    def _declare_parameters(self):
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
        # Numbered in boustrophedon order starting at row 0 col 0.
        # Set rows or cols < 2 to disable.
        self.declare_parameter('waypoint_rows', 6)
        self.declare_parameter('waypoint_cols', 6)
        self.declare_parameter('waypoint_z', 0.0)

    def _read_parameters(self):
        self.frame_id = self.get_parameter('frame_id').value
        self.cube_size = float(self.get_parameter('cube_size').value)
        self.label_height = float(self.get_parameter('label_height').value)
        self.roi_buffer = float(self.get_parameter('roi_buffer').value)
        self.roi_z = float(self.get_parameter('roi_z').value)
        self.wp_rows = int(self.get_parameter('waypoint_rows').value)
        self.wp_cols = int(self.get_parameter('waypoint_cols').value)
        self.wp_z = float(self.get_parameter('waypoint_z').value)

    def _load_anchors(self):
        ids = list(self.get_parameter('anchor_ids').value)
        flat = list(self.get_parameter('anchor_positions').value)
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

    def _build_roi(self):
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

    def _build_waypoints(self):
        """Waypoint grid (boustrophedon numbering, sorted by visit order)."""
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

    def _marker(self, now, ns, marker_id, marker_type, color, scale=(0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0), text=None):
        """Build a Marker in the node frame with identity orientation."""
        m = Marker()
        m.header.stamp = now
        m.header.frame_id = self.frame_id
        m.ns = ns
        m.id = marker_id
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = position
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = color
        if text is not None:
            m.text = text
        return m

    @staticmethod
    def _points(xy_pairs, z):
        pts = []
        for px, py in xy_pairs:
            p = Point()
            p.x = px
            p.y = py
            p.z = z
            pts.append(p)
        return pts

    def _anchor_markers(self, now):
        markers = []
        s = self.cube_size
        for idx, (aid, x, y, z) in enumerate(self.anchors):
            markers.append(self._marker(
                now, 'uwb_anchor_cube', idx, Marker.CUBE, (0.2, 0.5, 1.0, 1.0),
                scale=(s, s, s), position=(x, y, z)))
            markers.append(self._marker(
                now, 'uwb_anchor_label', idx, Marker.TEXT_VIEW_FACING,
                (1.0, 1.0, 1.0, 1.0), scale=(0.0, 0.0, 0.12),
                position=(x, y, z + self.label_height), text=f'anchor_{idx + 1}'))
        return markers

    def _roi_marker(self, now):
        x_min, x_max, y_min, y_max = self.roi_bbox
        roi = self._marker(now, 'uwb_roi', 0, Marker.LINE_STRIP, (1.0, 0.85, 0.0, 0.9),
                           scale=(0.02, 0.0, 0.0))  # line width (m)
        corners = [
            (x_min, y_min), (x_max, y_min),
            (x_max, y_max), (x_min, y_max),
            (x_min, y_min),  # close the loop
        ]
        roi.points.extend(self._points(corners, self.roi_z))
        return roi

    def _waypoint_markers(self, now):
        # Path line connecting waypoints in numeric (boustrophedon) order.
        path = self._marker(now, 'uwb_waypoint_path', 0, Marker.LINE_STRIP,
                            (0.4, 0.8, 1.0, 0.6), scale=(0.01, 0.0, 0.0))
        path.points.extend(self._points([(x, y) for _, x, y in self.waypoints], self.wp_z))
        markers = [path]

        # Per-waypoint sphere + number label.
        for n, x, y in self.waypoints:
            markers.append(self._marker(
                now, 'uwb_waypoint_dot', n, Marker.SPHERE, (0.4, 0.8, 1.0, 1.0),
                scale=(0.06, 0.06, 0.06), position=(x, y, self.wp_z)))
            markers.append(self._marker(
                now, 'uwb_waypoint_label', n, Marker.TEXT_VIEW_FACING,
                (1.0, 1.0, 1.0, 1.0), scale=(0.0, 0.0, 0.10),
                position=(x, y, self.wp_z + 0.10), text=str(n)))
        return markers

    def _publish(self):
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        arr.markers.extend(self._anchor_markers(now))
        if self.roi_bbox is not None:
            arr.markers.append(self._roi_marker(now))
        if self.waypoints:
            arr.markers.extend(self._waypoint_markers(now))

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
