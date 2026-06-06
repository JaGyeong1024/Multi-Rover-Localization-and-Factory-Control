#!/usr/bin/env python3
"""Publish memoryless corrected UWB poses for one or more rovers.

robots 리스트(기본 leader·follower)마다 `{robot}.*` 파라미터로 입력/출력 토픽·보정계수를 받아
`/uwb/{robot}_pose_corrected` 발행. 보정은 의도적 memoryless(시간평균 없음 — raw가 source of record).
⚠ follower 계수는 현재 leader affine 재사용(임시 A) — leader 높이(0.185) 전용 fit이라 follower(0.09)
태그 성분 미보정. 둘 다 웨이포인트 재캘리가 본책. config/uwb_config.yaml 참조.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

from host_localization.uwb_correction import correct_xy, make_correction_params


class UWBCorrectionNode(Node):
    # RViz 구분색(r,g,b): leader=cyan, follower=orange
    COLORS = {'leader': (0.0, 0.85, 1.0), 'follower': (1.0, 0.55, 0.0)}

    def __init__(self):
        super().__init__('uwb_correction_node')
        robots = self.declare_parameter('robots', ['leader', 'follower']).value

        self.robots = []
        for name in robots:
            inp = self.declare_parameter(f'{name}.input_topic', f'/uwb/{name}_pose').value
            outp = self.declare_parameter(f'{name}.output_topic', f'/uwb/{name}_pose_corrected').value
            mk = self.declare_parameter(f'{name}.marker_topic', f'/uwb/{name}_marker_corrected').value
            params = make_correction_params(
                self.declare_parameter(f'{name}.model', 'affine').value,
                self.declare_parameter(f'{name}.offset', [0.0, 0.0]).value,
                self.declare_parameter(f'{name}.affine_x', [1.0, 0.0, 0.0]).value,
                self.declare_parameter(f'{name}.affine_y', [0.0, 1.0, 0.0]).value,
            )
            z_mode = str(self.declare_parameter(f'{name}.output_z_mode', 'fixed').value).lower()
            if z_mode not in ('fixed', 'raw'):
                raise ValueError(f'{name}.output_z_mode must be "fixed" or "raw"')
            st = {
                'name': name,
                'params': params,
                'z_mode': z_mode,
                'fixed_z': float(self.declare_parameter(f'{name}.fixed_z', 0.146).value),
                'mscale': float(self.declare_parameter(f'{name}.marker_scale', 0.10).value),
                'color': self.COLORS.get(name, (0.0, 0.85, 1.0)),
                'pub': self.create_publisher(PoseStamped, outp, 10),
                'mpub': self.create_publisher(Marker, mk, 10),
            }
            self.create_subscription(PoseStamped, inp, lambda m, s=st: self._on_pose(m, s), 10)
            self.robots.append(st)
            self.get_logger().info(
                f'UWB correction[{name}] model={params.model} {inp} -> {outp}, '
                f'marker={mk}, z_mode={z_mode}')

    def _on_pose(self, msg: PoseStamped, st):
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        cx, cy = correct_xy(x, y, st['params'])

        out = PoseStamped()
        out.header = msg.header
        out.pose = msg.pose
        out.pose.position.x = cx
        out.pose.position.y = cy
        if st['z_mode'] == 'fixed':
            out.pose.position.z = st['fixed_z']
        st['pub'].publish(out)

        marker = Marker()
        marker.header = out.header
        marker.ns = f"uwb_{st['name']}_corrected"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = out.pose
        marker.scale.x = st['mscale']
        marker.scale.y = st['mscale']
        marker.scale.z = st['mscale']
        marker.color.r, marker.color.g, marker.color.b = st['color']
        marker.color.a = 1.0
        st['mpub'].publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = UWBCorrectionNode()
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
