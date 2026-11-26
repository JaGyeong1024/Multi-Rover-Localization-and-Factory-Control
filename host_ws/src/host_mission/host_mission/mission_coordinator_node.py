#!/usr/bin/env python3
"""mission_coordinator_node: entry point of the host mission coordinator.

Runs path_coordinator_node (named goal poses, host-planned paths sent as PathCommand on
/{robot}/path_cmd) under the node name ``mission_coordinator_node``. The ``nav_mode``
parameter is kept and must be ``path``."""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

NODE_NAME = 'mission_coordinator_node'
NAV_MODES = ('path',)


def read_nav_mode():
    """Read ``nav_mode`` through a short-lived node of the same name, so launch
    parameter overrides addressed to the coordinator apply."""
    probe = Node(NODE_NAME)
    mode = str(probe.declare_parameter('nav_mode', 'path').value).strip().lower()
    probe.destroy_node()
    if mode not in NAV_MODES:
        raise ValueError(f'nav_mode must be one of {NAV_MODES}, got {mode!r}')
    return mode


def make_node(mode):
    """Build the coordinator for ``mode`` (only ``path`` exists)."""
    from host_mission.path_coordinator_node import PathCoordinatorNode
    return PathCoordinatorNode()


def main():
    rclpy.init()
    node = make_node(read_nav_mode())
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
