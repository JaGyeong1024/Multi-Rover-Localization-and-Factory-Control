"""Host world model (ROS-free): latest pose per robot + arrival detection against
the lane graph. Arrival is position-based (host ground truth); the node may also use
NavFeedback ARRIVED as a fast path."""
import math


class WorldModel:
    def __init__(self, graph, position_tol):
        self.graph = graph
        self.tol = float(position_tol)
        self.pose = {}   # robot -> (x, y, yaw)

    def update_pose(self, robot, x, y, yaw):
        self.pose[robot] = (float(x), float(y), float(yaw))

    def nearest_node(self, robot):
        if robot not in self.pose:
            return -1
        x, y, _ = self.pose[robot]
        return self.graph.nearest_node(x, y)

    def arrived(self, robot, target_node):
        if robot not in self.pose or target_node is None or int(target_node) < 0:
            return False
        x, y, _ = self.pose[robot]
        tx, ty = self.graph.xy(target_node)
        return math.hypot(x - tx, y - ty) <= self.tol
