"""Lane graph wrapper — reuses tools/path_editor/graph_route.py (Dijkstra) so the
routing algorithm has a single source of truth. host_mission uses this for TARGET
selection (part→nearest lane node) and route/overlap prediction; it does NOT emit
dense waypoints (rover owns routing). frame = uwb_global.

graph dict schema (== tools/path_editor graph.json): {nodes:[{id,x,y}], edges:[{from,to,directed}]}.
"""
import math
import os
import sys


def _import_graph_route(path_editor_dir):
    d = os.path.expanduser(path_editor_dir)
    if d not in sys.path:
        sys.path.insert(0, d)
    import graph_route  # noqa: E402  (tools/path_editor/graph_route.py)
    return graph_route


class GraphMap:
    def __init__(self, graph_dict, path_editor_dir='~/capstone/tools/path_editor'):
        self._gr = _import_graph_route(path_editor_dir)
        self.nodes = {int(n['id']): (float(n['x']), float(n['y'])) for n in graph_dict['nodes']}
        # adjacency (id -> [(nbr, weight)]); directed edges one-way, else both ways
        self.adj = {i: [] for i in self.nodes}
        for e in graph_dict['edges']:
            a, b = int(e['from']), int(e['to'])
            w = math.hypot(self.nodes[a][0] - self.nodes[b][0], self.nodes[a][1] - self.nodes[b][1])
            self.adj[a].append((b, w))
            if not e.get('directed', False):
                self.adj[b].append((a, w))

    def nearest_node(self, x, y):
        return self._gr.nearest_node(self.nodes, x, y)

    def route(self, src, dst):
        """Ordered node-id list src..dst, or None if unreachable (reuses graph_route.dijkstra)."""
        return self._gr.dijkstra(self.adj, int(src), int(dst))

    def route_cost(self, src, dst):
        """Directed shortest-path length src->dst (m), inf if unreachable. Asymmetric on
        directed lanes (cost(a,b) != cost(b,a)) — used for time-optimal pickup ordering."""
        src, dst = int(src), int(dst)
        if src == dst:
            return 0.0
        path = self.route(src, dst)
        if not path or len(path) < 2:
            return math.inf
        return sum(math.hypot(self.nodes[path[k + 1]][0] - self.nodes[path[k]][0],
                              self.nodes[path[k + 1]][1] - self.nodes[path[k]][1])
                   for k in range(len(path) - 1))

    def xy(self, node):
        return self.nodes[int(node)]
