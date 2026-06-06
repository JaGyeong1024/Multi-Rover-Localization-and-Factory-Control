import os

from host_mission.graph_map import GraphMap

GRAPH = {
    'nodes': [
        {'id': 0, 'x': 1.8, 'y': 4.3},
        {'id': 1, 'x': 2.6, 'y': 4.3},
        {'id': 8, 'x': 2.6, 'y': 3.7},
    ],
    'edges': [
        {'from': 0, 'to': 1, 'directed': True},
        {'from': 1, 'to': 8, 'directed': False},
    ],
}
PED = os.path.expanduser('~/capstone/tools/path_editor')


def test_nearest_node_projects_part_to_lane():
    g = GraphMap(GRAPH, path_editor_dir=PED)
    assert g.nearest_node(2.6, 4.75) == 1   # part above up-lane node 1


def test_route_follows_directed_and_undirected():
    g = GraphMap(GRAPH, path_editor_dir=PED)
    assert g.route(0, 1) == [0, 1]
    assert g.route(0, 8) == [0, 1, 8]       # via cross-edge 1<->8
    assert g.xy(1) == (2.6, 4.3)
