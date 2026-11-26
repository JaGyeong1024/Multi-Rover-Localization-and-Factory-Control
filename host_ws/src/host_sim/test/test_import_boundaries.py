"""host_sim boundary: modules other than sim_node.py import without ROS and do not depend on host/rover packages."""
import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parents[1] / 'host_sim'
NO_ROS = ('rclpy', 'tf2', 'tf2_ros', 'host_mission_interfaces')
NO_OTHER = ('host_planning', 'host_mission', 'rover_drive')


def _imports(path):
    """(line, top-level module), including lazy imports inside functions."""
    for n in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield n.lineno, a.name.split('.')[0]
        elif isinstance(n, ast.ImportFrom) and n.level == 0:
            yield n.lineno, n.module.split('.')[0]


def test_non_node_modules_import_no_ros():
    bad = []
    for p in sorted(PKG.rglob('*.py')):
        for ln, m in _imports(p):
            ros = p.name != 'sim_node.py' and (m in NO_ROS or m.endswith('_msgs'))
            if ros or m in NO_OTHER:
                bad.append(f'{p.relative_to(PKG.parent)}:{ln} imports {m}')
    assert not bad, 'host_sim 경계 위반:\n' + '\n'.join(bad)
