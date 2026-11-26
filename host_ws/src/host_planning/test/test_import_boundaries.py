"""host_planning must be pure Python (numpy + stdlib): no ROS dependencies."""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1] / 'host_planning'
FORBIDDEN = ('rclpy', 'tf2', 'tf2_ros', 'ament_index_python', 'host_mission',
             'host_mission_interfaces', 'host_sim', 'rover_drive')


def _imports(path):
    """(line, top-level module), including lazy imports inside functions."""
    for n in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield n.lineno, a.name.split('.')[0]
        elif isinstance(n, ast.ImportFrom) and n.level == 0:
            yield n.lineno, n.module.split('.')[0]


def _is_forbidden(mod):
    return mod in FORBIDDEN or mod.endswith('_msgs')


def test_host_planning_has_no_ros_dependency():
    bad = [f'{p.relative_to(ROOT.parent)}:{ln} imports {m}'
           for p in sorted(ROOT.rglob('*.py')) for ln, m in _imports(p)
           if _is_forbidden(m)]
    assert not bad, 'host_planning 경계 위반:\n' + '\n'.join(bad)
