"""host_mission boundary: pure modules (not *_node.py) must import without rclpy."""
import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parents[1] / 'host_mission'
NO_ROS = ('rclpy', 'tf2', 'tf2_ros', 'host_mission_interfaces')
NO_ROVER = ('rover_drive', 'host_sim')
MSGS_ALLOWED = ()  # pure modules allowed to use *_msgs (currently none)


def _imports(path):
    """(line, top-level module), including lazy imports inside functions."""
    for n in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield n.lineno, a.name.split('.')[0]
        elif isinstance(n, ast.ImportFrom) and n.level == 0:
            yield n.lineno, n.module.split('.')[0]


def _pure_modules():
    return [p for p in sorted(PKG.rglob('*.py')) if not p.name.endswith('_node.py')]


def test_pure_modules_do_not_import_ros():
    bad = []
    for p in _pure_modules():
        for ln, m in _imports(p):
            msgs = m.endswith('_msgs') and p.name not in MSGS_ALLOWED
            if m in NO_ROS or msgs:
                bad.append(f'{p.relative_to(PKG.parent)}:{ln} imports {m}')
    assert not bad, '순수 모듈 경계 위반:\n' + '\n'.join(bad)


def test_no_rover_or_sim_imports():
    bad = [f'{p.relative_to(PKG.parent)}:{ln} imports {m}'
           for p in sorted(PKG.rglob('*.py')) for ln, m in _imports(p) if m in NO_ROVER]
    assert not bad, 'host_mission이 로버/시뮬 코드를 import:\n' + '\n'.join(bad)
