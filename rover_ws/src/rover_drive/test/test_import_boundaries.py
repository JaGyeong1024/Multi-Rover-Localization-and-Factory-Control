"""rover_drive boundary: core/ is pure Python; rover code does not import host packages."""
import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parents[1] / 'rover_drive'
ROS_CORE = ('rclpy', 'tf2', 'tf2_ros', 'ament_index_python', 'host_mission_interfaces')
HOST_PKGS = ('host_planning', 'host_mission', 'host_sim')


def _imports(path):
    """(line, top-level module), including lazy imports inside functions."""
    for n in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield n.lineno, a.name.split('.')[0]
        elif isinstance(n, ast.ImportFrom) and n.level == 0:
            yield n.lineno, n.module.split('.')[0]


def _violations(folder, forbidden, msgs=False):
    return [f'{p.relative_to(PKG.parent)}:{ln} imports {m}'
            for p in sorted(folder.rglob('*.py')) for ln, m in _imports(p)
            if m in forbidden or m in HOST_PKGS or (msgs and m.endswith('_msgs'))]


def test_core_is_pure_python():
    bad = _violations(PKG / 'core', ROS_CORE, msgs=True)
    assert not bad, 'rover_drive/core 경계 위반:\n' + '\n'.join(bad)


def test_rover_code_imports_no_host_packages():
    bad = _violations(PKG, ())
    assert not bad, '로버 코드가 호스트 패키지를 import:\n' + '\n'.join(bad)
