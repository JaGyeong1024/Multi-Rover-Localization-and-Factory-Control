"""host_ws/run_host.sh starts the path mission by default; the old lane-graph
mission is retired (MISSION=lane is refused).

Source-level: the script needs the real sensors and ROS, so it is read, not run (bash -n
checks the syntax)."""
import os
import re
import subprocess

from helpers import PACKAGE, SITE_CONFIG

SCRIPT = os.path.join(PACKAGE, '..', '..', 'run_host.sh')


def _source():
    with open(SCRIPT, encoding='utf-8') as f:
        return f.read()


def _code():
    """The script without comment lines (the header documents every mode)."""
    return '\n'.join(line for line in _source().splitlines()
                     if not line.lstrip().startswith('#'))


def _branch(code, mode):
    """Body of the ``"$MISSION" = <mode>`` if/elif branch (up to the next elif/else/fi)."""
    match = re.search(r'(?:if|elif) \[ "\$MISSION" = %s \]; then\n(.*?)\n(?:elif|else|fi)\b'
                      % mode, code, re.S)
    assert match, f'no MISSION={mode} branch'
    return match.group(1)


def test_script_parses():
    subprocess.run(['bash', '-n', SCRIPT], check=True)


def test_default_mission_is_on():
    match = re.search(r'^MISSION="\$\{MISSION:-(\w+)\}"', _code(), re.M)
    assert match and match.group(1) == 'on'


def test_on_branch_launches_mission_with_bridges():
    body = _branch(_code(), 'on')
    assert 'ros2 launch host_mission mission.launch.py' in body
    assert 'bridges:=false' not in body          # the rovers are on domains 25 / 73


def test_lane_mission_is_gone():
    code = _code()
    assert 'mission_runtime.launch.py' not in code
    assert 'MISSION" = lane' not in code


def test_unknown_mission_value_is_refused():
    assert re.search(r'case "\$MISSION" in\s*\n?\s*on\|off\)', _code())


def test_banner_queue_example_names_a_part():
    from host_mission.settings import load_scenario
    parts = load_scenario(os.path.join(SITE_CONFIG, 'mission_scenario.yaml')).parts
    examples = re.findall(r"/task/add std_msgs/String \\\"data: 'fetch:(\w+)'\\\"", _code())
    assert examples and all(part in parts for part in examples), examples
