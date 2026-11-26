"""A skipped test counts as a failure unless ALLOW_SKIPS=1.

Skips mean a missing environment (ROS not sourced, host_sim not on the path) and once hid
regressions. On a rover, where host_sim is absent, pass ALLOW_SKIPS=1 explicitly.
"""
import importlib.util
import os

# On a rover host_sim does not exist: leave its closed-loop tests out by name (a module-level
# importorskip would make pytest collect nothing at all).
collect_ignore = []
if os.environ.get('ALLOW_SKIPS') == '1' and importlib.util.find_spec('host_sim') is None:
    collect_ignore += ['test_host_sim.py', 'test_platoon_host_sim.py',
                       'test_xtrack_integral_host_sim.py', 'test_fused_host_sim.py',
                       'test_host_loop_depth.py']


def pytest_sessionfinish(session, exitstatus):
    reporter = session.config.pluginmanager.get_plugin('terminalreporter')
    skipped = reporter.stats.get('skipped', []) if reporter else []
    if skipped and exitstatus == 0 and os.environ.get('ALLOW_SKIPS') != '1':
        reporter.write_line(f'{len(skipped)} skipped -> failing (environment incomplete; '
                            'source ROS or set ALLOW_SKIPS=1)', red=True)
        session.exitstatus = 1
