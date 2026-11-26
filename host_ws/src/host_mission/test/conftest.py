"""A skipped test counts as a failure unless ALLOW_SKIPS=1.

Skips here mean a missing environment (ROS not sourced, host_sim not on the path),
and a silent skip once hid four mission regressions. Run with ROS sourced; on a
rover, where host_sim is absent, pass ALLOW_SKIPS=1 explicitly.
"""
import os


def pytest_sessionfinish(session, exitstatus):
    reporter = session.config.pluginmanager.get_plugin('terminalreporter')
    skipped = reporter.stats.get('skipped', []) if reporter else []
    if skipped and exitstatus == 0 and os.environ.get('ALLOW_SKIPS') != '1':
        reporter.write_line(f'{len(skipped)} skipped -> failing (environment incomplete; '
                            'source ROS or set ALLOW_SKIPS=1)', red=True)
        session.exitstatus = 1
