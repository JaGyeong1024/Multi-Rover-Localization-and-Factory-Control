"""Collects one result line per scenario and prints them as a table after the run.

Kept free of pytest >= 7 API (StashKey): ROS 2 Humble ships pytest 6.2.
"""
import pytest

_ROWS = []


@pytest.fixture
def record_row():
    return _ROWS.append


def pytest_terminal_summary(terminalreporter):
    if _ROWS:
        from layout_runner import front_path_ab
        terminalreporter.section(f'layout scenario results (FRONT_PATH_AB={front_path_ab()})')
        for row in _ROWS:
            terminalreporter.write_line(row)
