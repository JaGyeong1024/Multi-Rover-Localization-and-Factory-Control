"""lap_view: a hand-made recording becomes one self-contained HTML page (no simulator run)."""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scenarios'))
import lap_view  # noqa: E402


def _row(x, v):
    # true x, y, heading; reported x, y, heading, aligned, quality; state, detail; gate, reasons; v, w
    return [x, 1.0, 0.0, x + 0.01, 1.0, 0.02, True, 'GOOD', 'TRACKING', '', 'OK', [], v, 0.0]


def _recording():
    return {
        'meta': {'layout': 'empty', 'parts': ['gear_01'], 'seed': 1, 'profile': 'none',
                 'sample_period': 0.1, 'result': 'x <b>"quoted"</b>', 'completed': True, 'collisions': 0},
        'static': {'arena': {'x_min': 0.25, 'x_max': 5.1, 'y_min': 0.35, 'y_max': 3.1}, 'walls': True,
                   'shelves': [{'kind': 'rect', 'name': 'shelf_gear', 'x': 4.8, 'y': 1.24,
                                'w': 0.4, 'h': 0.4, 'yaw': 0.0}],
                   'goals': {'pick_gear': {'x': 4.3, 'y': 1.24, 'theta': 0.0}},
                   'footprint': {'leader': [0.13, 0.23, 0.18], 'follower': [0.15, 0.15, 0.155]}},
        't': [0.0, 0.1, 0.2],
        'rovers': {'leader': [_row(0.9, 0.0), _row(0.91, 0.05), _row(0.92, 0.1)],
                   'follower': [_row(0.9, 0.0), _row(0.9, 0.0), _row(0.9, 0.0)]},
        'paths': [{'t': 0.0, 'rover': 'leader', 'cmd_id': 1, 'mode': 'FOLLOW', 'pts': [[0.9, 1.0], [1.0, 1.0]]}],
        'mapped': [{'t': 0.1, 'shapes': [[2.6, 1.6, 0.05, 0.4, 0.4, 0.0]]}],
        'boxes': [{'t': 0.0, 'shapes': []}],
        'states': [[0.0, 'IDLE'], [0.1, 'RUNNING']],
        'events': [[0.1, 'path_issued', 'leader 5 points']]}


def test_html_embeds_the_recording():
    recording = _recording()
    html = lap_view.build_html(recording)
    assert lap_view.PLACEHOLDER not in html
    assert 'src=' not in html and '<link' not in html
    embedded = re.search(r'const DATA = (.*?);\n', html, re.S).group(1)
    assert json.loads(embedded) == recording


def test_no_external_reference_outside_data():
    recording = _recording()
    recording['meta']['result'] = 'see http://example.org/x'
    html = lap_view.build_html(recording)
    outside = html.replace(lap_view.recording_json(recording), '')
    assert 'http://' not in outside and 'https://' not in outside


def test_template_missing_placeholder_rejected():
    with pytest.raises(ValueError):
        lap_view.build_html(_recording(), '<html></html>')
