#!/usr/bin/env python3
"""Generate the dashboard's map data (mission_map.json) from the mission configuration.

    python3 dashboard/gen_mission_map.py --settings <mission.yaml> [--overlay <yaml>]... \
        --scenario <mission_scenario.yaml> --goals <goals.yaml> \
        [--fixtures-from <layouts.yaml>] [-o dashboard/public/mission_map.json]

Pass the files the coordinator runs with (overlays merge in order). Output (uwb_global, m, rad):
bounds, fixtures, goals (relative goals resolved), roles (goal names), parts (drawn on the
fixture the pick goal faces, else 0.30 m ahead of it). ``--fixtures-from`` replaces the fixtures
by the ``shelves`` of a scenarios layouts file. Needs PyYAML only.
"""
import argparse
import json
import math
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(REPO, 'host_ws', 'src', 'host_mission'),
                os.path.join(REPO, 'host_ws', 'src', 'host_planning')]
from host_mission.goals import GoalStore  # noqa: E402  (pure Python, no rclpy)
from host_mission.settings import read_settings_yaml  # noqa: E402

PART_AHEAD = 0.30   # m, part marker ahead of a pick goal that faces no fixture
FACE_REACH = 0.80   # m, a fixture centre this close ahead of the pick goal is its shelf


def _read(path):
    with open(path) as stream:
        return yaml.safe_load(stream) or {}


def _fixture(entry):
    return {'name': str(entry.get('name', '')), 'x': float(entry['x']), 'y': float(entry['y']),
            'width': float(entry['width']), 'height': float(entry['height']),
            'yaw': float(entry.get('yaw', 0.0))}


def _faced_fixture(goal, fixtures):
    """Nearest fixture whose centre lies ahead of the goal within FACE_REACH, or None."""
    best, best_d = None, FACE_REACH
    for f in fixtures:
        dx, dy = f['x'] - goal['x'], f['y'] - goal['y']
        ahead = dx * math.cos(goal['theta']) + dy * math.sin(goal['theta'])
        dist = math.hypot(dx, dy)
        if ahead > 0.0 and dist <= best_d:
            best, best_d = f, dist
    return best


def build(settings_path, scenario_path, goals_path, fixtures_from=None, overlays=()):
    settings = read_settings_yaml(settings_path, overlays)
    scenario = _read(scenario_path)
    store = GoalStore.load(goals_path)
    goals = store.resolved()
    if fixtures_from:
        raw_fixtures = _read(fixtures_from).get('shelves') or []
    else:
        raw_fixtures = settings.get('fixtures') or []
    fixtures = [_fixture(f) for f in raw_fixtures]
    b = settings.get('bounds')
    bounds = ({k: float(b[k]) for k in ('xmin', 'xmax', 'ymin', 'ymax')} if b else None)

    parts = []
    for p in scenario.get('parts') or []:
        pick = goals.get(p.get('leader_goal'))
        shelf = _faced_fixture(pick, fixtures) if pick else None
        if shelf:
            x, y = shelf['x'], shelf['y']
        elif pick:
            x = pick['x'] + PART_AHEAD * math.cos(pick['theta'])
            y = pick['y'] + PART_AHEAD * math.sin(pick['theta'])
        else:
            x = y = None                    # pick goal unknown: listed, not drawn
        parts.append({'id': str(p['id']), 'leader_goal': p.get('leader_goal'),
                      'follower_goal': p.get('follower_goal'), 'x': x, 'y': y,
                      'fixture': shelf['name'] if shelf else None})
    roles = {k: scenario.get(k) for k in ('leader_standby', 'follower_unload', 'follower_home')
             if scenario.get(k)}
    return {
        'frame_id': 'uwb_global',
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'sources': {'settings': os.path.abspath(settings_path),
                    'overlays': [os.path.abspath(o) for o in overlays],
                    'scenario': os.path.abspath(scenario_path),
                    'goals': os.path.abspath(goals_path),
                    'fixtures_from': os.path.abspath(fixtures_from) if fixtures_from else None},
        'bounds': bounds,
        'fixtures': fixtures,
        'goals': goals,
        'roles': roles,
        'parts': parts,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--settings', required=True)
    ap.add_argument('--overlay', action='append', default=[],
                    help='settings overlay merged over --settings (repeatable, later wins)')
    ap.add_argument('--scenario', required=True)
    ap.add_argument('--goals', required=True)
    ap.add_argument('--fixtures-from', default=None)
    ap.add_argument('-o', '--out', default=os.path.join(HERE, 'public', 'mission_map.json'))
    args = ap.parse_args(argv)
    data = build(args.settings, args.scenario, args.goals, args.fixtures_from, args.overlay)
    temp = args.out + '.tmp'
    with open(temp, 'w') as stream:
        json.dump(data, stream, indent=1)
    os.replace(temp, args.out)                 # the dev server never serves a half file
    print(f"[map] {args.out}: bounds={data['bounds']} fixtures={len(data['fixtures'])} "
          f"goals={len(data['goals'])} parts={[p['id'] for p in data['parts']]}")
    if data['bounds'] is None:
        print('[map] WARNING: the settings file has no bounds', file=sys.stderr)


if __name__ == '__main__':
    main()
