"""Checks on a finished lap, computed from the recorded TRUE state (``ScenarioResult.series`` /
``truth``): the rovers' own estimates and the engine's own verdicts are not evidence here.

``lap_failures`` returns {check: detail} for the checks that fail; a lap passes when it is empty."""
import math
import re

MISSION_STOPS = ('paused', 'phase_timeout', 'halt', 'fault', 'aborted')
IDLE = (0, 0)                    # (epoch, cmd_id) of the command a stack starts with
LEAD_TOL = 0.05                  # m, follower progress may exceed the leader's by this much
                                 # (the rover's own goal_pos_tol; projection noise)


def rect_corners(cx, cy, yaw, half_len, half_wid):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        px, py = a * half_len, b * half_wid
        pts.append((cx + c * px - s * py, cy + s * px + c * py))
    return pts


def footprint_rect(point, footprint):
    """Corners of a planner footprint [front, rear, half_width] at a tracked point (x, y, theta)
    (host_planning.geometry.body_shapes: centred (front - rear) / 2 ahead of the point)."""
    front, rear, half_wid = footprint
    x, y, theta = point
    shift = (front - rear) / 2.0
    return rect_corners(x + shift * math.cos(theta), y + shift * math.sin(theta), theta,
                        (front + rear) / 2.0, half_wid)


def box_rect(box):
    return rect_corners(box.x, box.y, box.yaw, box.width / 2.0, box.height / 2.0)


def _axes(poly):
    out = []
    for i in range(4):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % 4]
        n = math.hypot(x1 - x0, y1 - y0)
        out.append((-(y1 - y0) / n, (x1 - x0) / n))
    return out


def _point_segment(p, a, b):
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def rect_gap(a, b):
    """Gap between two rectangles (corner lists): > 0 apart, 0 touching, < 0 overlapping
    (minus the smallest penetration along a separating-axis candidate)."""
    sep = -math.inf
    for nx, ny in _axes(a) + _axes(b):
        pa = [nx * x + ny * y for x, y in a]
        pb = [nx * x + ny * y for x, y in b]
        sep = max(sep, min(pb) - max(pa), min(pa) - max(pb))
    if sep <= 0.0:
        return sep
    return min(min(_point_segment(p, b[i], b[(i + 1) % 4]) for p in a for i in range(4)),
               min(_point_segment(p, a[i], a[(i + 1) % 4]) for p in b for i in range(4)))


def project(poses, point):
    """(arc length, distance) of the point on the polyline ``poses`` (objects with x, y)
    closest to ``point``."""
    best, best_s, s = math.inf, 0.0, 0.0
    for p, q in zip(poses, poses[1:]):
        seg = math.hypot(q.x - p.x, q.y - p.y)
        if seg > 0.0:
            t = max(0.0, min(1.0, ((point[0] - p.x) * (q.x - p.x)
                                   + (point[1] - p.y) * (q.y - p.y)) / (seg * seg)))
            d = math.hypot(point[0] - p.x - t * (q.x - p.x), point[1] - p.y - t * (q.y - p.y))
            if d < best:
                best, best_s = d, s + t * seg
        s += seg
    return best_s, best


def loaded_commands(result):
    """rover -> distinct (epoch, cmd_id) the tracker actually loaded (the idle one excluded)."""
    return {rover: {(s.epoch, s.cmd_id) for s in rows} - {IDLE}
            for rover, rows in result.series.items()}


def final_goal_errors(result):
    """rover -> (position error [m], heading error [rad]) of the TRUE tracked point at the end
    against the last pose of the last path the rover was given."""
    out = {}
    for rover, rows in result.series.items():
        last = rows[-1]
        cmd = next((c for key in (((rover, s.epoch, s.cmd_id) for s in reversed(rows)))
                    if (c := result.host_cmds.get(key)) is not None and c.mode != 'STOP'
                    and c.poses), None)
        goal = cmd.poses[-1]
        out[rover] = (math.hypot(last.x - goal.x, last.y - goal.y),
                      abs(math.remainder(last.theta - goal.theta, math.tau)))
    return out


def goal_legs(result, rover):
    """Goal keys the engine sent ``rover`` to, in order, with the cmd ids that served each."""
    legs, ids = [], {}
    for event in result.events:
        m = event[1] == 'path_issued' and re.match(r'(\w+) -> (.+?) \(cmd (\d+),', event[2])
        if m and m.group(1) == rover:
            if m.group(2) not in legs:
                legs.append(m.group(2))
            ids[int(m.group(3))] = m.group(2)
    return legs, ids


def arrival_times(result, rover):
    """goal key -> first time the TRUE tracked point stood within arrive_pos_tol of the pose
    the loaded path ended in (None = never). The end of a path to a turn-in point
    (``PathCmd.via``) is no arrival: the goal is reached on the path that follows it."""
    legs, ids = goal_legs(result, rover)
    tol = result.robot_settings[rover].arrive_pos_tol
    out = {leg: None for leg in legs}
    for s in result.series[rover]:
        # a command re-issued without a 'path_issued' event (flags changed) serves the goal of
        # the last issued one before it
        goal_key = ids.get(max((i for i in ids if i <= s.cmd_id), default=None))
        cmd = result.host_cmds.get((rover, s.epoch, s.cmd_id))
        if goal_key is None or out[goal_key] is not None or cmd is None or not cmd.poses \
                or cmd.via:
            continue
        if math.hypot(s.x - cmd.poses[-1].x, s.y - cmd.poses[-1].y) <= tol:
            out[goal_key] = s.t
    return legs, out


def leg_order(result):
    """[(leader goal, leader arrival, follower goal, follower arrival)] per leg, paired by order."""
    ll, la = arrival_times(result, 'leader')
    fl, fa = arrival_times(result, 'follower')
    return [(a, la[a], b, fa[b]) for a, b in zip(ll, fl)], (len(ll), len(fl))


def lead_excess(result):
    """Largest amount [m] by which the follower's progress along the leader's issued path
    exceeded the leader's, over the time the follower's command names the leader and the
    follower is on that path's lane (closer to the path than the two footprints' half-widths
    together). Returns (excess, samples counted)."""
    lane = sum(s.plan.footprint[2] for s in result.robot_settings.values())
    worst, n = 0.0, 0
    for lead, foll in zip(result.series['leader'], result.series['follower']):
        if foll.lead_robot != 'leader':
            continue
        cmd = result.host_cmds.get(('leader', lead.epoch, lead.cmd_id))
        if cmd is None or len(cmd.poses) < 2:
            continue
        s_f, d_f = project(cmd.poses, (foll.x, foll.y))
        if d_f > lane:
            continue
        n += 1
        worst = max(worst, s_f - project(cmd.poses, (lead.x, lead.y))[0])
    return worst, n


def footprint_gaps(result):
    """Smallest gap over the lap between the two planner footprints, and between each rover's
    footprint and every true box / shelf rectangle (negative = overlap)."""
    fp = {r: s.plan.footprint for r, s in result.robot_settings.items()}
    boxes = [box_rect(b) for b in result.boxes]
    rr = math.inf
    rb = {'leader': math.inf, 'follower': math.inf}
    for _, lead, foll in result.truth:
        a, b = footprint_rect(lead, fp['leader']), footprint_rect(foll, fp['follower'])
        rr = min(rr, rect_gap(a, b))
        for rover, poly in (('leader', a), ('follower', b)):
            for box in boxes:
                rb[rover] = min(rb[rover], rect_gap(poly, box))
    return rr, rb


def min_centre_distance(result):
    return min(math.hypot(a[0] - b[0], a[1] - b[1]) for _, a, b in result.truth)


def dropout_report(result, rover, t0, duration):
    """What one camera-dropout window did to ``rover`` (true time series, window t0..t0+duration):
    seconds until its quality read LOST, until its output speed was 0 after that, whether it
    moved again inside the window, whether it drove just before the window and how long after
    the window it drove again (None = never / did not happen)."""
    rows = result.series[rover]
    window = [s for s in rows if t0 <= s.t <= t0 + duration]
    lost = next((s.t for s in window if s.quality == 'LOST'), None)
    stopped = next((s.t for s in window if lost is not None and s.t >= lost and s.v == 0.0), None)
    return {
        'lost_after': None if lost is None else round(lost - t0, 2),
        'stopped_after': None if stopped is None else round(stopped - t0, 2),
        'moved_after_stop': stopped is not None and any(
            s.v != 0.0 for s in window if s.t >= stopped),
        'moving_before': any(s.v != 0.0 for s in rows if t0 - 1.0 <= s.t < t0),
        'resumed_after': next((round(s.t - (t0 + duration), 2) for s in rows
                               if s.t > t0 + duration and s.v != 0.0), None),
    }


def dropout_failures(result, windows, duration):
    """{check: detail}: in every camera-dropout window the rover's quality became LOST, its
    output speed went to 0 and stayed there to the window's end, and it drove again afterwards
    (when it was driving before). ``windows``: [(t0, rover)]."""
    fails = {}
    for t0, rover in windows:
        rep = dropout_report(result, rover, t0, duration)
        bad = (rep['lost_after'] is None or rep['stopped_after'] is None or rep['moved_after_stop']
               or (rep['moving_before'] and rep['resumed_after'] is None))
        if bad:
            fails[f'dropout.{rover}@{t0:g}'] = str(rep)
    return fails


def dropout_summary(result, windows, duration):
    return {f'{rover}@{t0:g}': dropout_report(result, rover, t0, duration) for t0, rover in windows}


def lap_failures(result, parts=1, tol=None, clean=True, front_stop=True, platoon=True):
    """{check: detail} of every failed check of a completed lap (see the module docstring).

    parked_true      true tracked point / heading within the goal tolerances of the true goal
    loaded_cmds      (checked by the caller against its bound; reported by ``summary``)
    no_mission_stop  no pause / fault / halt event during the lap
    front_stop       no LIDAR_STOP of the leader (all obstacles are mapped before it gets there)
    platoon_arrival  on each leg the follower does not arrive before the leader
    platoon_lead     while the follower follows the leader it is not ahead of it on its path
    footprint        planner footprints of the rovers / boxes never overlap
    """
    fails = {}
    for rover, (pos, yaw) in final_goal_errors(result).items():
        cfg = result.robot_settings[rover]
        bad_yaw = cfg.goal_yaw_tol > 0.0 and yaw > cfg.goal_yaw_tol
        if pos > cfg.goal_pos_tol or bad_yaw:
            fails[f'parked_true.{rover}'] = (f'true error {pos:.3f} m / {yaw:.3f} rad against '
                                             f'{cfg.goal_pos_tol} m / {cfg.goal_yaw_tol} rad')
    if clean:
        stops = [e for e in result.events if e[1] in MISSION_STOPS]
        if stops:
            fails['no_mission_stop'] = str(stops[:3])
    if front_stop and result.blocked['leader'].get('LIDAR_STOP'):
        fails['front_stop'] = f"leader LIDAR_STOP x{result.blocked['leader']['LIDAR_STOP']}"
    if platoon:
        legs, counts = leg_order(result)
        if counts[0] != counts[1]:
            fails['platoon_arrival'] = f'leader goals {counts[0]} != follower goals {counts[1]}'
        for lg, lt, fg, ft in legs:
            if lt is None or ft is None or ft < lt:
                fails['platoon_arrival'] = (f'{fg} reached at {ft} before/without {lg} at {lt} '
                                            f'(follower / leader arrival, s)')
        worst, _ = lead_excess(result)
        if worst > LEAD_TOL:
            fails['platoon_lead'] = f'follower {worst:.3f} m ahead of the leader on its path'
    rr, rb = footprint_gaps(result)
    if rr < 0.0 or min(rb.values()) < 0.0:
        fails['footprint'] = f'min gap rover-rover {rr:.3f} m, rover-box {rb}'
    return fails


def summary(result):
    """One line of the measured values behind the checks, for the result table."""
    ids = loaded_commands(result)
    rr, rb = footprint_gaps(result)
    legs, _ = leg_order(result)
    errs = {r: (round(p, 3), round(y, 3)) for r, (p, y) in final_goal_errors(result).items()}
    return (f'loaded={ {r: len(v) for r, v in ids.items()} } '
            f'lidar_stop={result.blocked["leader"].get("LIDAR_STOP", 0)} '
            f'true_err={errs} gap_rr={rr:.3f} gap_rb={ {r: round(g, 3) for r, g in rb.items()} } '
            f'min_centre={min_centre_distance(result):.3f} '
            f'legs={[(lg, lt and round(lt, 1), fg, ft and round(ft, 1)) for lg, lt, fg, ft in legs]} '
            f'lead_excess={lead_excess(result)[0]:.3f}')
