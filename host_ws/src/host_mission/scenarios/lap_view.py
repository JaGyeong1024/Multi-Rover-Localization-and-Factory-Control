"""Record one simulated lap and write it as ONE self-contained HTML animation.

  lap_view.py LAYOUT PART[,PART...] [--seed N] [--profile field_site|none]
              [--out FILE.html] [--duration S]

Default output: lap_view_out/<layout>_<parts>_<profile>_seed<N>.html under the current directory.
The page is lap_view_template.html with the recording (JSON) put in place of its placeholder.
Frame: arena / ceiling camera frame (uwb_global), metres, radians, seconds; x right, y up.
"Pose" of a rover = its tracked point (what localization reports), heading = body heading.
Everything that needs the simulator is imported inside run_lap, so build_html and
recording_json work (and are tested) without it."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / 'lap_view_template.html'
PLACEHOLDER = '/*LAP_DATA*/null'
SAMPLE_PERIOD = 0.1              # s of simulated time between recorded samples


def recording_json(recording):
    """Compact JSON text of a recording; '<' escaped so it can sit inside a <script> element."""
    return json.dumps(recording, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')


def build_html(recording, template_text=None):
    """The page: the template with the recording in place of its placeholder."""
    template = template_text if template_text is not None else TEMPLATE.read_text(encoding='utf-8')
    if template.count(PLACEHOLDER) != 1:
        raise ValueError(f'template must contain {PLACEHOLDER!r} exactly once')
    return template.replace(PLACEHOLDER, recording_json(recording))


def _r(value, digits=3):
    return round(float(value), digits)


def _shape(o):
    """host_sim Rect / Circle obstacle -> dict (rect: centre x, y, size w, h in m, yaw in rad)."""
    if hasattr(o, 'radius'):
        return {'kind': 'circle', 'name': o.name, 'x': _r(o.x), 'y': _r(o.y), 'r': _r(o.radius)}
    return {'kind': 'rect', 'name': o.name, 'x': _r(o.x), 'y': _r(o.y), 'w': _r(o.width),
            'h': _r(o.height), 'yaw': _r(o.yaw)}


def _opt(value, convert):
    return None if value is None else convert(value)


class Recorder:
    """Wraps the loop's controllers; keeps one sample per SAMPLE_PERIOD and every change.

    Sample row per rover: [true x, y (tracked point), true heading, reported x, y, heading,
    heading_aligned, quality, tracker state, detail, gate state, gate reasons, v, w]
    (v in m/s and w in rad/s after the gate; reported fields are None before the first pose).
    """

    def __init__(self, loop):
        self.loop = loop
        self.names = list(loop.stacks)
        self.rows = {}                              # tick index -> {rover: row}
        self.paths, self.mapped, self.boxes, self.states = [], [], [], []
        self._cmd_id = {n: None for n in self.names}
        self._mapped_key = self._boxes_key = self._state = None

    def controllers(self):
        inner = self.loop.controllers()
        return {n: (lambda obs, n=n, f=inner[n]: self._step(obs, n, f)) for n in self.names}

    def _step(self, obs, name, inner):
        out = inner(obs)
        k = int(round(obs.t / SAMPLE_PERIOD))
        if abs(obs.t - k * SAMPLE_PERIOD) < 1e-6:
            self._sample(obs, name, out, k)
        return out

    def _sample(self, obs, name, out, k):
        loop, t = self.loop, _r(obs.t, 2)
        stack, robot, p = loop.stacks[name], loop.world.robots[name], obs.pose
        _, state, detail, gate, reasons, _, _ = stack.log[-1]
        x, y, theta = robot.tracked_point()
        self.rows.setdefault(k, {})[name] = [
            _r(x), _r(y), _r(theta),
            _opt(p, lambda p: _r(p.x)), _opt(p, lambda p: _r(p.y)), _opt(p, lambda p: _r(p.theta)),
            _opt(p, lambda p: bool(p.heading_aligned)), _opt(p, lambda p: p.quality),
            state, detail or '', gate, list(reasons), _r(out[0], 4), _r(out[1], 3)]
        cmd = stack.ctrl._cmd
        cmd_id = None if cmd is None else cmd.cmd_id
        if cmd_id != self._cmd_id[name]:
            self._cmd_id[name] = cmd_id
            self.paths.append({'t': t, 'rover': name, 'cmd_id': cmd_id,
                               'mode': None if cmd is None else cmd.mode,
                               'pts': [] if cmd is None else [[_r(q[0]), _r(q[1])] for q in cmd.poses]})
        if name == self.names[0]:
            self._sample_host(t)

    def _sample_host(self, t):
        loop = self.loop
        mapped = [[_r(v) for v in s] for s in loop.mapper.shapes()]   # x, y, pad, w, h, yaw
        if mapped != self._mapped_key:
            self._mapped_key = mapped
            self.mapped.append({'t': t, 'shapes': mapped})
        boxes = [_shape(o) for o in loop.world.obstacles]
        if boxes != self._boxes_key:
            self._boxes_key = boxes
            self.boxes.append({'t': t, 'shapes': boxes})
        mission = loop.engine.seq.system_state()
        if mission != self._state:
            self._state = mission
            self.states.append([t, mission])

    def recording(self, meta, static, result):
        ticks = sorted(k for k, r in self.rows.items() if len(r) == len(self.names))
        return {
            'meta': dict(meta, result=result.row(), completed=bool(result.completed),
                         collisions=len(result.collisions)),
            'static': static,
            't': [_r(k * SAMPLE_PERIOD, 2) for k in ticks],
            'rovers': {n: [self.rows[k][n] for k in ticks] for n in self.names},
            'paths': self.paths, 'mapped': self.mapped, 'boxes': self.boxes,
            'states': self.states,
            'events': [[e[0], e[1], ' '.join(str(v) for v in e[2:])] for e in result.events]}


def static_scene(loop, layout_runner):
    """Arena, shelves, goals and rover footprints, from the configs the stack itself uses."""
    from host_mission.goals import GoalStore
    arena = loop.world.arena
    robots = layout_runner.production_settings_raw()['robots']
    goals = GoalStore.load(str(layout_runner.DATA / 'goals.yaml')).resolved()
    return {
        'arena': {k: _r(getattr(arena, k)) for k in ('x_min', 'x_max', 'y_min', 'y_max')},
        'walls': bool(arena.walls_enabled),
        'shelves': [_shape(o) for o in loop.world.obstacles if o.name.startswith('shelf')],
        'goals': {n: {k: _r(v) for k, v in g.items()} for n, g in goals.items()},
        # (front, rear, half width) in m from the tracked point: plan.footprint of mission.yaml
        'footprint': {n: list(robots[n]['plan']['footprint']) for n in loop.stacks}}


def run_lap(layout, parts, seed=1, profile=None, duration=240.0):
    """Run the lap(s) once; returns the recording dict (duration = s of simulated time per part)."""
    import layout_runner as lr
    from host_sim.scenario import run_scenario
    loop = lr.LayoutLoop(layout, seed, None, profile)
    for part in parts:
        if not loop.engine.dispatch(part):
            raise ValueError(f'part {part!r} was not accepted by the mission engine')
    recorder = Recorder(loop)
    static = static_scene(loop, lr)
    try:
        run_scenario(loop.world, recorder.controllers(), duration=duration * len(parts),
                     stop_on_arrival=False)
    except lr._Finished:
        pass
    meta = {'layout': layout, 'parts': list(parts), 'seed': seed, 'profile': profile or 'none',
            'sample_period': SAMPLE_PERIOD}
    return recorder.recording(meta, static, loop.result(layout, '+'.join(parts)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('layout')
    ap.add_argument('parts', help='comma separated, e.g. gear_01 or gear_01,bolt_01')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--profile', default='field_site')
    ap.add_argument('--out')
    ap.add_argument('--duration', type=float, default=240.0, help='s of simulated time per part')
    args = ap.parse_args()
    parts = args.parts.split(',')
    profile = None if args.profile == 'none' else args.profile
    recording = run_lap(args.layout, parts, args.seed, profile, args.duration)
    out = Path(args.out or Path('lap_view_out') /
               f'{args.layout}_{"+".join(parts)}_{args.profile}_seed{args.seed}.html')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(recording), encoding='utf-8')
    print(recording['meta']['result'])
    print(f'{out.resolve()}  {out.stat().st_size / 1e6:.2f} MB  {len(recording["t"])} samples')


if __name__ == '__main__':
    main()
