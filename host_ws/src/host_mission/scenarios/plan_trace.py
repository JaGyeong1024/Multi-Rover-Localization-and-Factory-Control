"""Diagnosis only: records the PathCommander decisions behind every re-plan and every
follower wait of a lap, by wrapping its methods for the duration of a ``with PlanTrace()``
block (the production module is not edited; the class is restored on exit).

Records (t, robot, kind, detail):
  replan_request  who asked for a fresh plan (calling function and line)
  recheck_failed  _rest_still_clear was False: the obstacles the rest of the path no longer
                  keeps clearance from, and how each moved since the path was issued
  follower_plan   _follower_plan branch: shared / NO_PATH / YIELD / stand-alone
  hold_ride       _ride_to_hold result, lead_waits / blocked_only_by_peer verdicts
  computed        _compute outcome per rover"""
import math
import sys

from host_planning import api as planning_api

from host_mission import path_commander as pc

_WRAPPED = ('update', 'request_replan', '_rest_still_clear', '_follower_plan', '_compute',
            '_ride_to_hold', '_lead_waits', '_blocked_only_by_peer', '_parks_in_lead_way')


def _shape(o):
    if hasattr(o, 'width'):
        return (f'box({o.x:.3f},{o.y:.3f} {o.width:.2f}x{o.height:.2f} '
                f'yaw {o.yaw:.2f} pad {getattr(o, "pad", 0.0):.2f})')
    return f'circle({o.x:.3f},{o.y:.3f} r {o.r:.2f})'


def _plan(plan):
    if plan is None:
        return 'none'
    tags = [f'{len(plan.poses)} poses']
    if plan.lead_robot:
        tags.append('behind ' + plan.lead_robot)
    if plan.via:
        tags.append('hold' if plan.hold else 'turn-in')
    if plan.pivot_heading is not None:
        tags.append(f'pivot to {plan.pivot_heading:.2f}')
    return ', '.join(tags)


class PlanTrace:
    def __init__(self):
        self.records = []
        self.now = 0.0
        self._saved = {}
        self._issued_world = {}     # (robot, cmd_id) -> obstacle shapes when issued

    def __enter__(self):
        cls = pc.PathCommander
        for name in _WRAPPED:
            self._saved[name] = getattr(cls, name)
            setattr(cls, name, self._method('_w' + name))
        return self

    def _method(self, wrapper_name):
        wrapper = getattr(self, wrapper_name)

        def method(this, *args, **kwargs):
            return wrapper(this, *args, **kwargs)
        return method

    def __exit__(self, *exc):
        for name, fn in self._saved.items():
            setattr(pc.PathCommander, name, fn)

    def _add(self, robot, kind, detail):
        self.records.append((round(self.now, 2), robot, kind, detail))

    # ---- wrappers (``this`` is the PathCommander, or its frozen twin) ----
    def _wupdate(self, this, desired, world, arrived, now):
        self.now = now
        before = dict(this.cmds)
        events = self._saved['update'](this, desired, world, arrived, now)
        for robot, cmd in this.cmds.items():
            if cmd is not before.get(robot):
                self._issued_world[(robot, cmd.cmd_id)] = [
                    pc._planner_shape(o) for o in world.obstacles()]
        return events

    def _wrequest_replan(self, this, robot, reentry=False):
        caller = sys._getframe(2)      # 0 this wrapper, 1 the class method shim
        self._add(robot, 'replan_request',
                   f'{caller.f_code.co_name}:{caller.f_lineno} reentry={reentry}')
        return self._saved['request_replan'](this, robot, reentry)

    def _w_rest_still_clear(self, this, robot, cmd, world, arrived, now):
        ok = self._saved['_rest_still_clear'](this, robot, cmd, world, arrived, now)
        if not ok:
            self._add(robot, 'recheck_failed',
                       self._culprits(this, robot, cmd, world, arrived, now))
        return ok

    def _culprits(self, this, robot, cmd, world, arrived, now):
        params = this._issued_params(robot, cmd)
        poses = list(cmd.poses)
        pose, _ = world.usable_pose(robot, now, need_heading=False)
        first = 0 if pose is None else pc._nearest_index(poses, pose)
        rest = poses[first:]
        ignore = this._lead_zone_ignore(params, world, arrived) if cmd.lead_robot else []
        mapped = [pc._planner_shape(o) for o in world.obstacles()]
        peers = this._parked_peers(robot, world, arrived)
        labelled = ([('mapped', o) for o in mapped] + [('fixture', o) for o in this.settings.fixtures]
                    + [('peer', o) for o in peers])
        old = self._issued_world.get((robot, cmd.cmd_id), [])
        out = [f'cmd {cmd.cmd_id} rest from pose {first}/{len(poses)}, clearance {params.clearance}']
        for label, o in labelled:
            if planning_api.keeps_clearance(rest, [o], params, ignore):
                continue
            gap = planning_api.min_body_gap(rest, [o], params)
            line = f'{label} {_shape(o)} body gap {gap:.3f}'
            if gap >= params.clearance:
                line += ' (front-stop zone)'
            if label == 'mapped' and old:
                near = min(old, key=lambda s: math.hypot(s.x - o.x, s.y - o.y))
                line += f' | at issue: {_shape(near)}'
            elif label == 'mapped':
                line += ' | at issue: no mapped obstacle'
            out.append(line)
        if len(out) == 1:
            out.append('the leading pivot sweep')
        return '; '.join(out)

    def _w_follower_plan(self, this, shared, lead_path, provisional, goal):
        plan, reason = self._saved['_follower_plan'](this, shared, lead_path, provisional, goal)
        follower, lead = this.settings.platoon.follower, this.settings.platoon.lead_robot
        if shared is None:
            why = 'plan_follower_path found nothing (NO_PATH)'
        elif not this._parks_clear_of_lead(goal, lead_path):
            spot = this._footprint_shape(follower, (goal.x, goal.y, goal.theta))
            gap = planning_api.min_body_gap(lead_path, [spot],
                                            this._issued_params(lead, this.cmds.get(lead)))
            why = (f'YIELD: parked at goal ({goal.x:.2f},{goal.y:.2f}) it is {gap:.3f} m from '
                   f'the lead\'s remaining path ({len(lead_path)} poses, lead clearance '
                   f'{this.settings.robots[lead].plan.clearance})')
        elif shared.shared:
            why = f'shared ({len(shared.path)} poses)'
        else:
            d = min(math.hypot(p.x - q.x, p.y - q.y) for p in shared.path for q in lead_path)
            why = (f'stand-alone {len(shared.path)} poses, min distance to the lead path '
                   f'{d:.3f} (conflict_dist {this.settings.platoon.conflict_dist}) -> '
                   f'{"YIELD" if reason else "released"}')
        self._add(follower, 'follower_plan', why)
        return plan, reason

    def _w_compute(self, this, robot, want, world, arrived, now):
        plan, reason = self._saved['_compute'](this, robot, want, world, arrived, now)
        pose = world.robots[robot].pose
        where = pose and f'from ({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.2f})'
        self._add(robot, 'computed', f'{want.goal_key} {where}: {_plan(plan)} {reason}')
        return plan, reason

    def _w_ride_to_hold(self, this, robot, start, goal, obstacles, params, world, arrived):
        plan = self._saved['_ride_to_hold'](this, robot, start, goal, obstacles, params,
                                             world, arrived)
        self._add(robot, 'hold_ride', _plan(plan))
        return plan

    def _w_lead_waits(self, this, robot, plan, world, arrived, now):
        out = self._saved['_lead_waits'](this, robot, plan, world, arrived, now)
        if out[0] is None:
            self._add(robot, 'lead_waits', 'YIELD: the follower drives a stand-alone path ahead')
        return out

    def _w_blocked_only_by_peer(self, this, robot, start, goal, provisional, world, params):
        out = self._saved['_blocked_only_by_peer'](this, robot, start, goal, provisional,
                                                    world, params)
        self._add(robot, 'blocked_only_by_peer', f'{out} (True = YIELD, False = NO_PATH)')
        return out

    def _w_parks_in_lead_way(self, this, robot, goal, world, arrived, now):
        out = self._saved['_parks_in_lead_way'](this, robot, goal, world, arrived, now)
        self._add(robot, 'parks_in_lead_way', str(out))
        return out
