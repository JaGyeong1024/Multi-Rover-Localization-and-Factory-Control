"""Dict (JSON) command interface to a world: faults, obstacles, pose resets.

Every command is a dict with a "cmd" key. Examples:
  {"cmd": "fault", "type": "pose_blackout", "robot": "leader", "duration": 3.0}
  {"cmd": "fault", "type": "pose_jump", "robot": "leader", "dx": 0.5, "dy": 0.0}
  {"cmd": "fault", "type": "quality", "robot": "follower", "quality": "LOST"}
  {"cmd": "clear_faults"}                      (optional "robot")
  {"cmd": "add_obstacle", "type": "circle", "x": 3.0, "y": 3.0, "radius": 0.1}
  {"cmd": "remove_obstacle", "name": "box"}
  {"cmd": "reset_pose", "robot": "leader", "x": 2.0, "y": 3.0, "theta": 0.0}
Other commands: clear_obstacles, reset_record; fault types: see ``_FAULTS``.
"""
from host_sim.core.world import shape_from_dict


def apply_command(world, command):
    """Apply one command dict to the world; returns a short result text.

    Raises ValueError (or KeyError for a missing field / unknown rover) on bad input.
    """
    command = dict(command)
    name = command.pop('cmd', None)
    handler = _HANDLERS.get(name)
    if handler is None:
        raise ValueError(f'unknown cmd {name!r}; expected one of {sorted(_HANDLERS)}')
    return handler(world, command)


def _fault(world, args):
    kind = args.pop('type', None)
    handler = _FAULTS.get(kind)
    if handler is None:
        raise ValueError(f'unknown fault type {kind!r}; expected one of {sorted(_FAULTS)}')
    robot = args.pop('robot')
    if robot not in world.robots:
        raise KeyError(f'unknown robot {robot!r}')
    result = handler(world.injector, robot, **args)
    return f'fault {kind} {robot}' + (f' -> {result}' if result else '')


def _clear_faults(world, args):
    world.injector.clear(args.get('robot'))
    return 'faults cleared'


def _add_obstacle(world, args):
    return f'obstacle {world.add_obstacle(shape_from_dict(args))} added'


def _remove_obstacle(world, args):
    found = world.remove_obstacle(args['name'])
    return f"obstacle {args['name']} " + ('removed' if found else 'not found')


def _clear_obstacles(world, _args):
    world.clear_obstacles()
    return 'obstacles cleared'


def _reset_pose(world, args):
    world.reset_pose(args['robot'], args['x'], args['y'], args.get('theta', 0.0))
    return f"pose of {args['robot']} reset"


def _reset_record(world, _args):
    world.reset_record()
    return 'record reset'


_FAULTS = {
    'pose_blackout': lambda inj, robot, **kw: inj.pose_blackout(robot, **kw),
    'pose_jump': lambda inj, robot, **kw: inj.pose_jump(robot, **kw),
    'quality': lambda inj, robot, **kw: inj.set_quality(robot, **kw),
    'hold_unaligned': lambda inj, robot, **kw: inj.hold_unaligned(robot, **kw),
    'cmd_blocked': lambda inj, robot, **kw: inj.block_commands(robot, **kw),
    'stuck': lambda inj, robot, **kw: inj.stuck(robot, **kw),
    'camera_lost': lambda inj, robot, **kw: inj.camera_lost(robot, **kw),
    'obstacle_ahead': lambda inj, robot, **kw: inj.obstacle_ahead(robot, **kw),
}

_HANDLERS = {
    'fault': _fault,
    'clear_faults': _clear_faults,
    'add_obstacle': _add_obstacle,
    'remove_obstacle': _remove_obstacle,
    'clear_obstacles': _clear_obstacles,
    'reset_pose': _reset_pose,
    'reset_record': _reset_record,
}
