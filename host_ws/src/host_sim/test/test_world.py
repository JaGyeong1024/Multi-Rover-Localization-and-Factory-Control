"""World: collision checks, minimum distance record, config loading."""
import math

import pytest
from conftest import fixed_frame_config, instant_params, make_world

from host_sim.core import Circle, Rect, Robot, World
from host_sim.core.world import KIND_ARENA, KIND_OBSTACLE, KIND_ROBOT


def run(world, command, duration, dt=0.01, name='leader'):
    for _ in range(int(round(duration / dt))):
        world.set_command(name, *command)
        world.step(dt)


def test_obstacle_clearance_collision_event(world):
    world.add_obstacle(Circle(4.0, 3.0, 0.2, name='post'))
    world.step(0.01)
    assert world.record.min_obstacle_clearance == pytest.approx(1.0 - 0.2 - 0.15)
    assert world.record.collisions == []
    run(world, (0.2, 0.0), 4.0)  # 0.8 m forward: footprint overlaps the post
    events = world.record.collisions
    assert [(e.kind, e.robot, e.other) for e in events] == [(KIND_OBSTACLE, 'leader', 'post')]
    assert events[0].t == pytest.approx(0.65 / 0.2, abs=0.03)
    assert world.record.min_distance < 0.0


def test_collision_reported_once_per_contact(world):
    world.add_obstacle(Rect(3.5, 3.0, 0.2, 0.2, name='box'))
    run(world, (0.2, 0.0), 2.0)   # into the box
    run(world, (-0.2, 0.0), 2.0)  # back out
    run(world, (0.2, 0.0), 2.0)   # into it again
    assert len(world.record.collisions) == 2


def test_rect_clearance_to_surface(world):
    world.add_obstacle(Rect(3.0, 4.0, 1.0, 0.4))
    world.step(0.01)
    assert world.record.min_obstacle_clearance == pytest.approx(1.0 - 0.2 - 0.15)


def test_robot_robot_gap_and_collision(world):
    world.add_robot('follower', Robot(instant_params(radius=0.13), 2.0, 3.0, 0.0))
    world.step(0.01)
    assert world.record.min_robot_gap == pytest.approx(1.0 - 0.15 - 0.13)
    run(world, (0.2, 0.0), 4.0, name='follower')
    kinds = [(e.kind, e.robot, e.other) for e in world.record.collisions]
    assert kinds == [(KIND_ROBOT, 'follower', 'leader')]
    assert world.current_min_distance < 0.0


def test_leaving_the_arena_is_a_collision():
    world = make_world(start=(5.9, 3.0, 0.0))
    run(world, (0.2, 0.0), 1.0)
    assert [(e.kind, e.other) for e in world.record.collisions] == [(KIND_ARENA, 'arena')]


def test_reset_record(world):
    world.add_obstacle(Circle(3.1, 3.0, 0.1))
    world.step(0.01)
    assert world.record.collisions
    world.clear_obstacles()
    world.reset_record()
    world.step(0.01)
    assert world.record.collisions == [] and math.isinf(world.record.min_distance)


def test_obstacle_names_and_removal(world):
    first = world.add_obstacle(Circle(4.0, 3.0, 0.1))
    second = world.add_obstacle(Circle(4.0, 4.0, 0.1, name='mine'))
    assert (first, second) == ('obs1', 'mine')
    assert world.remove_obstacle('mine') and not world.remove_obstacle('mine')
    assert [o.name for o in world.obstacles] == ['obs1']


def test_default_config_two_rovers_in_arena():
    world = World.from_config(fixed_frame_config(), seed=1)
    assert sorted(world.robots) == ['follower', 'leader']
    assert (world.arena.x_min, world.arena.x_max) == (1.24, 6.03)
    assert (world.arena.y_min, world.arena.y_max) == (1.97, 4.66)
    for robot in world.robots.values():
        assert world.arena.contains(robot.x, robot.y)
    assert world.robots['leader'].params.tread_ratio < world.robots['follower'].params.tread_ratio
    assert world.sensors['leader'].params.delay == 0.08
    assert world.lidars['leader'].params.num_beams == 360
    assert sorted(world.depth_scans) == ['leader']              # leader depth camera
    assert world.depth_scans['leader'].params.out_of_range_value == 0.0
    world.step(0.01)
    assert world.record.collisions == []


def test_config_obstacles_and_typo_guard():
    config = {'robots': {'leader': {'start': [3.0, 3.0, 0.0]}},
              'obstacles': [{'type': 'circle', 'x': 4.0, 'y': 3.0, 'radius': 0.1},
                            {'type': 'rect', 'x': 5.0, 'y': 3.0, 'width': 0.2, 'height': 0.2}]}
    assert len(World.from_config(config).obstacles) == 2
    with pytest.raises(ValueError):
        World.from_config({'robots': {'leader': {'base': {'tred_ratio': 0.7}}}})
    with pytest.raises(ValueError):
        World.from_config({'obstacles': [{'type': 'triangle'}]})
