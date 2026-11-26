"""Small shared helpers for the simulator core."""
import dataclasses
import math


def wrap_angle(angle):
    """Wrap an angle to (-pi, pi]."""
    wrapped = math.atan2(math.sin(angle), math.cos(angle))
    return math.pi if wrapped == -math.pi else wrapped


def build_params(cls, values):
    """Build a params dataclass from a dict, rejecting unknown keys (typo guard)."""
    values = dict(values or {})
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f'{cls.__name__}: unknown keys {unknown}')
    return cls(**values)
