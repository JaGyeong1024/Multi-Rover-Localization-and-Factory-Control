"""Small, stateless UWB pose correction helpers.

The correction is intentionally memoryless: every input pose maps to one output
pose with no time averaging. Raw /uwb/leader_pose remains the source of record.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorrectionParams:
    model: str = 'none'
    offset: tuple[float, float] = (0.0, 0.0)
    affine_x: tuple[float, float, float] = (1.0, 0.0, 0.0)
    affine_y: tuple[float, float, float] = (0.0, 1.0, 0.0)


def _as_float_tuple(values, expected_len: int, name: str) -> tuple[float, ...]:
    vals = tuple(float(v) for v in values)
    if len(vals) != expected_len:
        raise ValueError(f'{name} must contain {expected_len} floats, got {len(vals)}')
    return vals


def make_correction_params(model, offset, affine_x, affine_y) -> CorrectionParams:
    return CorrectionParams(
        model=str(model).lower(),
        offset=_as_float_tuple(offset, 2, 'offset'),
        affine_x=_as_float_tuple(affine_x, 3, 'affine_x'),
        affine_y=_as_float_tuple(affine_y, 3, 'affine_y'),
    )


def correct_xy(x: float, y: float, params: CorrectionParams) -> tuple[float, float]:
    """Apply a configured memoryless 2D correction."""
    model = params.model
    if model in ('none', 'passthrough', 'raw', ''):
        return x, y
    if model == 'offset':
        ox, oy = params.offset
        return x + ox, y + oy
    if model == 'affine':
        ax, ay, a0 = params.affine_x
        bx, by, b0 = params.affine_y
        return ax * x + ay * y + a0, bx * x + by * y + b0
    raise ValueError(f'unsupported UWB correction model: {params.model}')
