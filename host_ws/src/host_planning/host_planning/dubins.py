"""Dubins words and their curvature-relaxed version (forward motion only).

A Dubins path is the shortest forward path between two poses under a curvature
bound. Its curvature jumps between 0 and +-1/R, so :func:`relax` replaces every
arc's rectangular curvature profile by a trapezoid (linear ramps) and re-solves the
three segment lengths with Gauss-Newton so the end pose is still met exactly.
"""
from __future__ import annotations

import functools
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .geometry import wrap_angle
from .types import Pose2D

Segment = Tuple[int, float]          # (turn sign: +1 left, 0 straight, -1 right, length m)
Word = Tuple[float, Tuple[Segment, ...]]  # (total length m, segments)

_TWO_PI = 2.0 * math.pi
_SNAP = 1e-9
_FD_STEP = 1e-5   # finite-difference step of the relaxation Jacobian (m)


def _mod2pi(angle: float) -> float:
    """Angle in [0, 2*pi), with values a hair below 2*pi snapped to 0."""
    value = angle % _TWO_PI
    return 0.0 if value > _TWO_PI - _SNAP else value


def _csc(alpha: float, beta: float, d: float, first: int, last: int):
    """Normalised (t, p, q) of a curve-straight-curve word, or None."""
    sa, ca, sb, cb = math.sin(alpha), math.cos(alpha), math.sin(beta), math.cos(beta)
    cab = math.cos(alpha - beta)
    if first == last:
        sign = first
        p_sq = 2.0 + d * d - 2.0 * cab + 2.0 * d * sign * (sa - sb)
        if p_sq < 0.0:
            return None
        tmp = math.atan2(sign * (cb - ca), d + sign * (sa - sb))
        return (_mod2pi(sign * (tmp - alpha)), math.sqrt(p_sq),
                _mod2pi(sign * (beta - tmp)))
    sign = first
    p_sq = -2.0 + d * d + 2.0 * cab + 2.0 * d * sign * (sa + sb)
    if p_sq < 0.0:
        return None
    p = math.sqrt(p_sq)
    tmp = (math.atan2(-sign * (ca + cb), d + sign * (sa + sb))
           - math.atan2(-2.0 * sign, p))
    return (_mod2pi(sign * (tmp - alpha)), p, _mod2pi(sign * (tmp - beta)))


def _ccc(alpha: float, beta: float, d: float, first: int):
    """Normalised (t, p, q) of a curve-curve-curve word, or None."""
    sa, ca, sb, cb = math.sin(alpha), math.cos(alpha), math.sin(beta), math.cos(beta)
    tmp = (6.0 - d * d + 2.0 * math.cos(alpha - beta)
           - 2.0 * d * first * (sa - sb)) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = _mod2pi(_TWO_PI - math.acos(tmp))
    base = math.atan2(ca - cb, d + first * (sa - sb))
    if first > 0:   # LRL
        t = _mod2pi(-alpha - base + p / 2.0)
        return (t, p, _mod2pi(beta - alpha - t + p))
    t = _mod2pi(alpha - base + p / 2.0)   # RLR
    return (t, p, _mod2pi(alpha - beta - t + p))


def end_pose(start: Pose2D, segments: Sequence[Segment],
             kappa: float) -> Tuple[float, float, float]:
    x, y, theta = start.x, start.y, start.theta
    for sign, length in segments:
        if sign == 0:
            x += length * math.cos(theta)
            y += length * math.sin(theta)
            continue
        turn = sign * kappa * length
        x += (math.sin(theta + turn) - math.sin(theta)) / (sign * kappa)
        y += (math.cos(theta) - math.cos(theta + turn)) / (sign * kappa)
        theta += turn
    return x, y, theta


def _reaches(start: Pose2D, goal: Pose2D, segments, kappa: float) -> bool:
    x, y, theta = end_pose(start, segments, kappa)
    return (math.hypot(x - goal.x, y - goal.y) < 1e-6
            and abs(wrap_angle(theta - goal.theta)) < 1e-6)


def dubins_words(start: Pose2D, goal: Pose2D, radius: float) -> List[Word]:
    """All feasible Dubins words between two poses, shortest first."""
    dx, dy = goal.x - start.x, goal.y - start.y
    d = math.hypot(dx, dy) / radius
    phi = math.atan2(dy, dx)
    alpha, beta = _mod2pi(start.theta - phi), _mod2pi(goal.theta - phi)
    shapes = [((1, 0, 1), _csc(alpha, beta, d, 1, 1)),
              ((-1, 0, -1), _csc(alpha, beta, d, -1, -1)),
              ((1, 0, -1), _csc(alpha, beta, d, 1, -1)),
              ((-1, 0, 1), _csc(alpha, beta, d, -1, 1)),
              ((-1, 1, -1), _ccc(alpha, beta, d, -1)),
              ((1, -1, 1), _ccc(alpha, beta, d, 1))]
    words: List[Word] = []
    for signs, tpq in shapes:
        if tpq is None:
            continue
        segments = tuple((s, v * radius) for s, v in zip(signs, tpq))
        if _reaches(start, goal, segments, 1.0 / radius):
            words.append((sum(seg[1] for seg in segments), segments))
    return sorted(words, key=lambda word: word[0])


def _sinc(x: np.ndarray) -> np.ndarray:
    """np.sinc without its array-function dispatch."""
    y = np.pi * np.where(x == 0, 1.0e-20, x)
    return np.sin(y) / y


def _read_only(*arrays):
    for array in arrays:
        array.setflags(write=False)
    return arrays


@functools.lru_cache(maxsize=1024)
def _layout(counts: Tuple[int, ...]):
    """Row bookkeeping of :func:`integrate_batch` for the segments' step counts (segment
    c has counts[c] + 1 sample rows). Returns, read-only: segment of each sample row,
    sample number of each row, last row of each segment, first / second sample row of
    each step."""
    steps = np.array(counts, dtype=int)
    sizes = steps + 1
    segment = np.repeat(np.arange(len(counts)), sizes)
    last = np.cumsum(sizes) - 1
    index = (np.arange(len(segment)) - (last - steps)[segment]).astype(float)[:, None]
    lo = np.arange(int(steps.sum())) + np.repeat(np.arange(len(counts)), steps)
    return _read_only(segment, index, last, lo, lo + 1)


def integrate_batch(start: Pose2D, signs: Sequence[int], lengths: np.ndarray,
                    kappa: float, ramp: float, counts: Sequence[int]) -> np.ndarray:
    """Sample the paths of one sign sequence for K length vectors: (K, N, 3).

    ``lengths`` is (K, segments). Each step uses the exact chord of a constant-curvature
    arc, so plain Dubins arcs (``ramp`` = 0) are exact. Curvature per segment is
    ``sign * kappa * min(1, min(s, L - s) / ramp)`` (constant without ramp, 0 on a
    straight); heading and position accumulate step by step from the previous segment.
    """
    lengths = np.asarray(lengths, dtype=float)
    k = len(lengths)
    counts = tuple(int(n) for n in counts)
    segment, index, last, lo, hi = _layout(counts)
    # np.linspace(0, length, count + 1) per column
    per_row = lengths.T[segment]                                       # (R, K): L of the row
    s = index * (lengths / np.array(counts, dtype=float)).T[segment]
    s[last] = lengths.T
    sign = np.array(signs)[segment]
    coef = (sign * kappa)[:, None]                                     # sign * kappa per row
    if ramp <= 0.0:
        curv = np.repeat(coef, k, axis=1)
    else:
        curv = coef * np.minimum(1.0, np.minimum(s, per_row - s) / ramp)
    curv[sign == 0] = 0.0
    step = s[hi] - s[lo]
    dtheta = 0.5 * (curv[lo] + curv[hi]) * step
    chord = step * _sinc(dtheta / (2.0 * np.pi))
    # Running sums into preallocated rows, then + the segment's start value.
    heading = np.empty((len(lo) + 1, k))          # row 0 the start, then one per step
    heading[0] = float(start.theta)
    first = 0
    for n in counts:
        rows = slice(first + 1, first + 1 + n)
        dtheta[first:first + n].cumsum(axis=0, out=heading[rows])
        heading[rows] += heading[first]           # + the previous segment's last heading
        first += n
    mid = heading[:-1] + 0.5 * dtheta
    moves = chord * np.array([np.cos(mid), np.sin(mid)])               # (2, steps, K)
    xy = np.empty((2, len(lo) + 1, k))
    xy[0, 0], xy[1, 0] = float(start.x), float(start.y)
    first = 0
    for n in counts:
        rows = slice(first + 1, first + 1 + n)
        moves[:, first:first + n].cumsum(axis=1, out=xy[:, rows])
        xy[:, rows] += xy[:, first:first + 1]     # + the previous segment's end point
        first += n
    return np.stack([xy[0].T, xy[1].T, heading.T], axis=-1)


def integrate(start: Pose2D, segments: Sequence[Segment], kappa: float,
              ramp: float, counts: Sequence[int]) -> np.ndarray:
    signs = [sign for sign, _ in segments]
    lengths = np.array([[length for _, length in segments]])
    return integrate_batch(start, signs, lengths, kappa, ramp, counts)[0]


def sample_counts(segments: Sequence[Segment], step: float) -> List[int]:
    return [max(2, int(math.ceil(length / step))) for _, length in segments]


def _ramped_guess(segments: Sequence[Segment], ramp: float) -> np.ndarray:
    lengths = []
    for sign, length in segments:
        if sign == 0:
            lengths.append(length)
        elif length >= ramp:
            lengths.append(length + ramp)
        else:
            lengths.append(2.0 * math.sqrt(ramp * length))
    return np.array(lengths)


def _end_errors(ends: np.ndarray, goal: Pose2D) -> np.ndarray:
    return np.column_stack([ends[:, 0] - goal.x, ends[:, 1] - goal.y,
                            wrap_angle(ends[:, 2] - goal.theta)])


def relax(start: Pose2D, goal: Pose2D, segments: Sequence[Segment],
          kappa: float, ramp: float, step: float,
          iterations: int = 12, tol: float = 1e-5) -> Optional[np.ndarray]:
    """Curvature-continuous version of a Dubins word, or None if not solvable.

    Peak curvature stays ``kappa``; curvature is zero at both ends and ramps
    linearly over ``ramp`` metres, so the result is G2 with straight neighbours.
    """
    signs = [sign for sign, _ in segments]
    lengths = _ramped_guess(segments, ramp)
    counts = sample_counts(list(zip(signs, lengths)), step)
    for _ in range(iterations):
        # Row 0 gives the residual, rows 1.. the finite-difference Jacobian columns.
        trials = np.vstack([lengths, lengths + _FD_STEP * np.eye(len(lengths))])
        paths = integrate_batch(start, signs, trials, kappa, ramp, counts)
        errors = _end_errors(paths[:, -1], goal)
        res = errors[0]
        if np.abs(res).max() < tol:
            path = paths[0]
            path[-1] = (goal.x, goal.y, path[-1, 2] - res[2])
            return path
        jac = (errors[1:] - res).T / _FD_STEP
        lengths = lengths + np.linalg.lstsq(jac, -res, rcond=None)[0]
        if lengths.min() < 0.0 or not np.all(np.isfinite(lengths)):
            return None
    return None
