"""Localization quality rule shared by fusion_node and the diagnostics node.

GOOD = camera fresh (receive age); UWB only is LOST by default (15-18 cm class), or DEGRADED
via `uwb_only`. Modes (quality.mode): camera_age, camera_age_hold (held rover stays GOOD
through a camera gap), estimator (position sigma).
"""
from typing import NamedTuple, Optional

GOOD, DEGRADED, LOST = 'GOOD', 'DEGRADED', 'LOST'
CAMERA_AGE, CAMERA_AGE_HOLD, ESTIMATOR = 'camera_age', 'camera_age_hold', 'estimator'
MODES = (CAMERA_AGE, CAMERA_AGE_HOLD, ESTIMATOR)

BEV_TIMEOUT = 1.0    # [s] receive age within which BEV counts as present
UWB_TIMEOUT = 1.0    # [s] same for UWB
POSE_TIMEOUT = 1.0   # [s] no UWB and no BEV for this long -> no pose


def _fresh(age, timeout):
    return age is not None and age <= timeout


def classify(uwb_age, bev_age, uwb_timeout=UWB_TIMEOUT, bev_timeout=BEV_TIMEOUT,
             uwb_only=LOST):
    """Receive ages [s] (None = never) -> GOOD / DEGRADED / LOST."""
    if _fresh(bev_age, bev_timeout):
        return GOOD
    if _fresh(uwb_age, uwb_timeout):
        return uwb_only
    return LOST


def classify_held(uwb_age, bev_age, held, sigma, sigma_good, pose_timeout=POSE_TIMEOUT, **camera_age):
    """`classify`, but GOOD while held still, unmoved since the last camera fix, with sigma <=
    sigma_good [m] and a position measurement within pose_timeout."""
    q = classify(uwb_age, bev_age, **camera_age)
    if q == GOOD or not held or sigma > sigma_good:
        return q
    return GOOD if _fresh(position_age(uwb_age, bev_age), pose_timeout) else q


def position_age(uwb_age, bev_age):
    """Age of the newest position measurement (None = none ever)."""
    ages = [a for a in (uwb_age, bev_age) if a is not None]
    return min(ages) if ages else None


class EstimatorLimits(NamedTuple):
    """Thresholds of the estimator rule (quality.* parameters)."""

    sigma_good: float = 0.03        # [m] worst-axis 1-sigma for full speed
    sigma_lost: float = 0.06        # [m] beyond this it stops (between: DEGRADED = crawl)
    blind_time_degraded: float = 0.25  # [s] camera age, wheels not still -> at most DEGRADED
    blind_time_max: float = 0.5     # [s] ... -> LOST
    blind_travel_degraded: float = 0.04  # [m] wheel travel since the last camera fix -> DEGRADED
    blind_travel_max: float = 0.10  # [m] ... -> LOST
    refused_degraded: int = 5       # unused camera fixes in a row -> at most DEGRADED


class EstimatorState(NamedTuple):
    """Estimator snapshot (RobotFusion.estimator_state)."""
    in_camera_frame: bool           # a camera fix has placed the rover
    position_age: Optional[float]   # [s] newest UWB or camera fix (None = never)
    camera_age: Optional[float]     # [s] newest used camera fix
    sigma: float                    # [m] position 1-sigma, worst axis
    wheels_still: bool
    blind_travel: Optional[float]   # [m] since the newest used camera fix (None = unknown)
    camera_refused: int             # consecutive camera fixes not used


def classify_estimator(state, limits=EstimatorLimits(), pose_timeout=POSE_TIMEOUT):
    """GOOD / DEGRADED / LOST from the estimator (hard limits first); can flicker while driving."""
    if not state.in_camera_frame or not _fresh(state.position_age, pose_timeout):
        return LOST
    moving_blind = not state.wheels_still
    if moving_blind and not _fresh(state.camera_age, limits.blind_time_max):
        return LOST
    travel = 0.0 if state.blind_travel is None else state.blind_travel
    if travel > limits.blind_travel_max or state.sigma > limits.sigma_lost:
        return LOST
    if (state.sigma > limits.sigma_good or travel > limits.blind_travel_degraded
            or (moving_blind and not _fresh(state.camera_age, limits.blind_time_degraded))
            or state.camera_refused >= limits.refused_degraded):
        return DEGRADED
    return GOOD
