"""Leader omega truncation (steps of 0.0697 rad/s, cut toward zero)."""
import math

import numpy as np
import pytest

from rover_drive.core import limits as lim
from rover_drive.core import state as st

from sim import Rover, Sensor, Sim, drive_config, latched, make_cmd, quantize_omega, s_path

Q = 0.0697
MARGIN = 0.3
# The measured step is "about 0.0697": the real one may be slightly off either way.
TRUE_QUANTA = (0.0690, 0.0697, 0.0700)


def _executed(quantizer, omega, true_q, n=2000):
    """Omega the truncating base really runs for a constant request."""
    return [quantize_omega(quantizer.apply(omega), true_q, truncate=True) for _ in range(n)]


def test_plant_model_matches_the_measurement():
    """Commanded -> executed pairs measured on the leader."""
    for cmd, executed in ((0.02, 0.0), (0.06, 0.0), (0.08, Q), (0.10, Q), (0.14, 2 * Q),
                          (0.20, 2 * Q), (0.30, 4 * Q), (-0.10, -Q)):
        assert quantize_omega(cmd, Q, truncate=True) == pytest.approx(executed)


@pytest.mark.parametrize('true_q', TRUE_QUANTA)
@pytest.mark.parametrize('omega', [0.02, 0.05, -0.05, 0.10, 0.167, -0.30])
def test_truncate_mode_preserves_mean(true_q, omega):
    quant = lim.OmegaQuantizer(Q, lim.QUANTIZE_TRUNCATE, MARGIN, omega_max=0.6)
    mean = np.mean(_executed(quant, omega, true_q))
    assert mean == pytest.approx(omega * true_q / Q, abs=0.002)


def test_round_mode_loses_step_on_larger_real_step():
    """Why the old scheme was wrong: k * q sits on the edge of the cut."""
    quant = lim.OmegaQuantizer(Q, lim.QUANTIZE_ROUND)
    mean = np.mean(_executed(quant, 0.10, 0.0700))
    assert mean < 0.05


def test_truncate_mode_sends_steps_with_margin():
    quant = lim.OmegaQuantizer(Q, lim.QUANTIZE_TRUNCATE, MARGIN, omega_max=0.6)
    sent = [quant.apply(0.05) for _ in range(200)]
    assert set(np.round(sent, 6)) == {0.0, round((1 + MARGIN) * Q, 6)}
    assert quant.apply(0.0) in (0.0, (1 + MARGIN) * Q)
    negative = lim.OmegaQuantizer(Q, lim.QUANTIZE_TRUNCATE, MARGIN, omega_max=0.6)
    assert min(negative.apply(-0.3) for _ in range(50)) == pytest.approx(-(5 + MARGIN) * Q)


def test_sent_value_within_angular_limit():
    quant = lim.OmegaQuantizer(Q, lim.QUANTIZE_TRUNCATE, MARGIN, omega_max=0.6)
    sent = [quant.apply(5.0) for _ in range(100)]
    assert max(sent) <= 0.6
    assert max(sent) == pytest.approx((8 + MARGIN) * Q)
    assert abs(quant.apply(0.0)) <= 0.6


def test_unknown_quantize_mode_is_rejected():
    with pytest.raises(ValueError):
        lim.OmegaQuantizer(Q, 'floor')


def test_leader_yaml_selects_truncation():
    cfg = drive_config('leader')
    assert (cfg.omega_quantum, cfg.omega_quantize) == (Q, lim.QUANTIZE_TRUNCATE)
    assert drive_config('follower').omega_quantum == 0.0


def _s_run(true_q, seed=0, **cfg_overrides):
    cfg = drive_config('leader', **cfg_overrides)
    cmd = make_cmd(s_path(), goal_pos_tol=0.05, goal_yaw_tol=0.15, pick_at_goal=True)
    rover = Rover(omega_quantum=true_q, omega_truncate=true_q > 0.0)
    sim = Sim(cfg, cmd, rover, Sensor(delay=0.10, seed=seed))
    sim.run(180.0, until=latched)
    return sim


@pytest.mark.parametrize('true_q', TRUE_QUANTA)
def test_truncated_loop_matches_continuous_base(true_q):
    """S path: cross-track and goal error with the cut == without any quantization."""
    ideal = _s_run(0.0, omega_quantum=0.0)
    cut = _s_run(true_q)
    for sim in (ideal, cut):
        assert sim.out.state == st.HOLDING
    print(f'q_true={true_q}: lateral {cut.max_lateral():.3f} (ideal {ideal.max_lateral():.3f}) '
          f'goal {cut.goal_error()[0]:.3f} (ideal {ideal.goal_error()[0]:.3f})')
    assert cut.max_lateral() < ideal.max_lateral() + 0.005
    assert cut.goal_error()[0] < ideal.goal_error()[0] + 0.005
    assert cut.goal_error()[1] < ideal.goal_error()[1] + 0.02


def test_closed_loop_commands_whole_steps():
    sim = _s_run(Q)
    executed = [quantize_omega(row[2], Q, truncate=True) / Q for row in sim.log]
    sent = [row[2] / Q for row in sim.log]
    assert all(math.isclose(e, round(e), abs_tol=1e-9) for e in executed)
    assert all(s == 0.0 or math.isclose(abs(s) - math.floor(abs(s)), MARGIN, abs_tol=1e-6)
               for s in sent)
