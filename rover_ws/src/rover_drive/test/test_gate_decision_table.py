"""Gate decision table: E-stop x front obstacle x stale input/scan x pick approach.

``approach_active`` only switches the FRONT checks off; it must never let motion through when
E-stop is active or the input timed out. Single-condition rows live in test_safety / test_lidar."""
import pytest

from rover_drive.core.safety import OK, STOPPED, SafetyGate

from sim import gate_config

NOW = 2.0
NEAR = 0.30
FAR = 3.0

# (estop, front, scan_stale, input_stale, approach, expected_state, expected_reasons)
ROWS = [
    pytest.param(False, FAR, False, False, False, OK, (), id='clear'),
    pytest.param(False, NEAR, False, False, False, STOPPED, ('LIDAR_STOP',), id='obstacle'),
    pytest.param(False, FAR, True, False, False, STOPPED, ('STALE_SCAN',), id='scan-stale'),
    pytest.param(True, FAR, False, False, False, STOPPED, ('ESTOP',), id='estop'),
    pytest.param(True, NEAR, False, False, False, STOPPED, ('ESTOP', 'LIDAR_STOP'),
                 id='estop+obstacle'),
    pytest.param(False, FAR, False, True, False, STOPPED, ('INPUT_TIMEOUT',), id='input-stale'),
    # approach: front checks are off ...
    pytest.param(False, NEAR, False, False, True, OK, (), id='approach+obstacle'),
    pytest.param(False, FAR, True, False, True, OK, (), id='approach+scan-stale'),
    # ... and nothing else is
    pytest.param(True, FAR, False, False, True, STOPPED, ('ESTOP',), id='approach+estop'),
    pytest.param(True, NEAR, True, False, True, STOPPED, ('ESTOP',),
                 id='approach+estop+obstacle+scan-stale'),
    pytest.param(False, NEAR, False, True, True, STOPPED, ('INPUT_TIMEOUT',),
                 id='approach+input-stale'),
]


@pytest.mark.parametrize('estop,front,scan_stale,input_stale,approach,state,reasons', ROWS)
def test_gate_decision_table(estop, front, scan_stale, input_stale, approach, state, reasons):
    gate = SafetyGate(gate_config('leader', lidar_resume_s=0.0))
    gate.on_pose(1.0, 2.0, 0.0, NOW)
    gate.on_cmd(NOW)
    gate.set_aligned(True)
    gate.on_input(0.10, 0.0, 0.0 if input_stale else NOW)
    gate.on_front_distance(front, 0.0 if scan_stale else NOW)
    gate.on_approach(approach, NOW)
    gate.set_estop(estop)
    out = gate.step(NOW)
    assert (out.state, out.reasons) == (state, reasons)
    assert (out.v, out.w) == ((0.10, 0.0) if state == OK else (0.0, 0.0))
