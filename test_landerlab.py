"""Numerical invariants, reference flights, and edge-case regressions."""

from dataclasses import replace
import math
import subprocess
import sys

import pytest

import landerlab as lab


@pytest.mark.parametrize("name,altitude", [("moon", 100e3), ("ceres", 50e3), ("67p", 3e3)])
def test_circular_orbit_closure_energy_and_momentum(name, altitude):
    body = lab.BODIES[name]
    r = body.radius + altitude
    state = lab.State(r, 0, 0, body.circular_v(r), 0)
    lander = lab.Lander(1, 0, 0, 300)
    period = 2 * math.pi * body.dynamical_time(r)
    energy = lab.specific_energy(state, body)
    momentum = state.x * state.vy
    for _ in range(20_000):
        lab.advance(state, body, lander, 0, 1, 0, period / 2000)
    assert abs(math.hypot(state.x, state.y) / r - 1) < 1e-5
    assert abs(lab.specific_energy(state, body) / energy - 1) < 1e-6
    assert math.hypot(state.x - r, state.y) / r < 1e-3
    assert (state.x * state.vy - state.y * state.vx) / momentum == pytest.approx(1, rel=1e-9)


def test_elliptical_orbit_conserves_angular_momentum():
    body = lab.BODIES["moon"]
    lander = lab.Lander(1, 0, 0, 300)
    state = lab.state_at_periapsis(body, lander, body.radius + 100e3, body.radius + 15e3)
    momentum = state.x * state.vy
    for _ in range(40_000):
        lab.advance(state, body, lander, 0, 1, 0, 0.5)
    assert (state.x * state.vy - state.y * state.vx) / momentum == pytest.approx(1, rel=1e-9)


def test_short_lunar_coast_energy():
    body = lab.BODIES["moon"]
    state = lab.State(body.radius + 15e3, 0, 0, 1500, 0)
    lander = lab.Lander(1, 0, 0, 300)
    energy = lab.specific_energy(state, body)
    for _ in range(3000):
        lab.advance(state, body, lander, 0, 1, 0, 0.1)
    assert lab.specific_energy(state, body) / energy == pytest.approx(1, rel=1e-9)


def test_pseudo_newtonian_orbit_energy():
    body = lab.BODIES["bh10"]
    r = 10 * body.gravity.rs
    speed = body.circular_v(r)
    period = 2 * math.pi * r / speed
    state = lab.State(r, 0, 0, speed, 0)
    lander = lab.Lander(1, 0, 0, 300)
    energy = lab.specific_energy(state, body)
    for _ in range(20_000):
        lab.advance(state, body, lander, 0, 1, 0, period / 4000)
    assert lab.specific_energy(state, body) / energy == pytest.approx(1, rel=1e-6)


def test_rocket_equation_and_inverse():
    lander = lab.Lander(1000, 1000, 20_000, 300)
    expected = 300 * lab.G0 * math.log(2)
    assert lander.delta_v == pytest.approx(expected)
    assert lab.propellant_for_dv(lander.isp, lander.wet_mass, expected) == pytest.approx(1000)
    body = lab.Body("deep space", 1e-9, 1)
    state = lab.State(1e7, 0, 0, 0, 1000)
    while state.fuel:
        lab.advance(state, body, lander, 1, 1, 0, 0.05)
    assert state.vx == pytest.approx(expected, rel=1e-4)


def test_burnout_splits_burn_and_coast():
    body = lab.Body("deep space", 1e-9, 1)
    lander = lab.Lander(1000, 1, 20_000, 300)
    combined = lab.State(1e7, 0, 0, 0, 1)
    split = replace(combined)
    burn_time = lander.fuel_mass * lander.isp * lab.G0 / lander.thrust_max
    lab.advance(combined, body, lander, 1, 1, 0, 10)
    lab.advance(split, body, lander, 1, 1, 0, burn_time)
    lab.advance(split, body, lander, 0, 1, 0, 10 - burn_time)
    assert combined.x == pytest.approx(split.x, abs=1e-8)
    assert combined.vx == pytest.approx(split.vx, abs=1e-10)
    assert combined.fuel == 0
    # A full-step average thrust would move only half this distance.
    assert combined.x - 1e7 > 0.9 * combined.vx * 10


@pytest.mark.parametrize("index", range(len(lab.SCENARIOS)))
def test_reference_landing(index):
    scenario = lab.SCENARIOS[index]
    calc, result = lab.run_scenario(scenario)
    crit = lab.Criteria.for_body(scenario.body)
    assert result.outcome == "LANDED"
    assert 0 < result.v_vert <= crit.max_vertical
    assert result.v_horiz <= crit.max_horizontal
    assert result.gravity_loss > 0
    assert calc["dv_ideal"] == result.dv_ideal
    assert result.fuel_left >= 0


@pytest.mark.parametrize("thrust,outcome", [(20_000, "LANDED"), (9000, "CRASHED")])
def test_lunar_low_initial_twr(thrust, outcome):
    scenario = lab.SCENARIOS[0]
    scenario = replace(scenario, lander=replace(scenario.lander, thrust_max=thrust))
    assert lab.thrust_to_weight(scenario.body, scenario.lander) < 1
    _, result = lab.run_scenario(scenario)
    assert result.outcome == outcome


def test_low_fuel_crashes():
    scenario = lab.SCENARIOS[0]
    scenario = replace(scenario, lander=replace(scenario.lander, fuel_mass=2000))
    _, result = lab.run_scenario(scenario)
    assert result.outcome == "CRASHED"
    assert result.fuel_left == 0


def test_vertical_drop():
    body = lab.BODIES["moon"]
    lander = lab.Lander(2000, 1000, 10_000, 311)
    initial = lab.State(body.radius + 3000, 0, 0, 0, 1000)
    result = lab.simulate_landing(body, lander, initial, lab.Guidance(tau_v=1.5))
    assert result.outcome == "LANDED"
    assert result.dv_used < math.sqrt(2 * body.surface_g * 3000) * 1.6
    assert initial == lab.State(body.radius + 3000, 0, 0, 0, 1000)


def test_calculators():
    body = lab.BODIES["moon"]
    ra, rp = body.radius + 100e3, body.radius + 15e3
    assert lab.periapsis_speed(body, rp, rp) == pytest.approx(body.circular_v(rp))
    assert lab.deorbit_dv(body, rp, rp) == pytest.approx(0)
    assert lab.ideal_landing_dv(body, ra, rp) == lab.periapsis_speed(body, ra, rp)
    assert lab.deorbit_dv(body, ra, rp) > 0
    assert lab.stopping_distance(10, 4, 2) == 25
    assert math.isinf(lab.stopping_distance(10, 2, 2))
    assert math.isinf(lab.stopping_distance(10, 1, 2))


def test_nonlandable_rejected_before_surface_criteria():
    body = lab.BODIES["bh10"]
    state = lab.State(10 * body.radius, 0, 0, 0, 1)
    with pytest.raises(ValueError, match="landable"):
        lab.simulate_landing(body, lab.Lander(1, 1, 1, 1), state)


@pytest.mark.parametrize("factor", [0, 0.5, 1])
def test_horizon_rejected_by_potential_and_acceleration(factor):
    gravity = lab.PseudoNewtonian(10)
    with pytest.raises(lab.HorizonCrossed):
        gravity.potential(10 * factor, 100)
    with pytest.raises(lab.HorizonCrossed):
        gravity.accel(10 * factor, 0, 100)


@pytest.mark.parametrize("calculator", [lab.periapsis_speed, lab.deorbit_dv, lab.ideal_landing_dv])
def test_newtonian_formulas_reject_pn(calculator):
    body = lab.BODIES["bh10"]
    with pytest.raises(ValueError, match="PointMass"):
        calculator(body, 20 * body.radius, 10 * body.radius)


@pytest.mark.parametrize("dt", [0, -1, math.nan, math.inf])
def test_invalid_timesteps(dt):
    scenario = lab.SCENARIOS[0]
    state = lab.state_at_periapsis(scenario.body, scenario.lander,
                                  scenario.body.radius + 100e3, scenario.body.radius + 15e3)
    with pytest.raises(ValueError):
        lab.advance(state, scenario.body, scenario.lander, 0, 1, 0, dt)
    with pytest.raises(ValueError):
        lab.simulate_landing(scenario.body, scenario.lander, state, dt_max=dt)


@pytest.mark.parametrize("factory", [
    lambda: lab.Body("invalid", 0, 1),
    lambda: lab.Body("invalid", 1, -1),
    lambda: lab.Lander(0, 1, 1, 1),
    lambda: lab.Lander(1, -1, 1, 1),
    lambda: lab.Lander(1, 1, 1, 0),
    lambda: lab.Lander(1, 1, 1, 1, 2),
    lambda: lab.State(1, 0, 0, math.nan, 1),
    lambda: lab.State(1, 0, 0, 0, -1),
    lambda: lab.State(0, 0, 0, 0, 1),
    lambda: lab.Guidance(tau_v=0),
    lambda: lab.Guidance(horiz_share=0),
    lambda: lab.Criteria(-1, 1),
    lambda: lab.tsiolkovsky_dv(300, 100, 200),
    lambda: lab.propellant_for_dv(300, 100, -1),
])
def test_invalid_core_inputs(factory):
    with pytest.raises(ValueError):
        factory()


def test_timeout_and_log():
    scenario = lab.SCENARIOS[0]
    state = lab.state_at_periapsis(scenario.body, scenario.lander,
                                  scenario.body.radius + 100e3, scenario.body.radius + 15e3)
    result = lab.simulate_landing(scenario.body, scenario.lander, state,
                                  t_max=1, dt_max=0.1, log_every=0.2)
    assert result.outcome == "TIMEOUT"
    assert result.t == pytest.approx(1)
    assert len(result.log) >= 4
    assert len(result.log[0]) == 5
    assert result.log[0] == (0, 15e3, 0, state.vy, state.fuel)
    for keyword in ("t_max", "log_every"):
        with pytest.raises(ValueError):
            lab.simulate_landing(scenario.body, scenario.lander, state, **{keyword: 0})


def test_cli():
    result = subprocess.run([sys.executable, "landerlab.py"],
                            text=True, capture_output=True, check=True)
    assert "LANDED" in result.stdout
    assert "Moon" in result.stdout
    assert "Analytic feasibility" in result.stdout
    assert "Simulated landing" in result.stdout


def test_public_api():
    body = lab.BODIES["moon"]
    lander = lab.Lander(7000, 8200, 45000, 311, 0.1)
    ra, rp = body.radius + 100e3, body.radius + 15e3
    state = lab.state_at_periapsis(body, lander, ra, rp)
    assert state.x == rp
    assert state.vy == lab.periapsis_speed(body, ra, rp)
    assert state.fuel == lander.fuel_mass
    assert lander.mdot_max == lander.thrust_max / (lander.isp * lab.G0)
    result = lab.Result("TIMEOUT", "test", 0, 0, 0, 1, 0, 0, 0)
    assert result.log == []
    calc, result = lab.run_scenario(lab.SCENARIOS[0], dt_max=0.1, t_max=0.2)
    assert result.outcome == "TIMEOUT"
    assert set(calc) == {"surface_g", "v_escape", "v_peri", "dv_deorbit",
                         "dv_ideal", "dv_avail", "twr", "limits"}


def test_custom_gravity_strategy():
    class Harmonic:
        def accel(self, x, y, GM):
            return -GM * x, -GM * y

        def potential(self, r, GM):
            return 0.5 * GM * r**2

    body = lab.Body("harmonic toy", 2, 1, Harmonic())
    assert body.g_at(3) == 6
    state = lab.State(3, 0, 0, 0, 0)
    lab.advance(state, body, lab.Lander(1, 0, 0, 300), 0, 1, 0, 0.01)
    assert state.vx < 0
    with pytest.raises(ValueError, match="PointMass"):
        lab.periapsis_speed(body, 3, 2)
