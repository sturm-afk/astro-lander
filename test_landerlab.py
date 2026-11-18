import math
import pytest
from landerlab import *

MOON = BODIES["moon"]


def coast(body, s, t_end, dt):
    L = Lander(1.0, 0.0, 1.0, 300.0)       # irrelevant: throttle is always 0
    while s.t < t_end - 1e-9:
        advance(s, body, L, 0.0, 1.0, 0.0, min(dt, t_end - s.t))


# ---- body sanity ---------------------------------------------------------
def test_moon_numbers():
    assert MOON.surface_g == pytest.approx(1.62, abs=0.01)
    assert MOON.escape_v == pytest.approx(2376, abs=5)


def test_comet_is_tiny():
    c = BODIES["67p"]
    assert c.escape_v < 1.0 and c.surface_g < 1e-3


# ---- integrator: orbits must stay orbits ---------------------------------
@pytest.mark.parametrize("name,alt", [("moon", 100e3), ("ceres", 50e3), ("67p", 3e3)])
def test_circular_orbit_closes_and_conserves_energy(name, alt):
    b = BODIES[name]
    r = b.radius + alt
    T = 2 * math.pi * b.dynamical_time(r)
    s = State(r, 0.0, 0.0, b.circular_v(r), 0.0)
    e0 = specific_energy(s, b)
    dt = T / 2000
    coast(b, s, 10 * T, dt)                          # ten full orbits
    assert abs(math.hypot(s.x, s.y) - r) / r < 1e-5
    assert abs(specific_energy(s, b) - e0) / abs(e0) < 1e-6
    # phase: back near the start after whole periods
    assert math.hypot(s.x - r, s.y) / r < 1e-3


def test_angular_momentum_conserved_elliptical():
    b = MOON
    r_p, r_a = b.radius + 15e3, b.radius + 100e3
    s = State(r_p, 0.0, 0.0, periapsis_speed(b, r_a, r_p), 0.0)
    h0 = s.x * s.vy - s.y * s.vx
    coast(b, s, 20000, 0.5)
    h1 = s.x * s.vy - s.y * s.vx
    assert abs(h1 - h0) / h0 < 1e-9


def test_pseudo_newtonian_energy_conserved():
    b = BODIES["bh10"]
    rs = b.gravity.rs
    r = 10 * rs
    v = b.circular_v(r)
    s = State(r, 0.0, 0.0, v, 0.0)
    e0 = specific_energy(s, b)
    T = 2 * math.pi * r / v
    coast(b, s, 5 * T, T / 4000)
    assert abs(specific_energy(s, b) - e0) / abs(e0) < 1e-6


def test_pseudo_newtonian_horizon():
    pn = PseudoNewtonian(rs=100.0)
    with pytest.raises(HorizonCrossed):
        pn.accel(99.0, 0.0, 1e10)


# ---- rocket: mass and thrust bookkeeping ---------------------------------
def test_burn_matches_tsiolkovsky_in_free_space():
    deep = Body("deep space", 1e-9, 1.0)             # effectively no gravity
    L = Lander(1000.0, 1000.0, 20_000.0, 300.0)
    s = State(1e7, 0.0, 0.0, 0.0, L.fuel_mass)
    dt = 0.05
    while s.fuel > 0:
        advance(s, deep, L, 1.0, 1.0, 0.0, dt)
    assert math.hypot(s.vx, s.vy) == pytest.approx(L.delta_v, rel=1e-4)
    assert s.fuel == pytest.approx(0.0, abs=1e-9)


def test_player_control_steers_from_current_velocity():
    s = State(10.0, 0.0, 3.0, 4.0, 1.0)
    assert PlayerControl(0.5).command(s) == pytest.approx((0.5, 0.6, 0.8))
    assert PlayerControl(1.0, math.pi).command(s) == pytest.approx((1.0, -0.6, -0.8))


def test_player_control_falls_back_to_local_tangent_at_rest():
    s = State(0.0, 10.0, 0.0, 0.0, 1.0)
    assert PlayerControl().command(s) == pytest.approx((0.0, -1.0, 0.0))


def test_advance_controlled_applies_throttle_and_burns_fuel():
    deep = Body("deep space", 1e-9, 1.0)
    L = Lander(100.0, 10.0, 100.0, 300.0)
    s = State(1e7, 0.0, 0.0, 10.0, L.fuel_mass)
    advance_controlled(s, deep, L, PlayerControl(0.5), 1.0)
    assert s.vy > 10.0
    assert s.fuel == pytest.approx(10.0 - 0.5 * L.mdot_max)
    assert s.t == 1.0


def test_simulate_player_landing_requests_fresh_controls_and_times_out():
    deep = Body("deep space", 1e-9, 1.0)
    L = Lander(100.0, 10.0, 100.0, 300.0)
    s0 = State(1e7, 0.0, 0.0, 10.0, L.fuel_mass)
    observed = []

    def control_source(state):
        observed.append((state.t, state.vy))
        return PlayerControl(0.5 if state.t == 0.0 else 0.25)

    result = simulate_player_landing(deep, L, s0, control_source, dt=1.0, t_max=2.0)

    assert result.outcome == "TIMEOUT"
    assert result.steps == 2
    assert [t for t, _ in observed] == [0.0, 1.0]
    assert observed[1][1] > observed[0][1]
    assert result.fuel_left == pytest.approx(10.0 - 0.75 * L.mdot_max)
    assert s0.t == 0.0 and s0.fuel == L.fuel_mass


@pytest.mark.parametrize(("vx", "outcome"), [(0.0, "LANDED"), (-10.0, "CRASHED")])
def test_simulate_player_landing_reports_touchdown_outcome(vx, outcome):
    s0 = State(MOON.radius + 0.001, 0.0, vx, 0.0, 0.0)
    L = Lander(100.0, 0.0, 100.0, 300.0)

    result = simulate_player_landing(
        MOON, L, s0, lambda state: PlayerControl(), dt=0.1, t_max=1.0,
    )

    assert result.outcome == outcome
    assert result.steps == 1


@pytest.mark.parametrize("dt", [0.0, -0.1, math.inf, math.nan])
def test_simulate_player_landing_rejects_invalid_timestep(dt):
    with pytest.raises(ValueError, match="dt"):
        simulate_player_landing(MOON, Lander(1.0, 0.0, 1.0, 300.0),
                                State(MOON.radius + 1.0, 0.0, 0.0, 0.0, 0.0),
                                lambda state: PlayerControl(), dt=dt)


@pytest.mark.parametrize("throttle", [-0.1, 1.1, math.inf, math.nan])
def test_player_control_rejects_invalid_throttle(throttle):
    with pytest.raises(ValueError):
        PlayerControl(throttle)


def test_player_control_rejects_nonfinite_angle():
    with pytest.raises(ValueError):
        PlayerControl(angle=math.inf)


def test_player_control_requires_position_when_at_rest():
    with pytest.raises(ValueError, match="body's center"):
        PlayerControl().command(State(0.0, 0.0, 0.0, 0.0, 1.0))


def test_calculators():
    assert tsiolkovsky_dv(300, 2000, 1000) == pytest.approx(300 * G0 * math.log(2))
    assert propellant_for_dv(300, 2000, tsiolkovsky_dv(300, 2000, 1000)) == pytest.approx(1000)
    # circular -> deorbit burn is small and positive
    assert 0 < deorbit_dv(MOON, MOON.radius + 100e3, MOON.radius + 15e3) < 100


# ---- landing outcomes -----------------------------------------------------
@pytest.mark.parametrize("i", range(len(SCENARIOS)))
def test_reference_scenarios_land(i):
    _, res = run_scenario(SCENARIOS[i])
    assert res.outcome == "LANDED", res.reason
    assert res.gravity_loss > 0            # you can't beat the ideal Delta-v


def test_moon_without_enough_fuel_crashes():
    sc = SCENARIOS[0]
    weak = Scenario(sc.name, sc.body, Lander(7000, 2000, 45_000, 311, 0.10),
                    sc.r_apo_alt, sc.r_peri_alt, sc.guidance)
    assert weak.lander.delta_v < ideal_landing_dv(weak.body, weak.body.radius + weak.r_apo_alt,
                                                  weak.body.radius + weak.r_peri_alt)
    _, res = run_scenario(weak)
    assert res.outcome == "CRASHED"


def test_twr_at_full_load_below_one_can_still_land():
    """Lighter as fuel burns + centrifugal relief in orbit: full-load TWR<1 is survivable."""
    sc = SCENARIOS[0]
    L = Lander(7000, 8200, 20_000, 311, 0.10)        # TWR ~0.8 at PDI
    assert thrust_to_weight(sc.body, L) < 1.0
    _, res = run_scenario(Scenario("weak engine", sc.body, L, sc.r_apo_alt, sc.r_peri_alt, sc.guidance))
    assert res.outcome == "LANDED"


def test_cannot_land_if_thrust_below_weight_at_dry_mass():
    """The real criterion: thrust vs. weight at landing mass. Below it, you can never stop."""
    sc = SCENARIOS[0]
    L = Lander(7000, 8200, 9_000, 311, 0.10)         # 9 kN < 7000 kg * 1.62
    assert L.thrust_max < L.dry_mass * sc.body.surface_g
    _, res = run_scenario(Scenario("hopeless engine", sc.body, L, sc.r_apo_alt, sc.r_peri_alt, sc.guidance))
    assert res.outcome == "CRASHED"


def test_vertical_drop_matches_stopping_distance():
    """Pure vertical case: the suicide burn should land using ~the analytic Delta-v."""
    b = MOON
    L = Lander(2000.0, 1000.0, 10_000.0, 311.0, 0.0)
    s0 = State(b.radius + 3000.0, 0.0, 0.0, 0.0, L.fuel_mass)
    res = simulate_landing(b, L, s0, Guidance(tau_v=1.5))
    assert res.outcome == "LANDED"
    # free-fall speed at ~(3000 m - braking) bounds the Delta-v from above
    v_fall = math.sqrt(2 * b.surface_g * 3000)
    assert res.dv_used < v_fall * 1.6


def test_energy_never_increases_without_thrust():
    b = MOON
    s = State(b.radius + 15e3, 0.0, 0.0, 1500.0, 0.0)
    e0 = specific_energy(s, b)
    coast(b, s, 300.0, 0.1)
    assert abs(specific_energy(s, b) - e0) / abs(e0) < 1e-9
