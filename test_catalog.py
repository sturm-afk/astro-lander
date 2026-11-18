import math
import pytest
from landerlab import *
from catalog import *

MOON, ORB = BODIES["moon"], DEFAULT_ORBITS["moon"]
LM = LANDERS["apollo_lm"]


def scaled(L, k):
    return L.with_(dry_mass=L.dry_mass * k, fuel_mass=L.fuel_mass * k,
                   payload_mass=L.payload_mass * k, thrust_max=L.thrust_max * k)


@pytest.mark.parametrize("k", [0.01, 10.0, 1e4])
def test_scale_invariance(k):
    """Same mass ratio + same TWR => same outcome. Absolute size must not matter."""
    a = try_landing(MOON, LM, ORB)
    b = try_landing(MOON, scaled(LM, k), ORB)
    assert a.outcome == b.outcome == "LANDED"
    assert b.v_vert == pytest.approx(a.v_vert, rel=0.02)
    assert b.fuel_left / scaled(LM, k).fuel_mass == pytest.approx(a.fuel_left / LM.fuel_mass, rel=0.02)


def test_payload_reduces_leftover_fuel_until_failure():
    left, outcomes = [], []
    for p in (0, 500, 1000, 1500, 2000, 3000):
        r = try_landing(MOON, LM.with_(payload_mass=float(p)), ORB)
        outcomes.append(r.outcome)
        left.append(r.fuel_left)
    assert outcomes[0] == "LANDED" and outcomes[-1] == "CRASHED"
    landed_left = [l for l, o in zip(left, outcomes) if o == "LANDED"]
    assert landed_left == sorted(landed_left, reverse=True)


def test_cannot_beat_the_rocket_equation():
    dv0 = ideal_landing_dv(MOON, MOON.radius + ORB[0], MOON.radius + ORB[1])
    assert max_payload_sim(MOON, LM, ORB) < max_payload_ideal(LM.dry_mass, LM.fuel_mass, LM.isp, dv0)
    for L in LANDERS.values():
        assert min_fuel_sim(MOON, L, ORB) > fuel_needed_ideal(L.empty_mass, L.isp, dv0)


def test_sizing_roundtrip():
    # fuel_needed_ideal and delta_v are inverses
    L = LM.with_(fuel_mass=fuel_needed_ideal(LM.empty_mass, LM.isp, 1234.0))
    assert L.delta_v == pytest.approx(1234.0, rel=1e-9)
    # max payload at exactly the needed dv: payload comes back as the original
    assert max_payload_ideal(LM.dry_mass, L.fuel_mass, LM.isp, 1234.0) == pytest.approx(LM.payload_mass, abs=1e-6)


@pytest.mark.parametrize("key", ["satcube", "apollo_lm", "heavy_cargo", "enterprise"])
@pytest.mark.parametrize("world", ["moon", "ceres"])
def test_every_catalog_lander_lands_on_moon_and_ceres(key, world):
    r = try_landing(BODIES[world], LANDERS[key], DEFAULT_ORBITS[world])
    assert r.outcome == "LANDED", r.reason


def test_enterprise_resolves_comet_precisely():
    """Regression for the timestep bug: huge thrust authority vs. a tiny touchdown window."""
    r = try_landing(BODIES["67p"], LANDERS["enterprise"], DEFAULT_ORBITS["67p"])
    assert r.outcome == "LANDED"


def test_peak_g_matches_engine_and_mass():
    r = try_landing(MOON, LM, ORB)
    assert 0 < r.peak_g <= LM.thrust_max / LM.empty_mass / G0 + 1e-9


def test_hopeless_engine_flagged_by_calculator_and_sim():
    L = LM.with_(thrust_max=9_000.0)
    assert L.thrust_max < min_thrust_to_hover_empty(MOON, L)
    assert try_landing(MOON, L, ORB).outcome == "CRASHED"
