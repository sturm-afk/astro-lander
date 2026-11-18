"""
catalog.py -- lander designs, sizing calculators, and comparison tables.

Sits on top of landerlab.py. Numbers marked ILLUSTRATIVE are round-number
stand-ins to exercise the physics at very different scales; they are NOT
official specifications.
"""
from __future__ import annotations

import math
from typing import Optional

from landerlab import (
    BODIES, G0, Body, Criteria, Guidance, Lander, Result, State,
    ideal_landing_dv, periapsis_speed, simulate_landing, state_at_periapsis,
    thrust_to_weight, tsiolkovsky_dv,
)

# --------------------------------------------------------------------------
# Catalog (data, not code: later this becomes JSON/TOML for modding)
# --------------------------------------------------------------------------
LANDERS = {
    "satcube": Lander(
        name="SatCube lander (fictional, ~14 kg)",
        dry_mass=8.0, payload_mass=2.0, fuel_mass=14.0,
        thrust_max=60.0, isp=230.0, min_throttle=0.10),
    "apollo_lm": Lander(
        name="Apollo LM-like (approx. PDI mass)",
        dry_mass=7000.0, payload_mass=0.0, fuel_mass=8200.0,
        thrust_max=45_000.0, isp=311.0, min_throttle=0.10),
    "heavy_cargo": Lander(
        name="Heavy cargo lander (ILLUSTRATIVE, ~Starship-HLS class)",
        dry_mass=120_000.0, payload_mass=100_000.0, fuel_mass=250_000.0,
        thrust_max=3_000_000.0, isp=380.0, min_throttle=0.10),
    "enterprise": Lander(
        name="Starship Enterprise (FICTIONAL, ~4.2 Mt, fusion torch)",
        dry_mass=4.2e9, payload_mass=0.0, fuel_mass=1.0e6,
        thrust_max=1.4e10, isp=5.0e6, min_throttle=0.0),
}

# Default parking orbits (altitudes above the surface, apo x peri)
DEFAULT_ORBITS = {
    "moon":  (100e3, 15e3),
    "ceres": (100e3, 10e3),
    "67p":   (7e3, 5e3),
}


def auto_guidance(body: Body, lander: Lander, crit: Optional[Criteria] = None) -> Guidance:
    """Response times scaled to what the engine can do, so one controller spans 12 orders of magnitude."""
    crit = crit or Criteria.for_body(body)
    v_touch = 0.5 * crit.max_vertical
    a_dry = lander.thrust_max / lander.empty_mass
    tau = max(1.0, 3.0 * v_touch / a_dry)
    return Guidance(tau_v=tau, tau_h=2.0 * tau)


def try_landing(body: Body, lander: Lander, orbit: tuple[float, float],
                guidance: Optional[Guidance] = None, **kw) -> Result:
    r_apo, r_peri = body.radius + orbit[0], body.radius + orbit[1]
    s0 = state_at_periapsis(body, lander, r_apo, r_peri)
    return simulate_landing(body, lander, s0, guidance or auto_guidance(body, lander), **kw)


# --------------------------------------------------------------------------
# Closed-form sizing (the quick calculator)
# --------------------------------------------------------------------------
def fuel_needed_ideal(empty_mass: float, isp: float, dv: float) -> float:
    """Propellant to give an empty_mass vehicle `dv`. Rocket equation solved for fuel."""
    return empty_mass * (math.exp(dv / (isp * G0)) - 1.0)


def max_payload_ideal(dry_mass: float, fuel_mass: float, isp: float, dv: float) -> float:
    """Largest payload for which fixed fuel still delivers `dv`. Negative => structure alone can't."""
    ratio = math.exp(dv / (isp * G0))
    return fuel_mass / (ratio - 1.0) - dry_mass


def min_thrust_to_hover_empty(body: Body, lander: Lander) -> float:
    """Below this thrust you can never stop: weight at landing mass exceeds the engine."""
    return lander.empty_mass * body.surface_g


# --------------------------------------------------------------------------
# Simulation-based sizing (includes gravity losses, throttle limits, guidance)
# --------------------------------------------------------------------------
def _lands(body, lander, orbit) -> bool:
    return try_landing(body, lander, orbit).outcome == "LANDED"


def min_fuel_sim(body: Body, lander: Lander, orbit, rel_tol: float = 0.005) -> Optional[float]:
    """Smallest fuel mass (same structure, payload, engine) that still lands. Bisection.
    Assumes 'more fuel lands' holds across the bracket; returns None if no bracket lands."""
    dv0 = ideal_landing_dv(body, body.radius + orbit[0], body.radius + orbit[1])
    hi = fuel_needed_ideal(lander.empty_mass, lander.isp, 1.8 * dv0)
    for _ in range(4):
        if _lands(body, lander.with_(fuel_mass=hi), orbit):
            break
        hi *= 2.0
    else:
        return None
    lo = 0.0
    while (hi - lo) > rel_tol * hi:
        mid = 0.5 * (lo + hi)
        if _lands(body, lander.with_(fuel_mass=mid), orbit):
            hi = mid
        else:
            lo = mid
    return hi


def max_payload_sim(body: Body, lander: Lander, orbit, rel_tol: float = 0.005) -> Optional[float]:
    """Largest payload that still lands with the lander's fixed fuel. None if even zero payload fails."""
    base = lander.with_(payload_mass=0.0)
    if not _lands(body, base, orbit):
        return None
    dv0 = ideal_landing_dv(body, body.radius + orbit[0], body.radius + orbit[1])
    hi = max(max_payload_ideal(lander.dry_mass, lander.fuel_mass, lander.isp, dv0), 0.0) * 1.5 + 1e-9
    while _lands(body, base.with_(payload_mass=hi), orbit) and hi < 1e3 * lander.wet_mass:
        hi *= 2.0
    lo = 0.0
    while (hi - lo) > rel_tol * hi:
        mid = 0.5 * (lo + hi)
        if _lands(body, base.with_(payload_mass=mid), orbit):
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def _fmt_mass(kg: float) -> str:
    for unit, scale in (("Mt", 1e9), ("t", 1e3), ("kg", 1.0)):
        if abs(kg) >= scale:
            return f"{kg / scale:.3g} {unit}"
    return f"{kg:.3g} kg"


def table_catalog() -> None:
    print("LANDER CATALOG (what each design can do on the Moon, closed form)")
    print(f"{'lander':58} {'m_wet':>9} {'m_empty':>9} {'dv_avail':>9} {'TWR_wet':>8} {'TWR_empty':>9}")
    moon = BODIES["moon"]
    for L in LANDERS.values():
        twr_empty = L.thrust_max / (L.empty_mass * moon.surface_g)
        print(f"{L.name:58} {_fmt_mass(L.wet_mass):>9} {_fmt_mass(L.empty_mass):>9} "
              f"{L.delta_v:9.0f} {thrust_to_weight(moon, L):8.2f} {twr_empty:9.2f}")


def table_matrix() -> None:
    print("\nEVERY LANDER ON EVERY WORLD (default parking orbit, auto-tuned guidance)")
    print(f"{'lander':16} {'world':10} {'result':8} {'v_vert':>7} {'limit':>6} {'fuel_left':>10} "
          f"{'peak g':>7} {'g-loss dv':>9}  note")
    for key, L in LANDERS.items():
        for bkey, orbit in DEFAULT_ORBITS.items():
            b = BODIES[bkey]
            res = try_landing(b, L, orbit)
            lim = Criteria.for_body(b).max_vertical
            note = "" if res.outcome == "LANDED" else res.reason
            print(f"{key:16} {bkey:10} {res.outcome:8} {res.v_vert:7.2f} {lim:6.2f} "
                  f"{_fmt_mass(res.fuel_left):>10} {res.peak_g:7.2f} {res.gravity_loss:9.0f}  {note}")


def table_payload(lander_key: str = "apollo_lm", bkey: str = "moon") -> None:
    L, b, orbit = LANDERS[lander_key], BODIES[bkey], DEFAULT_ORBITS[bkey]
    dv0 = ideal_landing_dv(b, b.radius + orbit[0], b.radius + orbit[1])
    print(f"\nPAYLOAD SWEEP: {L.name} on {b.name} (fixed {_fmt_mass(L.fuel_mass)} of fuel)")
    print(f"{'payload':>9} {'m_wet':>9} {'dv_avail':>9} {'TWR_start':>9} {'TWR_end':>8} {'result':8} "
          f"{'v_vert':>7} {'fuel_left':>10}")
    for p in (0, 500, 1000, 1500, 2000, 2500):
        Lp = L.with_(payload_mass=float(p))
        res = try_landing(b, Lp, orbit)
        twr_end = Lp.thrust_max / (Lp.empty_mass * b.surface_g)
        print(f"{_fmt_mass(p):>9} {_fmt_mass(Lp.wet_mass):>9} {Lp.delta_v:9.0f} "
              f"{thrust_to_weight(b, Lp):9.2f} {twr_end:8.2f} {res.outcome:8} "
              f"{res.v_vert:7.2f} {_fmt_mass(res.fuel_left):>10}")
    ideal = max_payload_ideal(L.dry_mass, L.fuel_mass, L.isp, dv0)
    sim = max_payload_sim(b, L, orbit)
    print(f"  max payload, rocket-equation estimate (ideal dv {dv0:.0f} m/s): {_fmt_mass(ideal)}")
    print(f"  max payload, simulated (gravity losses, throttle, guidance):    "
          f"{_fmt_mass(sim) if sim is not None else 'none'}")


def table_fuel_sizing() -> None:
    print("\nHOW MUCH FUEL DOES EACH DESIGN NEED ON THE MOON? (structure, payload, engine fixed)")
    print(f"{'lander':16} {'carries':>10} {'ideal':>10} {'simulated':>10} {'penalty':>8}")
    b, orbit = BODIES["moon"], DEFAULT_ORBITS["moon"]
    dv0 = ideal_landing_dv(b, b.radius + orbit[0], b.radius + orbit[1])
    for key, L in LANDERS.items():
        ideal = fuel_needed_ideal(L.empty_mass, L.isp, dv0)
        sim = min_fuel_sim(b, L, orbit)
        pen = f"{100 * (sim / ideal - 1):6.1f}%" if sim else "    n/a"
        print(f"{key:16} {_fmt_mass(L.fuel_mass):>10} {_fmt_mass(ideal):>10} "
              f"{_fmt_mass(sim) if sim else 'none':>10} {pen:>8}")


if __name__ == "__main__":
    table_catalog()
    table_matrix()
    table_payload()
    table_fuel_sizing()
