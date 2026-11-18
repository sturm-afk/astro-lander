#!/usr/bin/env python3
"""A standard-library-only, planar, airless landing laboratory. Units are SI."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Protocol

G0 = 9.80665
C_LIGHT = 299_792_458.0


def _positive(name: str, value: float, zero: bool = False) -> None:
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


class HorizonCrossed(ValueError):
    """The pseudo-Newtonian horizon was reached."""


class GravityModel(Protocol):
    def accel(self, x: float, y: float, GM: float) -> tuple[float, float]: ...

    def potential(self, r: float, GM: float) -> float: ...


@dataclass(frozen=True)
class PointMass:
    def accel(self, x: float, y: float, GM: float) -> tuple[float, float]:
        r = math.hypot(x, y)
        _positive("radius", r)
        return -GM * x / r**3, -GM * y / r**3

    def potential(self, r: float, GM: float) -> float:
        _positive("radius", r)
        return -GM / r


@dataclass(frozen=True)
class PseudoNewtonian:
    """Paczyński–Wiita toy potential, not a general-relativistic integrator."""

    rs: float

    def __post_init__(self) -> None:
        _positive("rs", self.rs)

    def _check(self, r: float) -> None:
        if not math.isfinite(r):
            raise ValueError("radius must be finite")
        if r <= self.rs:
            raise HorizonCrossed("at or inside the Schwarzschild radius")

    def accel(self, x: float, y: float, GM: float) -> tuple[float, float]:
        r = math.hypot(x, y)
        self._check(r)
        scale = -GM / ((r - self.rs)**2 * r)
        return scale * x, scale * y

    def potential(self, r: float, GM: float) -> float:
        self._check(r)
        return -GM / (r - self.rs)


@dataclass(frozen=True)
class Body:
    name: str
    GM: float
    radius: float
    gravity: GravityModel = field(default_factory=PointMass)
    landable: bool = True

    def __post_init__(self) -> None:
        _positive("GM", self.GM)
        _positive("radius", self.radius)

    def g_at(self, r: float) -> float:
        return -self.gravity.accel(r, 0, self.GM)[0]

    @property
    def surface_g(self) -> float:
        return self.g_at(self.radius)

    @property
    def escape_v(self) -> float:
        return math.sqrt(-2 * self.gravity.potential(self.radius, self.GM))

    def circular_v(self, r: float) -> float:
        return math.sqrt(r * self.g_at(r))

    def dynamical_time(self, r: float) -> float:
        _positive("radius", r)
        return math.sqrt(r**3 / self.GM)


_BH_GM = 10 * 1.32712440018e20
_BH_RS = 2 * _BH_GM / C_LIGHT**2
BODIES = {
    "moon": Body("Moon", 4.9048695e12, 1_737_400),
    "ceres": Body("Ceres", 6.2628e10, 469_700),
    "67p": Body("67P", 666.2, 2_000),
    "bh10": Body("10 solar-mass black hole", _BH_GM, _BH_RS,
                 PseudoNewtonian(_BH_RS), landable=False),
}


@dataclass(frozen=True)
class Lander:
    dry_mass: float
    fuel_mass: float
    thrust_max: float
    isp: float
    min_throttle: float = 0.0

    def __post_init__(self) -> None:
        _positive("dry_mass", self.dry_mass)
        _positive("fuel_mass", self.fuel_mass, True)
        _positive("thrust_max", self.thrust_max, True)
        _positive("isp", self.isp)
        if not math.isfinite(self.min_throttle) or not 0 <= self.min_throttle <= 1:
            raise ValueError("min_throttle must be between zero and one")

    @property
    def wet_mass(self) -> float:
        return self.dry_mass + self.fuel_mass

    @property
    def delta_v(self) -> float:
        return tsiolkovsky_dv(self.isp, self.wet_mass, self.dry_mass)

    @property
    def mdot_max(self) -> float:
        return self.thrust_max / (self.isp * G0)


@dataclass(frozen=True)
class Criteria:
    max_vertical: float = 3.0
    max_horizontal: float = 1.0

    def __post_init__(self) -> None:
        _positive("max_vertical", self.max_vertical, True)
        _positive("max_horizontal", self.max_horizontal, True)

    @classmethod
    def for_body(cls, body: Body, vertical: float = 3.0,
                 horizontal: float = 1.0) -> Criteria:
        if not body.landable:
            raise ValueError("body has no landable surface")
        cap = 0.1 * body.escape_v
        return cls(min(vertical, cap), min(horizontal, 0.5 * cap))


def tsiolkovsky_dv(isp: float, m0: float, mf: float) -> float:
    _positive("isp", isp)
    _positive("m0", m0)
    _positive("mf", mf)
    if mf > m0:
        raise ValueError("final mass cannot exceed initial mass")
    return isp * G0 * math.log(m0 / mf)


def propellant_for_dv(isp: float, m0: float, dv: float) -> float:
    _positive("isp", isp)
    _positive("m0", m0)
    _positive("dv", dv, True)
    return -m0 * math.expm1(-dv / (isp * G0))


def _newtonian(body: Body) -> None:
    if not isinstance(body.gravity, PointMass):
        raise ValueError("Newtonian orbital formulas require PointMass gravity")


def periapsis_speed(body: Body, r_apo: float, r_peri: float) -> float:
    _newtonian(body)
    _positive("r_apo", r_apo)
    _positive("r_peri", r_peri)
    if r_apo < r_peri:
        raise ValueError("apoapsis must be at least periapsis")
    return math.sqrt(2 * body.GM * r_apo / (r_peri * (r_apo + r_peri)))


def deorbit_dv(body: Body, r_apo: float, r_peri: float) -> float:
    periapsis_speed(body, r_apo, r_peri)
    return math.sqrt(body.GM / r_apo) - math.sqrt(
        2 * body.GM * r_peri / (r_apo * (r_apo + r_peri)))


def ideal_landing_dv(body: Body, r_apo: float, r_peri: float) -> float:
    """Cancel periapsis velocity instantaneously; excludes the deorbit burn."""
    return periapsis_speed(body, r_apo, r_peri)


def thrust_to_weight(body: Body, lander: Lander) -> float:
    return lander.thrust_max / (lander.wet_mass * body.surface_g)


def stopping_distance(v: float, a_thrust: float, g: float) -> float:
    if not math.isfinite(v):
        raise ValueError("v must be finite")
    _positive("a_thrust", a_thrust, True)
    _positive("g", g, True)
    return v**2 / (2 * (a_thrust - g)) if a_thrust > g else math.inf


@dataclass
class State:
    x: float
    y: float
    vx: float
    vy: float
    fuel: float
    t: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x", "y", "vx", "vy", "t"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if math.hypot(self.x, self.y) == 0:
            raise ValueError("position cannot be the origin")
        _positive("fuel", self.fuel, True)
        _positive("t", self.t, True)


def _verlet(s: State, body: Body, ax: float, ay: float, dt: float) -> None:
    gx, gy = body.gravity.accel(s.x, s.y, body.GM)
    x = s.x + s.vx * dt + 0.5 * (gx + ax) * dt**2
    y = s.y + s.vy * dt + 0.5 * (gy + ay) * dt**2
    gx1, gy1 = body.gravity.accel(x, y, body.GM)
    s.vx += (0.5 * (gx + gx1) + ax) * dt
    s.vy += (0.5 * (gy + gy1) + ay) * dt
    s.x, s.y = x, y
    s.t += dt


def advance(s: State, body: Body, lander: Lander, throttle: float,
            ux: float, uy: float, dt: float) -> None:
    """Mutate state using velocity Verlet; split a partially fueled timestep."""
    _positive("dt", dt)
    if not math.isfinite(throttle) or not 0 <= throttle <= 1:
        raise ValueError("throttle must be between zero and one")
    if not math.isfinite(ux) or not math.isfinite(uy):
        raise ValueError("thrust direction must be finite")
    _positive("fuel", s.fuel, True)
    force = lander.thrust_max * throttle
    if force and s.fuel:
        norm = math.hypot(ux, uy)
        if not math.isclose(norm, 1, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("powered thrust direction must be a unit vector")
        flow = force / (lander.isp * G0)
        fuel_time = s.fuel / flow
        burn_dt = min(dt, fuel_time)
        fuel_used = s.fuel if fuel_time <= dt else flow * burn_dt
        m0 = lander.dry_mass + s.fuel
        mf = m0 - fuel_used
        # Exact integrated impulse, with a coast after (not during) fuel burnout.
        a = lander.isp * G0 * math.log(m0 / mf) / burn_dt
        _verlet(s, body, a * ux, a * uy, burn_dt)
        s.fuel = max(0.0, s.fuel - fuel_used)
        if burn_dt < dt:
            _verlet(s, body, 0, 0, dt - burn_dt)
    else:
        _verlet(s, body, 0, 0, dt)


def specific_energy(s: State, body: Body) -> float:
    return 0.5 * (s.vx**2 + s.vy**2) + body.gravity.potential(
        math.hypot(s.x, s.y), body.GM)


def state_at_periapsis(body: Body, lander: Lander,
                      r_apo: float, r_peri: float) -> State:
    return State(r_peri, 0, 0, periapsis_speed(body, r_apo, r_peri), lander.fuel_mass)


@dataclass(frozen=True)
class Guidance:
    tau_v: float = 3.0
    tau_h: float = 3.0
    decel_fraction: float = 0.6
    horiz_share: float = 0.5
    touch_speed: float | None = None

    def __post_init__(self) -> None:
        _positive("tau_v", self.tau_v)
        _positive("tau_h", self.tau_h)
        for name in ("decel_fraction", "horiz_share"):
            value = getattr(self, name)
            _positive(name, value)
            if value > 1:
                raise ValueError(f"{name} must not exceed one")
        if self.touch_speed is not None:
            _positive("touch_speed", self.touch_speed, True)

    def command(self, s: State, body: Body, lander: Lander,
                crit: Criteria) -> tuple[float, float, float]:
        r = math.hypot(s.x, s.y)
        rx, ry = s.x / r, s.y / r
        tx, ty = -ry, rx
        vr, vh = s.vx * rx + s.vy * ry, s.vx * tx + s.vy * ty
        h = max(r - body.radius, 0)
        a_max = lander.thrust_max / (lander.dry_mass + s.fuel)
        if not a_max or not s.fuel:
            return 0.0, 1.0, 0.0
        g = body.g_at(r)
        v_touch = self.touch_speed if self.touch_speed is not None else 0.5 * crit.max_vertical
        a_dec = self.decel_fraction * max(a_max - g, 0.05 * a_max)
        a_stop = 0.0
        if h > 0 and vr < -v_touch:
            a_req = (vr**2 - v_touch**2) / (2 * h)
            if a_req >= a_dec:
                a_stop = a_req
        sink_alt = v_touch + h / (abs(vh) / (self.horiz_share * a_max) + 1)
        a_alt = max(0.0, (-sink_alt - vr) / self.tau_v)
        demand = max(a_stop, a_alt)
        a_up = g - vh * vh / r + demand if demand > 0 else 0.0
        a_up = min(max(a_up, 0.0), a_max)
        a_h_lim = math.sqrt(max(a_max**2 - a_up**2, 0.0))
        a_h = min(max(-vh / self.tau_h, -a_h_lim), a_h_lim)
        mag = math.hypot(a_up, a_h)
        if mag <= 0:
            return 0.0, 1.0, 0.0
        throttle = min(1.0, mag / a_max)
        if throttle < lander.min_throttle:
            throttle = lander.min_throttle if throttle >= 0.5 * lander.min_throttle else 0.0
        return (throttle, (a_up * rx + a_h * tx) / mag,
                (a_up * ry + a_h * ty) / mag)


@dataclass
class Result:
    outcome: str
    reason: str
    t: float
    v_vert: float
    v_horiz: float
    fuel_left: float
    dv_used: float
    dv_ideal: float
    steps: int
    log: list[tuple[float, float, float, float, float]] = field(default_factory=list)

    @property
    def gravity_loss(self) -> float:
        return self.dv_used - self.dv_ideal


def simulate_landing(body: Body, lander: Lander, s0: State,
                     guidance: Guidance | None = None, crit: Criteria | None = None,
                     dt_max: float | None = None, t_max: float | None = None,
                     log_every: float | None = None) -> Result:
    if not body.landable:
        raise ValueError(f"{body.name} has no landable surface")
    s = replace(s0)
    s.__post_init__()
    r0 = math.hypot(s.x, s.y)
    if r0 < body.radius:
        raise ValueError("initial state is below the surface")
    if s.fuel > lander.fuel_mass:
        raise ValueError("initial fuel exceeds lander capacity")
    guidance = guidance or Guidance()
    crit = crit or Criteria.for_body(body)
    tdyn = body.dynamical_time(body.radius)
    dt_max = max(0.001 * tdyn, 0.001) if dt_max is None else dt_max
    dt_min = min(0.0001 * tdyn, dt_max)
    t_max = 20 * body.dynamical_time(r0) if t_max is None else t_max
    _positive("dt_max", dt_max)
    _positive("t_max", t_max)
    if log_every is not None:
        _positive("log_every", log_every)
    m_start = lander.dry_mass + s.fuel
    dv_ideal = math.hypot(s.vx, s.vy)
    steps = 0
    log = []
    next_log = s.t
    deadline = s.t + t_max
    while True:
        r = math.hypot(s.x, s.y)
        h = r - body.radius
        vr = (s.vx * s.x + s.vy * s.y) / r
        vh = (-s.vx * s.y + s.vy * s.x) / r
        if log_every is not None and s.t >= next_log:
            log.append((s.t, h, vr, vh, s.fuel))
            next_log += log_every
        if h <= 0:
            outcome = "LANDED" if -vr <= crit.max_vertical and abs(vh) <= crit.max_horizontal else "CRASHED"
            reason = "touchdown within limits" if outcome == "LANDED" else "touchdown speed exceeds limits"
            break
        if s.t >= deadline:
            outcome, reason = "TIMEOUT", "maximum simulation time reached"
            break
        dt = min(dt_max, 0.01 * body.dynamical_time(r),
                 0.02 * h / max(math.hypot(s.vx, s.vy), 1e-9))
        dt = min(max(dt, dt_min), deadline - s.t)
        throttle, ux, uy = guidance.command(s, body, lander, crit)
        advance(s, body, lander, throttle, ux, uy, dt)
        steps += 1
    return Result(outcome, reason, s.t, -vr, abs(vh), s.fuel,
                  tsiolkovsky_dv(lander.isp, m_start, lander.dry_mass + s.fuel),
                  dv_ideal, steps, log)


@dataclass(frozen=True)
class Scenario:
    name: str
    body: Body
    lander: Lander
    r_apo_alt: float
    r_peri_alt: float
    guidance: Guidance = field(default_factory=Guidance)

    def __post_init__(self) -> None:
        _positive("r_apo_alt", self.r_apo_alt, True)
        _positive("r_peri_alt", self.r_peri_alt, True)
        if self.r_apo_alt < self.r_peri_alt:
            raise ValueError("apoapsis altitude must be at least periapsis altitude")


SCENARIOS = [
    Scenario("Moon", BODIES["moon"], Lander(7000, 8200, 45000, 311, 0.10),
             100e3, 15e3, Guidance(tau_v=1.5, tau_h=3)),
    Scenario("Ceres", BODIES["ceres"], Lander(1000, 500, 1500, 300, 0.10),
             100e3, 10e3, Guidance(tau_v=3, tau_h=3)),
    Scenario("67P", BODIES["67p"], Lander(90, 10, 5, 220),
             7e3, 5e3, Guidance(tau_v=15, tau_h=15)),
]


def run_scenario(sc: Scenario, **kw) -> tuple[dict[str, float | Criteria], Result]:
    ra, rp = sc.body.radius + sc.r_apo_alt, sc.body.radius + sc.r_peri_alt
    ideal = ideal_landing_dv(sc.body, ra, rp)
    calc = dict(surface_g=sc.body.surface_g, v_escape=sc.body.escape_v,
                v_peri=periapsis_speed(sc.body, ra, rp),
                twr=thrust_to_weight(sc.body, sc.lander), dv_avail=sc.lander.delta_v,
                dv_deorbit=deorbit_dv(sc.body, ra, rp), dv_ideal=ideal,
                limits=Criteria.for_body(sc.body))
    result = simulate_landing(sc.body, sc.lander,
                              state_at_periapsis(sc.body, sc.lander, ra, rp),
                              guidance=sc.guidance, **kw)
    return calc, result


def print_table(scenarios: list[Scenario] = SCENARIOS) -> None:
    rows = [(sc, *run_scenario(sc)) for sc in scenarios]
    print("Analytic feasibility (SI units; delta-v and speeds in m/s)")
    print(f"{'Scenario':<10} {'g':>9} {'Escape':>9} {'Periapsis':>10} {'Deorbit':>9} "
          f"{'Ideal dv':>10} {'Avail dv':>10} {'TWR':>7} {'dv margin':>10}")
    for sc, calc, result in rows:
        print(f"{sc.name:<10} {calc['surface_g']:9.5f} {calc['v_escape']:9.2f} "
              f"{calc['v_peri']:10.2f} {calc['dv_deorbit']:9.2f} {calc['dv_ideal']:10.2f} "
              f"{calc['dv_avail']:10.2f} {calc['twr']:7.3f} "
              f"{calc['dv_avail'] - calc['dv_ideal']:10.2f}")
    print("\nSimulated landing (speeds in m/s; time in s; fuel in kg)")
    print(f"{'Scenario':<10} {'Outcome':<9} {'Time':>9} {'Vertical':>9} {'Horiz':>9} "
          f"{'V limit':>9} {'H limit':>9} {'dv used':>10} {'Grav loss':>10} {'Fuel left':>10}")
    for sc, calc, result in rows:
        limits = calc["limits"]
        print(f"{sc.name:<10} {result.outcome:<9} {result.t:9.1f} {result.v_vert:9.3f} "
              f"{result.v_horiz:9.3f} {limits.max_vertical:9.3f} {limits.max_horizontal:9.3f} "
              f"{result.dv_used:10.2f} {result.gravity_loss:10.2f} {result.fuel_left:10.3f}")
        if result.outcome != "LANDED":
            print(f"  {result.reason}")


if __name__ == "__main__":
    print_table()
