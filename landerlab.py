"""
landerlab.py -- headless landing physics for a configurable central body.

No graphics. Everything is SI (m, s, kg, N). 2D planar, point-mass lander,
instantaneous attitude (thrust direction is free). Atmosphere, rotation of the
body, and attitude dynamics are deliberately left out for now.

Layers
  1. Body + gravity models (swappable)            -> "what world is this?"
  2. Lander + analytic calculators (closed form)  -> "is this even feasible?"
  3. Integrator + guidance + simulate_landing()   -> "what actually happens?"
  4. SCENARIOS + print_table()                    -> the comparison table
Run `python landerlab.py` for the table, `pytest` for the validation tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

G0 = 9.80665          # m/s^2, standard gravity (used only in the rocket equation)
C_LIGHT = 299_792_458.0


# --------------------------------------------------------------------------
# 1. Bodies and gravity models
# --------------------------------------------------------------------------
class HorizonCrossed(Exception):
    pass


class GravityModel(Protocol):
    def accel(self, x: float, y: float, GM: float) -> tuple[float, float]: ...
    def potential(self, r: float, GM: float) -> float: ...


@dataclass(frozen=True)
class PointMass:
    """Newtonian a = -GM r_hat / r^2."""
    def accel(self, x, y, GM):
        r2 = x * x + y * y
        k = -GM / (r2 * math.sqrt(r2))
        return k * x, k * y

    def potential(self, r, GM):
        return -GM / r


@dataclass(frozen=True)
class PseudoNewtonian:
    """Paczynski-Wiita: Phi = -GM/(r - Rs). Gives an ISCO at 3 Rs and a horizon at Rs."""
    rs: float

    def accel(self, x, y, GM):
        r = math.hypot(x, y)
        d = r - self.rs
        if d <= 0:
            raise HorizonCrossed("r <= Rs")
        k = -GM / (d * d * r)
        return k * x, k * y

    def potential(self, r, GM):
        return -GM / (r - self.rs)


@dataclass(frozen=True)
class Body:
    name: str
    GM: float                       # m^3/s^2 (GM is known better than G or M separately)
    radius: float                   # m
    gravity: GravityModel = field(default_factory=PointMass)
    landable: bool = True

    def g_at(self, r: float) -> float:
        gx, gy = self.gravity.accel(r, 0.0, self.GM)
        return -gx

    @property
    def surface_g(self) -> float:
        return self.g_at(self.radius)

    @property
    def escape_v(self) -> float:
        return math.sqrt(-2.0 * self.gravity.potential(self.radius, self.GM))

    def circular_v(self, r: float) -> float:
        return math.sqrt(r * self.g_at(r))

    def dynamical_time(self, r: float) -> float:
        """sqrt(r^3/GM): the natural time unit at radius r (orbital period = 2*pi*this)."""
        return math.sqrt(r ** 3 / self.GM)


_BH10_GM = 1.32712440018e20 * 10
BODIES = {
    "moon":  Body("Moon", 4.9048695e12, 1_737_400.0),
    "ceres": Body("Ceres", 6.2628e10, 469_700.0),
    "67p":   Body("Comet 67P", 666.2, 2_000.0),
    "bh10":  Body("10 Msun black hole", _BH10_GM, 2 * _BH10_GM / C_LIGHT ** 2,
                  PseudoNewtonian(2 * _BH10_GM / C_LIGHT ** 2), landable=False),
}


# --------------------------------------------------------------------------
# 2. Lander and analytic calculators
# --------------------------------------------------------------------------
@dataclass
class Lander:
    dry_mass: float                 # structure + engines + tanks (kg)
    fuel_mass: float                # propellant (kg)
    thrust_max: float               # N
    isp: float                      # s
    min_throttle: float = 0.0       # engine can't run below this fraction (or is off)
    payload_mass: float = 0.0       # cargo/crew/instruments, carried all the way down (kg)
    name: str = ""

    @property
    def empty_mass(self) -> float:
        """Everything that is still there after the propellant is gone."""
        return self.dry_mass + self.payload_mass

    @property
    def mdot_max(self) -> float:
        return self.thrust_max / (self.isp * G0)

    @property
    def wet_mass(self) -> float:
        return self.empty_mass + self.fuel_mass

    @property
    def delta_v(self) -> float:
        return tsiolkovsky_dv(self.isp, self.wet_mass, self.empty_mass)

    def with_(self, **changes) -> "Lander":
        """Copy with some fields changed, e.g. lander.with_(payload_mass=500)."""
        from dataclasses import replace
        return replace(self, **changes)


@dataclass
class Criteria:
    """Success window. Defaults scale with the body so a comet isn't judged by lunar rules."""
    max_vertical: float
    max_horizontal: float

    @staticmethod
    def for_body(body: Body, vertical: float = 3.0, horizontal: float = 1.0) -> "Criteria":
        cap = 0.1 * body.escape_v           # touchdown must be << escape speed
        return Criteria(min(vertical, cap), min(horizontal, 0.5 * cap))


def tsiolkovsky_dv(isp: float, m0: float, m1: float) -> float:
    return isp * G0 * math.log(m0 / m1)


def propellant_for_dv(isp: float, m0: float, dv: float) -> float:
    return m0 * (1.0 - math.exp(-dv / (isp * G0)))


def periapsis_speed(body: Body, r_apo: float, r_peri: float) -> float:
    return math.sqrt(2 * body.GM * r_apo / (r_peri * (r_apo + r_peri)))


def deorbit_dv(body: Body, r_circ: float, r_peri: float) -> float:
    """Hohmann-style retrograde burn lowering periapsis from a circular orbit."""
    a = 0.5 * (r_circ + r_peri)
    v_apo = math.sqrt(body.GM * (2 / r_circ - 1 / a))
    return body.circular_v(r_circ) - v_apo


def ideal_landing_dv(body: Body, r_apo: float, r_peri: float) -> float:
    """Lower bound: cancel all periapsis speed, zero gravity losses. Reality costs more."""
    return periapsis_speed(body, r_apo, r_peri)


def thrust_to_weight(body: Body, lander: Lander) -> float:
    return lander.thrust_max / (lander.wet_mass * body.surface_g)


def stopping_distance(v: float, a_thrust: float, g: float) -> float:
    """Vertical suicide-burn distance to null a downward speed v. Needs a_thrust > g."""
    if a_thrust <= g:
        return math.inf
    return v * v / (2.0 * (a_thrust - g))


# --------------------------------------------------------------------------
# 3. Integrator, guidance, simulation
# --------------------------------------------------------------------------
@dataclass
class State:
    x: float
    y: float
    vx: float
    vy: float
    fuel: float
    t: float = 0.0


@dataclass(frozen=True)
class PlayerControl:
    """Manual engine command: throttle fraction and steering angle from prograde, in radians."""
    throttle: float = 0.0
    angle: float = 0.0
    time_scale: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.throttle) or not 0.0 <= self.throttle <= 1.0:
            raise ValueError("throttle must be between 0 and 1")
        if not math.isfinite(self.angle):
            raise ValueError("angle must be finite")
        if not math.isfinite(self.time_scale) or self.time_scale <= 0.0:
            raise ValueError("time_scale must be a positive finite number")

    def command(self, s: State) -> tuple[float, float, float]:
        speed = math.hypot(s.vx, s.vy)
        if speed > 0.0:
            px, py = s.vx / speed, s.vy / speed
        else:
            r = math.hypot(s.x, s.y)
            if r == 0.0:
                raise ValueError("cannot steer at the body's center")
            px, py = -s.y / r, s.x / r
        c, sn = math.cos(self.angle), math.sin(self.angle)
        return self.throttle, c * px - sn * py, sn * px + c * py


def state_at_periapsis(body: Body, lander: Lander, r_apo: float, r_peri: float) -> State:
    return State(r_peri, 0.0, 0.0, periapsis_speed(body, r_apo, r_peri), lander.fuel_mass)


def specific_energy(s: State, body: Body) -> float:
    return 0.5 * (s.vx ** 2 + s.vy ** 2) + body.gravity.potential(math.hypot(s.x, s.y), body.GM)


def advance(s: State, body: Body, lander: Lander, throttle: float,
            ux: float, uy: float, dt: float) -> None:
    """One velocity-Verlet step. Thrust is held constant over the step; mass is
    taken at mid-step; fuel runs out mid-step correctly."""
    m0 = lander.empty_mass + s.fuel
    thr = throttle if s.fuel > 0 else 0.0
    dm = thr * lander.mdot_max * dt
    if dm > s.fuel:                       # ran dry partway through the step
        thr *= s.fuel / dm
        dm = s.fuel
    a_t = thr * lander.thrust_max / (m0 - 0.5 * dm)
    atx, aty = a_t * ux, a_t * uy

    gx, gy = body.gravity.accel(s.x, s.y, body.GM)
    a0x, a0y = gx + atx, gy + aty
    s.x += s.vx * dt + 0.5 * a0x * dt * dt
    s.y += s.vy * dt + 0.5 * a0y * dt * dt
    gx1, gy1 = body.gravity.accel(s.x, s.y, body.GM)
    s.vx += 0.5 * (a0x + gx1 + atx) * dt
    s.vy += 0.5 * (a0y + gy1 + aty) * dt
    s.fuel -= dm
    s.t += dt


def advance_controlled(s: State, body: Body, lander: Lander,
                       control: PlayerControl, dt: float) -> None:
    """Apply one player command for dt * time_scale seconds."""
    scaled_dt = dt * control.time_scale
    if not math.isfinite(scaled_dt) or scaled_dt <= 0.0:
        raise ValueError("scaled timestep must be a positive finite number")
    throttle, ux, uy = control.command(s)
    advance(s, body, lander, throttle, ux, uy, scaled_dt)


@dataclass
class Guidance:
    """Vertical-priority braking. Tracks a descent-rate profile v_r = -(v_touch + sqrt(2 a h)),
    spends whatever thrust is left killing horizontal speed. No target site: it lands where
    the orbit takes it."""
    tau_v: float = 3.0              # s, vertical-rate response time
    tau_h: float = 3.0              # s, horizontal-speed response time
    decel_fraction: float = 0.6     # fraction of available net decel the profile assumes
    horiz_share: float = 0.5        # fraction of max accel assumed available for horizontal braking
    touch_speed: Optional[float] = None   # target sink rate at contact (default: half the limit)

    def command(self, s: State, body: Body, lander: Lander, crit: Criteria):
        r = math.hypot(s.x, s.y)
        rx, ry = s.x / r, s.y / r
        tx, ty = -ry, rx
        vr = s.vx * rx + s.vy * ry
        vh = s.vx * tx + s.vy * ty
        h = max(r - body.radius, 0.0)
        m = lander.empty_mass + s.fuel
        a_max = lander.thrust_max / m
        g = body.g_at(r)
        v_touch = self.touch_speed if self.touch_speed is not None else 0.5 * crit.max_vertical

        a_dec = self.decel_fraction * max(a_max - g, 0.05 * a_max)
        # (1) Suicide-burn law: start braking once the decel needed to reach v_touch
        #     exactly at the surface equals what we're willing to use (a_dec); then track it.
        a_stop = 0.0
        if h > 0.0 and vr < -v_touch:
            a_req = (vr * vr - v_touch ** 2) / (2.0 * h)
            if a_req >= a_dec:
                a_stop = a_req
        # (2) Time-to-go limit: don't sink faster than altitude/(time to kill horizontal speed)
        sink_alt = v_touch + h / (abs(vh) / (self.horiz_share * a_max) + 1.0)
        a_alt = max(0.0, (-sink_alt - vr) / self.tau_v)
        # gravity minus centrifugal relief, plus the larger of the two demands
        demand = max(a_stop, a_alt)
        a_up = (g - vh * vh / r + demand) if demand > 0.0 else 0.0   # coast when nothing is needed
        a_up = min(max(a_up, 0.0), a_max)
        a_h_lim = math.sqrt(max(a_max ** 2 - a_up ** 2, 0.0))
        a_h = min(max(-vh / self.tau_h, -a_h_lim), a_h_lim)

        mag = math.hypot(a_up, a_h)
        if mag <= 0.0:
            return 0.0, 1.0, 0.0
        throttle = mag / a_max
        mt = lander.min_throttle
        if throttle < mt:
            throttle = mt if throttle >= 0.5 * mt else 0.0
        ux = (a_up * rx + a_h * tx) / mag
        uy = (a_up * ry + a_h * ty) / mag
        return throttle, ux, uy


@dataclass
class Result:
    outcome: str                    # LANDED | CRASHED | TIMEOUT
    reason: str
    t: float
    v_vert: float                   # touchdown vertical speed (m/s, downward positive)
    v_horiz: float
    fuel_left: float
    dv_used: float
    dv_ideal: float
    steps: int
    log: list = field(default_factory=list)
    peak_g: float = 0.0             # peak thrust acceleration felt, in Earth g

    @property
    def gravity_loss(self) -> float:
        return self.dv_used - self.dv_ideal


def simulate_landing(body: Body, lander: Lander, s0: State,
                     guidance: Optional[Guidance] = None,
                     crit: Optional[Criteria] = None,
                     dt_max: Optional[float] = None,
                     t_max: Optional[float] = None,
                     log_every: Optional[float] = None) -> Result:
    guidance = guidance or Guidance()
    crit = crit or Criteria.for_body(body)
    s = State(s0.x, s0.y, s0.vx, s0.vy, s0.fuel, s0.t)
    r0 = math.hypot(s.x, s.y)
    tdyn_surface = body.dynamical_time(body.radius)
    dt_max = dt_max or max(1e-3 * tdyn_surface, 1e-3)       # ~1 s at the Moon
    dt_min = 1e-3                                           # s, absolute floor
    t_max = t_max or 20.0 * body.dynamical_time(r0)
    m_start = lander.empty_mass + s.fuel
    v0 = math.hypot(s.vx, s.vy)
    dv_ideal = v0                                           # speed to cancel (surface at rest)
    log, next_log, steps = [], 0.0, 0
    peak_a = 0.0

    while True:
        r = math.hypot(s.x, s.y)
        h = r - body.radius
        vr = (s.vx * s.x + s.vy * s.y) / r
        vh = (-s.vx * s.y + s.vy * s.x) / r
        m = lander.empty_mass + s.fuel
        dv_used = tsiolkovsky_dv(lander.isp, m_start, m)

        if log_every and s.t >= next_log:
            log.append((s.t, h, vr, vh, s.fuel))
            next_log += log_every

        if h <= 0.0:
            vv, vhh = -vr, abs(vh)
            if vv <= crit.max_vertical and vhh <= crit.max_horizontal:
                outcome, reason = "LANDED", "within limits"
            else:
                bad = []
                if vv > crit.max_vertical:
                    bad.append(f"vertical {vv:.2f} > {crit.max_vertical:.2f}")
                if vhh > crit.max_horizontal:
                    bad.append(f"horizontal {vhh:.2f} > {crit.max_horizontal:.2f}")
                outcome, reason = "CRASHED", "; ".join(bad)
                if s.fuel <= 0:
                    reason += " (out of fuel)"
            return Result(outcome, reason, s.t, vv, vhh, s.fuel, dv_used, dv_ideal, steps, log, peak_g=peak_a / G0)
        if s.t > t_max:
            return Result("TIMEOUT", "never reached surface", s.t, -vr, abs(vh),
                          s.fuel, dv_used, dv_ideal, steps, log, peak_g=peak_a / G0)

        speed = math.hypot(s.vx, s.vy)
        dt = min(dt_max, 0.01 * body.dynamical_time(r), 0.02 * h / max(speed, 1e-9))
        dt = max(dt, dt_min)
        throttle, ux, uy = guidance.command(s, body, lander, crit)
        if throttle > 0.0 and s.fuel > 0.0:
            # While firing, one step must not change velocity by more than a fraction
            # of the touchdown tolerance, or the controller can't resolve its own authority.
            a_now = throttle * lander.thrust_max / m
            dt = max(dt_min, min(dt, 0.25 * 0.5 * crit.max_vertical / a_now))
        if s.fuel > 0:
            peak_a = max(peak_a, throttle * lander.thrust_max / m)
        advance(s, body, lander, throttle, ux, uy, dt)
        steps += 1


def simulate_player_landing(body: Body, lander: Lander, s0: State,
                            control_source: Callable[[State], PlayerControl],
                            dt: float = 0.1,
                            crit: Optional[Criteria] = None,
                            t_max: Optional[float] = None,
                            log_every: Optional[float] = None) -> Result:
    """Run a landing with fresh player input each step; the callback receives a state snapshot."""
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite number")
    if t_max is not None and (not math.isfinite(t_max) or t_max <= 0.0):
        raise ValueError("t_max must be a positive finite number")

    crit = crit or Criteria.for_body(body)
    s = State(s0.x, s0.y, s0.vx, s0.vy, s0.fuel, s0.t)
    r0 = math.hypot(s.x, s.y)
    t_max = t_max or 20.0 * body.dynamical_time(r0)
    m_start = lander.empty_mass + s.fuel
    v0 = math.hypot(s.vx, s.vy)
    log, next_log, steps = [], 0.0, 0
    peak_a = 0.0

    while True:
        r = math.hypot(s.x, s.y)
        h = r - body.radius
        vr = (s.vx * s.x + s.vy * s.y) / r
        vh = (-s.vx * s.y + s.vy * s.x) / r
        m = lander.empty_mass + s.fuel
        dv_used = tsiolkovsky_dv(lander.isp, m_start, m)

        if log_every and s.t >= next_log:
            log.append((s.t, h, vr, vh, s.fuel))
            next_log += log_every

        if h <= 0.0:
            vv, vhh = -vr, abs(vh)
            if vv <= crit.max_vertical and vhh <= crit.max_horizontal:
                outcome, reason = "LANDED", "within limits"
            else:
                bad = []
                if vv > crit.max_vertical:
                    bad.append(f"vertical {vv:.2f} > {crit.max_vertical:.2f}")
                if vhh > crit.max_horizontal:
                    bad.append(f"horizontal {vhh:.2f} > {crit.max_horizontal:.2f}")
                outcome, reason = "CRASHED", "; ".join(bad)
                if s.fuel <= 0:
                    reason += " (out of fuel)"
            return Result(outcome, reason, s.t, vv, vhh, s.fuel, dv_used, v0,
                          steps, log, peak_g=peak_a / G0)
        if s.t >= t_max:
            return Result("TIMEOUT", "never reached surface", s.t, -vr, abs(vh),
                          s.fuel, dv_used, v0, steps, log, peak_g=peak_a / G0)

        snapshot = State(s.x, s.y, s.vx, s.vy, s.fuel, s.t)
        control = control_source(snapshot)
        if not isinstance(control, PlayerControl):
            raise TypeError("control_source must return a PlayerControl")
        step = min(dt, (t_max - s.t) / control.time_scale)
        throttle = control.throttle if s.fuel > 0 else 0.0
        if s.fuel > 0:
            peak_a = max(peak_a, throttle * lander.thrust_max / m)
        advance_controlled(s, body, lander, control, step)
        steps += 1


# --------------------------------------------------------------------------
# 4. Scenarios and the table
# --------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    body: Body
    lander: Lander
    r_apo_alt: float                # m above surface
    r_peri_alt: float
    guidance: Guidance = field(default_factory=Guidance)


SCENARIOS = [
    Scenario("Moon / LM-like, 15x100 km", BODIES["moon"],
             Lander(7000, 8200, 45_000, 311, 0.10), 100e3, 15e3,
             Guidance(tau_v=1.5, tau_h=3)),
    Scenario("Ceres / small lander, 10x100 km", BODIES["ceres"],
             Lander(1000, 500, 1_500, 300, 0.10), 100e3, 10e3,
             Guidance(tau_v=3, tau_h=3)),
    Scenario("Comet 67P / micro lander, 5x7 km", BODIES["67p"],
             Lander(90, 10, 5.0, 220, 0.0), 7e3, 5e3,
             Guidance(tau_v=15, tau_h=15)),
]


def run_scenario(sc: Scenario, **kw) -> tuple[dict, Result]:
    b, L = sc.body, sc.lander
    r_apo, r_peri = b.radius + sc.r_apo_alt, b.radius + sc.r_peri_alt
    s0 = state_at_periapsis(b, L, r_apo, r_peri)
    res = simulate_landing(b, L, s0, sc.guidance, **kw)
    calc = dict(
        surface_g=b.surface_g, v_escape=b.escape_v,
        v_peri=periapsis_speed(b, r_apo, r_peri),
        dv_deorbit=deorbit_dv(b, r_apo, r_peri) if r_apo > r_peri else 0.0,
        dv_ideal=ideal_landing_dv(b, r_apo, r_peri),
        dv_avail=L.delta_v, twr=thrust_to_weight(b, L),
        limits=Criteria.for_body(b),
    )
    return calc, res


def print_table(scenarios=SCENARIOS) -> None:
    print("ANALYTIC CALCULATOR (closed form, no simulation)")
    hdr = f"{'scenario':36} {'g_surf':>8} {'v_esc':>9} {'v_peri':>8} {'dv_ideal':>9} {'dv_avail':>9} {'TWR':>7} {'margin':>7}"
    print(hdr)
    rows = []
    for sc in scenarios:
        calc, res = run_scenario(sc)
        rows.append((sc, calc, res))
        margin = calc["dv_avail"] - calc["dv_ideal"]
        print(f"{sc.name:36} {calc['surface_g']:8.4f} {calc['v_escape']:9.2f} "
              f"{calc['v_peri']:8.1f} {calc['dv_ideal']:9.1f} {calc['dv_avail']:9.1f} "
              f"{calc['twr']:7.2f} {margin:7.0f}")
    print("\nSIMULATION vs LANDING WINDOW")
    print(f"{'scenario':36} {'result':8} {'t(s)':>8} {'v_vert':>7} {'v_hor':>7} "
          f"{'limit v/h':>11} {'dv_used':>8} {'g-loss':>7} {'fuel_left':>9}")
    for sc, calc, res in rows:
        lim = calc["limits"]
        print(f"{sc.name:36} {res.outcome:8} {res.t:8.0f} {res.v_vert:7.2f} {res.v_horiz:7.2f} "
              f"{lim.max_vertical:5.2f}/{lim.max_horizontal:4.2f} {res.dv_used:8.1f} "
              f"{res.gravity_loss:7.1f} {res.fuel_left:9.1f}")
        if res.outcome != "LANDED":
            print(f"    -> {res.reason}")


if __name__ == "__main__":
    print_table()
