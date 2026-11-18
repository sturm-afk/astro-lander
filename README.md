# landerlab

A headless, standard-library-only Python landing prototype: analytic feasibility
calculators and a planar, powered-descent simulation for the Moon, Ceres, and
comet 67P. Run with Python 3.10 or newer:

```sh
python landerlab.py
```

The first table compares surface gravity, escape/periapsis speeds, deorbit
delta-v, ideal landing delta-v, available rocket delta-v, initial surface TWR,
and delta-v margin. The second reports simulated touchdown speeds against their
landing windows, propellant remaining, delta-v spent, and gravity loss. All three
reference scenarios land; an analytic margin alone does not guarantee success.

## Experimenting

All quantities use SI units: metres, seconds, kilograms, newtons, and seconds of
specific impulse. Orbital calculator and `state_at_periapsis` inputs are **radii
from the body's centre**; `Scenario` inputs are **altitudes above its surface**.

```python
from dataclasses import replace
from landerlab import SCENARIOS, run_scenario, print_table

moon = SCENARIOS[0]
low_thrust = replace(moon, name="Moon 20 kN",
                     lander=replace(moon.lander, thrust_max=20_000))
calc, result = run_scenario(low_thrust, dt_max=0.1)
print(result.outcome, result.fuel_left, result.gravity_loss)
print_table([moon, low_thrust])
```

`Lander(dry_mass, fuel_mass, thrust_max, isp, min_throttle=0)` exposes wet mass,
maximum mass flow, and available delta-v. `State(x, y, vx, vy, fuel, t=0)` is
mutable. `advance(state, body, lander, throttle, ux, uy, dt)` advances it in
place; powered thrust directions must be unit vectors. `simulate_landing` copies
its input state, accepts guidance, touchdown criteria, timestep/time limits,
and optional `log_every` seconds, and returns a `Result`. Log entries are
`(t, altitude, radial_velocity, tangential_velocity, fuel)` tuples; radial
velocity is positive outward, whereas `Result.v_vert` is positive downward.

For player-directed maneuvers, create `PlayerControl(throttle, angle)` and call
`advance_controlled(state, body, lander, control, dt)` one step at a time. Throttle
is a fraction from 0 to 1; angle is in radians from current prograde (positive
counterclockwise, so pi is retrograde). At rest, prograde falls back to the local
counterclockwise tangent. `PlayerControl` also accepts `time_scale`, a positive
finite multiplier for the simulated duration of each step (default `1.0`); for
example, `PlayerControl(time_scale=10)` advances ten times as much simulated
time per call. Inspect the updated state and issue another command as needed.
`simulate_player_landing(body, lander, state, control_source, dt=0.1)` does this
until touchdown or timeout, asking `control_source(state_snapshot)` for a fresh
`PlayerControl` each step and returning the resulting `Result`. The command,
including its time scale, is held for that scaled step, so use larger scales for
coasting and smaller scales near maneuvers for finer control. A
keyboard/mouse or gamepad adapter can translate its inputs into the same
throttle-and-angle command; no device handling or UI is included here.

The default guidance prioritizes vertical stopping and uses remaining thrust
for orbital braking. Initial **surface** TWR below one is not an automatic
failure: orbital motion initially supplies centrifugal support, and burning fuel
raises thrust acceleration. The 20 kN lunar example lands; 9 kN does not.
Default touchdown limits are 3 m/s vertical and 1 m/s horizontal, reduced for
small bodies relative to escape speed.

## Physics and limitations

- Gravity is a swappable `GravityModel` strategy. `PointMass` supplies Newtonian
  gravity; `Body` wraps its gravitational parameter, radius, and surface helpers.
- Coasting uses velocity Verlet. Thrust uses the rocket equation as fuel mass
  changes; a burn exhausting fuel midstep is followed by a separate coast.
  Integration and surface-crossing resolution are finite, so refine `dt_max`
  when checking numerical sensitivity. This is not a flight-qualified controller.
- Bodies are spherical, fixed, and nonrotating. There is no atmosphere, drag,
  terrain, attitude dynamics, engine slew, navigation error, or third-body gravity.
- Flights begin **at periapsis**. Their simulated fuel and ideal landing delta-v
  do **not** include the earlier deorbit burn, which is listed separately.
  `gravity_loss = dv_used - dv_ideal` is the prototype's excess delta-v metric,
  including steering/controller losses, not a pure gravity-only integral.
- `PseudoNewtonian(rs)` is a Paczyński–Wiita toy potential, **not real GR**.
  It raises `HorizonCrossed` at/below `rs`. The `bh10` body is nonlandable and
  rejected before any surface criteria are evaluated. Newtonian closed-form
  orbital calculators explicitly reject non-`PointMass` strategies; numerical
  orbital integration and local circular-speed calculations remain available.

## Tests

The runtime module has no third-party dependencies. Tests require pytest:

```sh
python -m pytest -q
```

Tests cover orbit closure, energy/angular-momentum conservation, rocket-equation
accuracy, burnout timing, reference landings, insufficient fuel/thrust,
vertical descent, gravity-model boundaries, API contracts, and invalid inputs.
