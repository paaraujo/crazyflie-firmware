---
title: Swarm encirclement app
page_id: swarm_encircle_app
---

The swarm encirclement app estimates where a Crazyflie sits on a closed 3D
trajectory around a fixed centre, and how far its two neighbours are from their
nominal positions in the formation — using **only UWB ranges**. No motion
capture, no GPS, and no exchange of state between vehicles.

It lives out of tree in `examples/app_swarm_encircle/` and builds on the
[Loco TDoA3 hybrid mode](/docs/functional-areas/loco-positioning-system/tdoa3_hybrid_mode.md),
which supplies the two-way ranging. The mathematics is derived in
`docs/swarm_encircle/theory.tex`.

## Functionality

Each vehicle follows a curve of radius `r` about an encirclement centre
`c = (cx, cy, cz)`, parameterised by a single scalar: its **phase** `theta`,
measured counter-clockwise from the `+x` axis of the anchor frame. The
formation is held by keeping the phase *separations* between neighbours at
`2*pi/n` radians.

The app answers two questions from five scalar ranges — three to fixed anchors,
two to the neighbouring Crazyflies:

* **Where am I on the curve?** — the phase `theta` and radius `r`.
* **Where are my neighbours relative to where they should be?** — the formation
  errors `delta_k` (leader) and `delta_j` (follower), and hence the separations
  the encirclement controller consumes.

### Why a filter on the curve, and not trilateration

Constraining the vehicle to the curve reduces its state from three degrees of
freedom to one. That is the whole point: three anchors give three ranges, which
is exactly enough to trilaterate a 3D point and therefore leaves no redundancy
and poor conditioning — in practice a metre of vertical error is easy to
produce. The same three ranges determine a single phase very well.

The filter therefore does **not** consult the vehicle's own position estimate.
It fuses the ranges directly.

### Two independent estimates, on purpose

| Source | Log group | Path |
|--------|-----------|------|
| Range-driven filter | `swarmFilter` | UWB ranges → EKF on the curve |
| Position-derived | `swarmPhase` | IMU + anchor ranges → onboard EKF → `atan2` |

The second is a **cross-check**, not a fallback. Because the filter never looks
at the position estimate, nothing else would catch it diverging or latching onto
the wrong branch of the chord ambiguity. Their difference is published as
`swarmDiagnostics.dTh`.

### Units

The log groups split along a machine/human line, and mixing them up is the
easiest mistake to make:

| Group | Units | Intended reader |
|-------|-------|-----------------|
| `swarmFilter` | **SI** — radians, rad/s, metres | downstream code |
| `swarmCov` | **SI squared** — rad², (rad/s)², m² | downstream code |
| `swarmDiagnostics` | degrees and metres | a person watching a plot |
| `swarmPhase` | radians (`th`) and degrees (`thDeg`) | either |

`swarmFilter` and `swarmCov` are ordered identically, so element *i* of one is
the state and element *i* of the other is its variance. That is what lets a
subscriber assemble a state-with-covariance message without converting or
squaring anything.

### The curve

The trajectory is a circle distorted by a phase-dependent rotation. Resolving
the rotation generator in the moving frame of the circle shows that its radial
component does nothing at all, and that the remaining tangential component
reduces the construction to spherical coordinates:

```
q(theta) = [ r cos b(theta) cos theta,
             r cos b(theta) sin theta,
             cz - r sin b(theta) ]
```

with azimuth `theta` and elevation `-b(theta)`. Two consequences shape the
implementation:

* **No matrix exponential.** The whole curve is a handful of `cos`/`sin` calls,
  which is why it runs comfortably on the STM32.
* **`atan2` is exact.** `theta = atan2(qy - cy, qx - cx)` holds identically, for
  any elevation profile however severe. The embedding cannot move a point in
  azimuth, only in elevation, so there is no fixed-point inversion to perform.

The elevation profile is a truncated Fourier series,

```
b(theta) = b0 + sum_k [ a_k cos(k theta) + b_k sin(k theta) ]
```

fitted **offline** from waypoints. `K = 0` with `b0 = 0` is the flat circle, and
is the default.

## Building

The app requires hybrid mode and a pinned ranging algorithm, both provided by
the supplied defconfig:

```bash
cd examples/app_swarm_encircle
make cf21bl_swarm_defconfig
make -j$(nproc)
cfloader flash build/cf21bl.bin stm32-fw -w radio://0/80/2M/E7E7E7E718
```

> The `make cf21bl_swarm_defconfig` step is **not optional**. Out-of-tree builds
> default to `alldefconfig`, which enables everything and overflows CCM. Running
> plain `make cf21bl_defconfig` is equally wrong: it silently drops the Loco
> settings and produces a binary with no `tdoa3.hm*` parameters at all.

The resulting binary is a complete firmware image — flash it *instead of* the
one built from the repository root, not alongside it.

The anchors must be running in **TDoA3 mode**. Hybrid mode performs two-way
ranging inside the TDoA3 protocol; the Crazyflies transmit while the anchors
broadcast as usual.

## Parameters

### `swarmCurve` — the trajectory

| Parameter | Type | Default | Unit | Meaning |
|-----------|------|---------|------|---------|
| `cx` | float | 0.0 | m | X of the encirclement centre in the anchor frame |
| `cy` | float | 0.0 | m | Y of the encirclement centre |
| `cz` | float | 1.0 | m | Altitude of the centre above the anchor plane. Must be strictly positive — see below. |
| `K` | uint8 | 0 | — | Harmonic order of the elevation profile, 0–4. `0` is a flat circle. |
| `b0` | float | 0.0 | rad | Constant term of the elevation profile |
| `a1` | float | 0.0 | rad | Coefficient of `cos(theta)` |
| `b1` | float | 0.0 | rad | Coefficient of `sin(theta)` |
| `a2` | float | 0.0 | rad | Coefficient of `cos(2 theta)` |
| `b2` | float | 0.0 | rad | Coefficient of `sin(2 theta)` |
| `a3` | float | 0.0 | rad | Coefficient of `cos(3 theta)` |
| `b3` | float | 0.0 | rad | Coefficient of `sin(3 theta)` |
| `a4` | float | 0.0 | rad | Coefficient of `cos(4 theta)` |
| `b4` | float | 0.0 | rad | Coefficient of `sin(4 theta)` |

Coefficients above `K` are ignored, so leaving unused ones at zero is harmless.
The defaults (`K = 0`, `b0 = 0`) describe a **flat circle at 1 m altitude** — the
vehicle flies a level ring until an elevation profile is loaded.

The centre is **not** the corner anchor. Placing it in the anchor plane
(`cz = 0`) is the degenerate configuration: the vertical direction becomes
unobservable and the corner anchor stops carrying phase information. Keep `cz`
comparable to `r`.

Coefficients are produced by `tools/swarm/fit_elevation.py`, which projects
sketched waypoints onto the reachable family and fits the smallest adequate
harmonic order. It prints them ready to paste, both as a cflib dict and as a
Crazyswarm2 YAML block.

### `swarmFilter` — the estimator

| Parameter | Type | Default | Unit | Meaning |
|-----------|------|---------|------|---------|
| `en` | uint8 | 1 | — | Run the filter (1) or idle (0). Idling **freezes** the state rather than resetting it, so re-enabling resumes from a stale estimate — follow with `rst` unless nothing moved. |
| `rst` | uint8 | 0 | — | Write 1 to force re-initialisation; self-clearing. Re-seeds `theta` and `r` from the position-derived phase, zeroes the formation and rate errors, and restores the initial covariance. |
| `n` | uint8 | 3 | — | Agents in the ring; sets the nominal spacing `2*pi/n`. Must match both the real count and the ring order used by `setup_swarm.py`. |
| `wz` | float | 0.0 | rad/s | Commanded angular rate, used by the prediction. This is what you are *commanding*, not a measurement — 0 for stationary tests. |
| `sigA` | float | 0.03 | m | Anchor range measurement noise, one sigma |
| `sigC` | float | 0.03 | m | Chord (Crazyflie-to-Crazyflie) measurement noise, one sigma. Kept separate because chords are typically noisier under airtime contention. |
| `gate` | float | 25.0 | — | Innovation gate as normalised innovation squared. The threshold is the **square** of a sigma multiple: 9 = 3σ, 25 = 5σ, 100 = 10σ. `0` disables. |
| `qTh` | float | 1e-3 | rad²/s | Process noise on the common phase `theta` |
| `qD` | float | 1e-5 | rad²/s | Process noise on the formation errors `delta_k`, `delta_j`. **Keep well below `qTh`** — see below. |
| `qW` | float | 1e-4 | rad²/s³ | Process noise on the rate error `omega_e`. Small: it models a slowly varying bias, not fast dynamics. |
| `qR` | float | 1e-5 | m²/s | Process noise on the radius `r`. Small unless the radius is being commanded to change. |
| `pTh` | float | 0.25 | rad² | Initial variance of `theta` (one sigma ≈ 0.5 rad, 29°) |
| `pD` | float | 0.50 | rad² | Initial variance of the formation errors (one sigma ≈ 0.71 rad, 40°). **Should be large** even though `qD` is small. |
| `pW` | float | 0.25 | (rad/s)² | Initial variance of `omega_e` (one sigma 0.5 rad/s) |
| `pR` | float | 0.09 | m² | Initial variance of `r` (one sigma 0.3 m) |

The `q*` entries are variances added **per second**: `qTh = 1e-3` lets the phase
wander with a standard deviation of about `sqrt(1e-3) ≈ 0.03` rad after one
second of dead reckoning. Larger means the filter trusts its own prediction less
and the measurements more — faster to react, noisier at rest.

The `p*` entries apply **only at initialisation**. Change one, then trigger
`rst` for it to take effect. A non-positive or absurd value is ignored and a
safe default substituted: a zero variance would claim perfect knowledge of the
seed, and the filter would then reject every measurement that disagreed with
it — silently, and permanently.

Three things are easy to get wrong:

* **`n` silently corrupts every separation if it is wrong.** The formation
  errors absorb the mismatch as a constant offset, so the estimate looks healthy
  and every separation is wrong by the same amount. Change it, then set `rst`.
* **`qD` must stay well below `qTh`.** This is the entire reason for the
  common/differential coordinates: wind, a mis-scaled rate command and a slow
  control loop all displace the formation *as a whole*, leaving the spacing
  comparatively undisturbed. Setting them comparable throws the benefit away.
* **`pD` should be large even though `qD` is small.** The formation errors are
  genuinely unknown at startup — they are seeded to zero and only the chords can
  reveal them — but they drift slowly thereafter.

The `p*` parameters take effect **only at initialisation**. Change them, then
trigger `rst`.

### Range slots

The filter reads ranges from the hybrid-mode logging slots, using this
convention (matching `tools/swarm/setup_swarm.py`):

| Slot | Contents |
|------|----------|
| `tdoa3.hmLId0..2` | the three template anchors |
| `tdoa3.hmLId3` | leader (next agent, counter-clockwise) |
| `tdoa3.hmLId4` | follower (previous agent) |

Each vehicle also needs a unique `tdoa3.hmId` (never 255, and never colliding
with an anchor id), and `tdoa3.hmTwrTXPos` must be **0** — see
[Safety notes](#safety-notes).

## Configuring from the ground

### With cflib

`tools/swarm/setup_swarm.py` configures the whole swarm in one pass, assigning
ids and the leader/follower ring automatically:

```bash
python3 tools/swarm/setup_swarm.py \
    radio://0/80/2M/E7E7E7E718=254 \
    radio://0/80/2M/E7E7E7E719=253 \
    radio://0/80/2M/E7E7E7E71A=252
```

All `tdoa3` and `swarm*` parameters are **volatile** — rerun after any reboot.

### With Crazyswarm2

Crazyswarm2 reapplies parameters on connect, which removes the volatility
problem. Custom log topics and per-robot parameter overrides are configured in
`crazyflies.yaml`:

```yaml
all:
  firmware_params:

    # --- trajectory -------------------------------------------------------
    # Coefficients come from tools/swarm/fit_elevation.py. The values below
    # are the "saddle" example: b(theta) = -20 deg * cos(2 theta).
    swarmCurve:
      cx: 0.0              # encirclement centre X [m]
      cy: 0.0              # encirclement centre Y [m]
      cz: 1.0              # centre altitude above the anchor plane [m], > 0
      K: 2                 # harmonic order in use, 0..4 (0 = flat circle)
      b0: 0.0              # elevation profile, constant term [rad]
      a1: 0.0              # cos(theta)   [rad]
      b1: 0.0              # sin(theta)   [rad]
      a2: -0.349066        # cos(2 theta) [rad]
      b2: 0.0              # sin(2 theta) [rad]
      a3: 0.0              # cos(3 theta) [rad]
      b3: 0.0              # sin(3 theta) [rad]
      a4: 0.0              # cos(4 theta) [rad]
      b4: 0.0              # sin(4 theta) [rad]

    # --- estimator --------------------------------------------------------
    swarmFilter:
      en: 1                # run the filter
      rst: 0               # write 1 to force re-seeding; self-clearing
      n: 3                 # agents in the ring -> nominal spacing 2*pi/n
      wz: 0.0              # commanded angular rate [rad/s]

      sigA: 0.03           # anchor range noise, one sigma [m]
      sigC: 0.03           # chord noise, one sigma [m]
      gate: 25.0           # innovation gate, sigma SQUARED (25 = 5 sigma)

      qTh: 0.001           # process noise, common phase [rad^2/s]
      qD: 0.00001          # process noise, formation errors [rad^2/s] -- keep << qTh
      qW: 0.0001           # process noise, rate error [rad^2/s^3]
      qR: 0.00001          # process noise, radius [m^2/s]

      pTh: 0.25            # initial variance, phase [rad^2]
      pD: 0.50             # initial variance, formation errors [rad^2] -- keep large
      pW: 0.25             # initial variance, rate error [(rad/s)^2]
      pR: 0.09             # initial variance, radius [m^2]

    # --- ranging ----------------------------------------------------------
    tdoa3:
      hmTwr: 1             # transmit TWR packets
      hmTwrEstPos: 1       # feed anchor ranges to the onboard estimator
      hmTwrTXPos: 0        # do NOT broadcast own position -- see Safety notes
      hmOutTh: 0.5         # median outlier filter on ranges [m]
      hmLId0: 0            # slot 0 -> anchor A0
      hmLId1: 1            # slot 1 -> anchor A1
      hmLId2: 2            # slot 2 -> anchor A2

  firmware_logging:
    enabled: true
    custom_topics:

      # State vector, SI. Order: th, dk, dj, we, r, seq
      swarm_state:
        frequency: 50
        vars: ["swarmFilter.th", "swarmFilter.dk", "swarmFilter.dj",
               "swarmFilter.we", "swarmFilter.r", "swarmFilter.seq"]

      # Variances, in the SAME ORDER as swarm_state
      swarm_cov:
        frequency: 50
        vars: ["swarmCov.th", "swarmCov.dk", "swarmCov.dj",
               "swarmCov.we", "swarmCov.r", "swarmCov.seq"]

      # Separations the controller consumes [rad]
      swarm_sep:
        frequency: 50
        vars: ["swarmFilter.phiK", "swarmFilter.phiJ"]

      # Health, at a lower rate
      swarm_health:
        frequency: 10
        vars: ["swarmDiagnostics.dTh", "swarmDiagnostics.nis",
               "swarmDiagnostics.nAcc", "swarmDiagnostics.nRej"]

robots:
  cf254:
    uri: radio://0/80/2M/E7E7E7E718
    type: cf21bl
    firmware_params:
      tdoa3:
        hmId: 254
        hmLId3: 253        # leader
        hmLId4: 252        # follower
```

Every parameter of both groups is listed above, at its firmware default except
`K` and `a2`. Values equal to the default can be omitted; they are spelled out
here so the file doubles as a reference, and so that a value someone changed by
hand in cfclient is restored on the next connect.

`rst` is a self-clearing trigger rather than a setting. Listing it as `0` is
harmless; to force a re-seed, set it to 1 from the command line or client, not
from this file.

Three practical points:

* A custom topic publishes `crazyflie_interfaces/msg/LogDataGeneric`, which
  carries a **bare float array** in the order the variables are listed — there
  are no names in the message. Load the same YAML in the subscribing node and
  index by name rather than hard-coding offsets, otherwise inserting a variable
  silently shifts every reading.
* Keep each topic to **six floats or fewer**; the CRTP log payload is 26 bytes.
  The five-float `swarm_state` and `swarm_cov` topics use 20 bytes each.
* `swarm_state` and `swarm_cov` are index-aligned by construction, so a
  subscriber can zip them into a state-with-covariance message directly.

## Monitoring the filter

Everything needed is in `swarmDiagnostics`.

| Log | What it tells you |
|-----|-------------------|
| `dTh` | Filter phase minus position-derived phase [deg] — has the filter diverged? |
| `nis` | Normalised innovation squared — is the filter correctly tuned? |
| `sTh`, `sDk`, `sDj`, `sWe`, `sR` | Current one-sigma uncertainty: phase [deg], formation errors [deg], rate error [rad/s], radius [m] |
| `nAcc`, `nRej` | Accepted and gate-rejected measurement counts |
| `init` | 1 once the filter has been seeded |

### Is it right? — `dTh`

The cross-check. Expect a few degrees of disagreement, since the
position-derived value inherits whatever error the onboard position estimate
has. A **persistent** value beyond about 20 degrees means the filter has
diverged or picked the wrong branch of the chord ambiguity; force a re-seed with
`swarmFilter.rst`, which always returns to the counter-clockwise-ordered branch.

### Is it tuned? — `nis`

For a correctly tuned filter, `nis` is chi-squared with one degree of freedom
and should **average about 1**.

| Observation | Meaning | Action |
|-------------|---------|--------|
| `nis` >> 1 persistently | Overconfident | Raise `sigA` / `sigC`, or look for a bias: antenna delay, or an anchor position that does not match reality |
| `nis` << 1 persistently | Over-conservative | Lower `sigA` / `sigC`; the estimate is sluggish for no reason |
| `nRej` a large fraction of `nAcc` | Gate firing too often | `gate` too tight, or the noise parameters too small |

Tune against this number rather than by eye.

### Is it converging? — `sTh`, `sD`

After a reset, `sTh` should fall from the seeded `pTh` and then hold steady.

* **`sTh` collapsing toward zero while the estimate is still wrong** is the
  signature of an overconfident filter — measurements being counted more than
  once, or a measurement noise far below the truth.
* **`sD` staying wide** means the chords are not reaching the filter. Check that
  the leader and follower slots are populated and their ranges fresh, using
  `tools/swarm/hybrid_ranging.py`.

### Pairing state with covariance

The two are **separate log blocks**, and the firmware drives each block from its
own FreeRTOS timer. They are therefore *not* sampled atomically: nothing forces
two blocks of equal period to fire on the same tick, and they arrive as two
independent ROS messages.

Two mechanisms make this tractable:

* **Every log packet carries a firmware timestamp** in milliseconds, taken when
  the block was sampled, and Crazyswarm2 exposes it on the message. Matching on
  it is exact, and better than assuming the two topics line up.
* **Both blocks publish `seq`**, incremented once per filter cycle. When
  `swarm_state.seq == swarm_cov.seq`, the state and its covariance came from the
  same cycle. This is the cheaper check and it is what the example YAML above
  uses the sixth slot for.

Without either, the worst case is a skew of one filter cycle — 20 ms at the
default 50 Hz, about one degree of phase at 1 rad/s. Small, but it is a genuine
inconsistency rather than noise, and it costs one float to remove.

> A subtler point: the log task and the app task run at the **same priority**
> (`LOG_TASK_PRI` and `CONFIG_APP_PRIORITY` are both 1), so a block can in
> principle be sampled midway through the app writing its outputs, mixing fields
> from two adjacent cycles. `seq` is written last, so a reader that sees a new
> `seq` has seen the whole update.

### The covariance itself — `swarmCov`

The `sTh`/`sDk`/`sDj`/`sWe`/`sR` entries above are the readable view; `swarmCov` is the same information
as raw `P` entries in SI units, for code rather than eyes:

| Log | State | Unit |
|-----|-------|------|
| `swarmCov.th` | phase | rad² |
| `swarmCov.dk`, `swarmCov.dj` | formation errors | rad² |
| `swarmCov.we` | rate error | (rad/s)² |
| `swarmCov.r` | radius | m² |
| `swarmCov.seq` | — | matches `swarmFilter.seq` |

> **Diagonal only.** The off-diagonal entries of `P` are genuinely non-zero —
> the strongest is phase against rate error, which the prediction step creates
> directly, and the chord updates correlate the formation errors with the
> radius. Treating `swarmCov` as a full covariance will understate those
> correlations. The complete symmetric `P` is 15 unique elements and would need
> three log blocks; it is not published by default.

### A good first test

With the vehicles stationary and `swarmFilter.wz = 0`:

1. `swarmDiagnostics.init` reaches 1.
2. `dTh` settles near zero.
3. `nis` hovers around 1.
4. `nRej` stays small relative to `nAcc`.
5. `sTh` falls and then holds.

That exercises the whole chain — ranging, slot mapping, curve evaluation and the
filter — before anything moves.

## Tools

| Tool | Purpose |
|------|---------|
| `tools/swarm/scan_anchors.py` | Discover which anchor ids are actually ranging, and detect stale readings |
| `tools/swarm/hybrid_ranging.py` | Log all range slots simultaneously with per-remote statistics |
| `tools/swarm/setup_swarm.py` | Configure the whole swarm: ids, slots, leader/follower ring |
| `tools/swarm/fit_elevation.py` | Fit an elevation profile from waypoints; produces the `swarmCurve` coefficients and plots |
| `tools/swarm/phase_validation.py` | Compare measured chords against positions, and validate the chord model |
| `tools/swarm/check_swarm_app.py` | Verify the onboard phase computation against the same maths offboard |

Two host unit tests run without hardware:

```bash
cd examples/app_swarm_encircle
gcc -DSWARM_CURVE_HOST_TEST -Isrc -o /tmp/tc test/test_swarm_curve.c src/swarm_curve.c -lm && /tmp/tc
gcc -DSWARM_CURVE_HOST_TEST -Isrc -o /tmp/te test/test_swarm_ekf.c src/swarm_ekf.c src/swarm_curve.c -lm && /tmp/te
```

They check the curve against an independent reference and the filter against
closed-loop simulations with known truth, including that the covariance
propagation matches an explicit `F P F'`.

## Safety notes

**Keep `tdoa3.hmTwrTXPos` at 0.** Setting it to 1 makes a Crazyflie broadcast its
own position estimate, and its neighbours then treat it as a positioning anchor.
Each vehicle would consume its neighbours' estimate errors and feed its own back
— circular error propagation, which the architecture otherwise avoids
structurally. Chord ranging does **not** require it: the distance is computed
before any position is consulted.

**The filter does not fly the vehicle.** It runs alongside the onboard state
estimator for coordination only. Position, velocity and attitude for the flight
controller still come from the vehicle's own EKF, so a formation cannot be
flown on a position estimate that will not hold a stable hover.

**Do not feed `q(theta, r)` back into the state estimator as a position
measurement.** That constrains the position estimate onto the reference
trajectory and makes tracking error partly invisible to the controller — the
estimator would report "on trajectory" while the vehicle is not.

## Known limitations

* The chord is even in the phase difference, so it cannot by itself distinguish
  a neighbour ahead from one behind. The branch is selected at initialisation by
  the counter-clockwise ordering convention and maintained by prediction
  continuity.
* Chord sensitivity is proportional to `cos(separation/2)` and collapses as the
  separation approaches 180 degrees, so near-diametric pairings — notably `n = 2`
  — lose separation observability.
* The elevation profile must satisfy `|b| < 90` degrees, and the design family
  cannot represent a curve that reverses in azimuth or varies its distance from
  the encirclement centre.
* Ranging accuracy scales with the ratio of operating distance to anchor
  baseline, so a small anchor template multiplies the ranging noise. Aim for a
  baseline comparable to the encirclement radius.
