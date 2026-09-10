#!/usr/bin/env python3
"""
Fit an elevation profile b(theta) to 3D waypoints, offline.

Usage:
    python3 tools/swarm/fit_elevation.py [--no-plot] [--out PREFIX]

Everything expensive happens here; the firmware only ever EVALUATES

    b(theta) = beta0 + sum_k [ alpha_k cos(k theta) + beta_k sin(k theta) ]
    q(theta) = [ r cos b cos theta,  r cos b sin theta,  h - r sin b ]

which is a handful of cos/sin calls. No matrix exponential, no least squares,
and no linear algebra of any kind on the vehicle.

Pipeline (theory.tex, "Designing an elevation profile from waypoints")
----------------------------------------------------------------------
1. Project each waypoint onto the reachable family:

       theta_i = atan2(w_y - c_y, w_x - c_x)
       b_i     = -arcsin( (w_z - c_z) / ||w_i - c|| )

   The direction of the waypoint from the encirclement centre is matched
   EXACTLY; its distance is discarded. The leftover is the "radial slip",
   | ||w_i - c|| - r |, reported below so it is known before the vehicle flies
   rather than discovered in the air.

2. Fit the truncated Fourier series by linear least squares, sweeping the
   harmonic order K upward and stopping at the first acceptable fit. Low K is
   preferred: the cost of harmonic order is steep.

3. Acceptance is checked on a DENSE grid, not at the waypoints. A trigonometric
   interpolant overshoots between its data, and it is the overshoot that
   grounds the vehicle. The |b| < pi/2 pole guard is checked there too.

Sign convention: b > 0 is BELOW the encirclement centre, since q_z = h - r sin b.
"""

import argparse
import math

import numpy as np

# ----------------------------------------------------------------------------
# Geometry. The encirclement centre sits at altitude h above the anchor plane;
# it is NOT the corner anchor (theory.tex, Remark "the centre is not the corner").
# ----------------------------------------------------------------------------
CENTRE = np.array([0.0, 0.0, 1.00])   # c = h * e_z
RADIUS = 1.00                         # r

# Acceptance thresholds
TAU_DEG = 1.0       # waypoint RMS residual accepted, degrees
B_LIMIT_DEG = 80.0  # dense-grid |b| ceiling; the hard pole is 90 degrees
K_MAX = 8

# ----------------------------------------------------------------------------
# Waypoint sets. Leave exactly ONE uncommented.
# Each row is a sketched 3D point [x, y, z] in the anchor frame.
# ----------------------------------------------------------------------------

# --- (A) Saddle: two lobes up, two down. The classic "Pringle". Needs K = 2.
#         Two waypoints are deliberately off-radius so the radial-slip report
#         has something to show.
# WAYPOINTS = np.array([
#     [ 0.940,  0.000,  1.342],   # theta   0 deg, b = -20 deg  (above centre)
#     [ 0.000,  0.996,  0.637],   # theta  90 deg, b = +20 deg, radius 1.06
#     [-0.940,  0.000,  1.342],   # theta 180 deg, b = -20 deg
#     [ 0.000, -0.883,  0.679],   # theta 270 deg, b = +20 deg, radius 0.94
# ])
# CURVE_NAME = 'A: saddle (2 lobes)'

# --- (B) Three-lobed crown: alternating up/down every 60 deg. Needs K = 3.
WAYPOINTS = np.array([
    [ 0.906,  0.000,  1.423],   # theta   0 deg, b = -25 deg
    [ 0.453,  0.785,  0.577],   # theta  60 deg, b = +25 deg
    [-0.453,  0.785,  1.423],   # theta 120 deg, b = -25 deg
    [-0.906,  0.000,  0.577],   # theta 180 deg, b = +25 deg
    [-0.453, -0.785,  1.423],   # theta 240 deg, b = -25 deg
    [ 0.453, -0.785,  0.577],   # theta 300 deg, b = +25 deg
])
CURVE_NAME = 'B: three-lobed crown'

# --- (C) Asymmetric sweep: no symmetry to exploit, so it needs a higher K and
#         shows the overshoot behaviour clearly.
# WAYPOINTS = np.array([
#     [ 0.866,  0.000,  1.500],   # theta   0 deg, b = -30 deg
#     [ 0.304,  0.937,  1.174],   # theta  72 deg, b = -10 deg
#     [-0.781,  0.568,  0.741],   # theta 144 deg, b = +15 deg
#     [-0.760, -0.552,  0.658],   # theta 216 deg, b = +20 deg
#     [ 0.308, -0.947,  1.087],   # theta 288 deg, b =  -5 deg
# ])
# CURVE_NAME = 'C: asymmetric sweep'


# ----------------------------------------------------------------------------
# Core maths
# ----------------------------------------------------------------------------

def project(waypoints, centre, radius):
    """Project 3D waypoints onto the reachable family -> (theta, b, slip)."""
    d = waypoints - centre
    rho = np.linalg.norm(d, axis=1)
    if np.any(rho < 1e-9):
        raise SystemExit('a waypoint coincides with the encirclement centre')

    theta = np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi)
    b = -np.arcsin(np.clip(d[:, 2] / rho, -1.0, 1.0))
    slip = np.abs(rho - radius)
    return theta, b, slip, rho


def design_matrix(theta, K):
    """M = [1, cos t, sin t, ..., cos Kt, sin Kt]."""
    cols = [np.ones_like(theta)]
    for k in range(1, K + 1):
        cols.append(np.cos(k * theta))
        cols.append(np.sin(k * theta))
    return np.column_stack(cols)


def fit(theta, b, K):
    """Least-squares Fourier coefficients; exact interpolation when 2K+1 >= N."""
    M = design_matrix(theta, K)
    coeffs, *_ = np.linalg.lstsq(M, b, rcond=None)
    return coeffs


def eval_b(coeffs, theta):
    """Evaluate b(theta). This is the ONLY part the firmware reproduces."""
    theta = np.atleast_1d(theta)
    out = np.full_like(theta, coeffs[0], dtype=float)
    K = (len(coeffs) - 1) // 2
    for k in range(1, K + 1):
        out += coeffs[2 * k - 1] * np.cos(k * theta)
        out += coeffs[2 * k] * np.sin(k * theta)
    return out


def eval_q(coeffs, theta, centre, radius):
    """Normal form: spherical coordinates, azimuth theta, elevation -b."""
    b = eval_b(coeffs, theta)
    cb, sb = np.cos(b), np.sin(b)
    return np.column_stack([
        radius * cb * np.cos(theta),
        radius * cb * np.sin(theta),
        centre[2] - radius * sb,
    ]) + np.array([centre[0], centre[1], 0.0])


def sweep(theta, b, tau_deg, b_limit_deg, k_max, dense):
    """Increase K until the fit is acceptable ON THE DENSE GRID."""
    tau = math.radians(tau_deg)
    b_limit = math.radians(b_limit_deg)

    rows, accepted = [], None
    for K in range(0, k_max + 1):
        coeffs = fit(theta, b, K)
        resid = eval_b(coeffs, theta) - b
        rms = float(np.sqrt(np.mean(resid ** 2)))
        b_dense = eval_b(coeffs, dense)
        max_dense = float(np.max(np.abs(b_dense)))

        ok = (rms <= tau) and (max_dense < b_limit)
        rows.append((K, 2 * K + 1, rms, max_dense, ok))
        if ok and accepted is None:
            accepted = (K, coeffs)
    return rows, accepted


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def report_projection(theta, b, slip, rho):
    print(f'--- waypoints projected onto the design family ---')
    print(f'{"#":>2} {"theta [deg]":>12} {"b [deg]":>9} {"|w-c| [m]":>10} {"radial slip":>12}')
    print('-' * 50)
    for i, (t, bb, s, rr) in enumerate(zip(theta, b, slip, rho)):
        print(f'{i:>2} {math.degrees(t):>12.1f} {math.degrees(bb):>9.1f} '
              f'{rr:>10.3f} {s:>12.3f}')
    print()
    print(f'  max radial slip {slip.max():.3f} m  '
          f'(direction is matched exactly; distance is discarded)')
    if slip.max() > 0.15:
        print('  NOTE: large slip -- these waypoints are not near the sphere of')
        print(f'        radius {RADIUS:.2f} m about the centre. Consider adjusting r.')
    print()


def report_sweep(rows, accepted, b):
    print('--- harmonic order sweep (acceptance on a dense grid) ---')
    print(f'{"K":>3} {"coeffs":>7} {"wp RMS [deg]":>13} {"max|b| dense [deg]":>20} {"ok":>4}')
    print('-' * 52)
    for K, ncoef, rms, mx, ok in rows:
        print(f'{K:>3} {ncoef:>7} {math.degrees(rms):>13.3f} '
              f'{math.degrees(mx):>20.2f} {"yes" if ok else "":>4}')
    print()

    if accepted is None:
        raise SystemExit(
            f'no acceptable fit up to K={K_MAX}.\n'
            '  Either loosen TAU_DEG, or the waypoints are not well represented\n'
            '  by a graph over azimuth (check for near-duplicate azimuths).')

    K, coeffs = accepted
    max_wp = float(np.max(np.abs(b)))
    dense = np.linspace(0, 2 * np.pi, 2001)
    max_dense = float(np.max(np.abs(eval_b(coeffs, dense))))
    print(f'  accepted K = {K}  ({2*K+1} coefficients)')
    print(f'  max |b| at waypoints {math.degrees(max_wp):.1f} deg, '
          f'on dense grid {math.degrees(max_dense):.1f} deg')
    over = math.degrees(max_dense - max_wp)
    if over > 0.5:
        print(f'  OVERSHOOT {over:.1f} deg between waypoints -- this is the value')
        print('  that matters for clearance, not the waypoint maximum.')
    print()


def report_coeffs(K, coeffs):
    names = ['b0'] + [f'{c}{k}' for k in range(1, K + 1) for c in ('a', 'b')]
    print('--- firmware coefficients ---')
    for n, v in zip(names, coeffs):
        print(f'    swarm.{n:<4} = {v:+.6f}')
    print()
    print('  cflib:')
    print('    PARAMS = {')
    print(f"        'swarm.K': {K},")
    for n, v in zip(names, coeffs):
        print(f"        'swarm.{n}': {v:+.6f},")
    print('    }')
    print()
    print('  crazyswarm2 crazyflies.yaml:')
    print('    all:')
    print('      firmware_params:')
    print('        swarm:')
    print(f'          K: {K}')
    for n, v in zip(names, coeffs):
        print(f'          {n}: {v:+.6f}')
    print()


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def make_plots(theta, b, coeffs, K, prefix):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        print(f'(plots skipped: {e})')
        return

    dense = np.linspace(0, 2 * np.pi, 721)
    b_dense = eval_b(coeffs, dense)
    q = eval_q(coeffs, dense, CENTRE, RADIUS)
    flat = eval_q(np.array([0.0]), dense, CENTRE, RADIUS)

    fig = plt.figure(figsize=(13, 10))
    fig.suptitle(f'Elevation profile fit  --  {CURVE_NAME}  (K = {K})',
                 fontsize=13, fontweight='bold')

    # (1) elevation profile ---------------------------------------------------
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(np.degrees(dense), np.degrees(b_dense), lw=2, color='#1f77b4',
            label=f'fitted b(theta), K={K}')
    ax.plot(np.degrees(theta), np.degrees(b), 'o', ms=8, color='#d62728',
            zorder=5, label='waypoints')
    ax.axhline(0, color='k', lw=0.8, alpha=0.4)
    for s in (+1, -1):
        ax.axhline(s * B_LIMIT_DEG, color='#ff7f0e', ls='--', lw=1,
                   label='pole guard' if s > 0 else None)
    mx = np.degrees(np.max(np.abs(b_dense)))
    ax.axhline(mx, color='#2ca02c', ls=':', lw=1.2,
               label=f'max|b| = {mx:.1f} deg')
    ax.axhline(-mx, color='#2ca02c', ls=':', lw=1.2)
    ax.set_xlabel('phase theta [deg]')
    ax.set_ylabel('elevation b [deg]   (positive = below centre)')
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 45))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    ax.set_title('elevation profile (overshoot is between the waypoints)',
                 fontsize=10)

    # (2) 3D trajectory -------------------------------------------------------
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    # faint sphere: the curve provably lies on it (norm(q - c) = r)
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_wireframe(CENTRE[0] + RADIUS * np.cos(u) * np.sin(v),
                      CENTRE[1] + RADIUS * np.sin(u) * np.sin(v),
                      CENTRE[2] + RADIUS * np.cos(v),
                      color='k', alpha=0.06, lw=0.5)
    ax.plot(flat[:, 0], flat[:, 1], flat[:, 2], color='#7f7f7f', ls='--',
            lw=1.2, label='flat circle (b=0)')
    ax.plot(q[:, 0], q[:, 1], q[:, 2], lw=2.5, color='#1f77b4',
            label='embedded curve')
    ax.scatter(WAYPOINTS[:, 0], WAYPOINTS[:, 1], WAYPOINTS[:, 2],
               s=60, color='#d62728', depthshade=False, zorder=5,
               label='waypoints')
    ax.scatter(*CENTRE, s=50, marker='x', color='k', label='centre c')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
    ax.set_box_aspect((1, 1, 1))
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('trajectory on the sphere about c', fontsize=10)

    # (3) top view: azimuth is preserved exactly -------------------------------
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(flat[:, 0], flat[:, 1], color='#7f7f7f', ls='--', lw=1.2,
            label='flat circle')
    ax.plot(q[:, 0], q[:, 1], lw=2, color='#1f77b4', label='curve (top view)')
    for wp, t in zip(WAYPOINTS, theta):
        ax.plot([CENTRE[0], CENTRE[0] + 1.35 * RADIUS * np.cos(t)],
                [CENTRE[1], CENTRE[1] + 1.35 * RADIUS * np.sin(t)],
                color='#d62728', lw=0.8, alpha=0.55)
    ax.plot(WAYPOINTS[:, 0], WAYPOINTS[:, 1], 'o', ms=8, color='#d62728',
            zorder=5, label='waypoint shadows')
    ax.plot(CENTRE[0], CENTRE[1], 'kx', ms=9)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    ax.set_title('shadows land on their rays: azimuth = phase, exactly',
                 fontsize=10)

    # (4) harmonic order cost -------------------------------------------------
    ax = fig.add_subplot(2, 2, 4)
    dense_g = np.linspace(0, 2 * np.pi, 2001)
    Ks, rmss = [], []
    for k in range(0, K_MAX + 1):
        c = fit(theta, b, k)
        Ks.append(k)
        rmss.append(math.degrees(float(np.sqrt(np.mean((eval_b(c, theta) - b) ** 2)))))
    ax.semilogy(Ks, np.maximum(rmss, 1e-12), 'o-', color='#1f77b4')
    ax.axhline(TAU_DEG, color='#d62728', ls='--', lw=1.2,
               label=f'tolerance {TAU_DEG} deg')
    ax.axvline(K, color='#2ca02c', ls=':', lw=1.5, label=f'accepted K = {K}')
    ax.set_xlabel('harmonic order K')
    ax.set_ylabel('waypoint RMS residual [deg]')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)
    ax.set_title('smallest adequate K is the right one', fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = f'{prefix}.png'
    fig.savefig(out, dpi=140)
    print(f'--- plots written to {out} ---')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--out', default='elevation_fit')
    args = ap.parse_args()

    print(f'curve: {CURVE_NAME}')
    print(f'centre c = {CENTRE.tolist()},  radius r = {RADIUS} m')
    print(f'{len(WAYPOINTS)} waypoints\n')

    theta, b, slip, rho = project(WAYPOINTS, CENTRE, RADIUS)
    report_projection(theta, b, slip, rho)

    dense = np.linspace(0, 2 * np.pi, 2001)
    rows, accepted = sweep(theta, b, TAU_DEG, B_LIMIT_DEG, K_MAX, dense)
    report_sweep(rows, accepted, b)

    K, coeffs = accepted
    report_coeffs(K, coeffs)

    if not args.no_plot:
        make_plots(theta, b, coeffs, K, args.out)


if __name__ == '__main__':
    main()
