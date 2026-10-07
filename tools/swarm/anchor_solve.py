#!/usr/bin/env python3
"""
Build anchor coordinates for N anchors from tape measurements only.

Usage
-----
    python3 tools/swarm/anchor_solve.py \
        --heights 0.19 0.19 0.49 0.20 0.16 \
        --dist 0-1:0.620 0-2:0.810 0-3:0.445 0-4:0.772 \
               1-2:0.898 1-3:0.430 1-4:0.572 \
               2-3:0.440 2-4:0.531 3-4:0.337

Why distances rather than coordinates
-------------------------------------
Coordinates have to be derived from something, and on a compact template the
only thing you can measure well is a length. A tape at 2-3 mm beats every other
source available, including the UWB itself. So measure lengths, and let this
tool turn them into the coordinates the anchors must broadcast.

The shape of a point set is fully determined by its pairwise distances, up to a
rigid motion and a reflection. This tool pins those down as:

    A0 at the origin
    A1 on the +x axis (horizontally)
    A2 in the  y > 0  half-space
    z measured UP from A0

Why heights are measured separately
-----------------------------------
Distances alone leave the constellation free to rotate, which would tilt the
frame. That is not acceptable: the EKF fuses an accelerometer that knows which
way gravity points, and `swarmCurve.cz` has to mean altitude. So z comes from
direct height measurements and only x,y are solved from distances. Heights are
the easy measurement anyway -- a tape against a wall, or a spirit level and a
rule.

Redundancy
----------
N anchors have 2N horizontal coordinates, minus 3 rigid degrees of freedom
(two translations and a rotation), so 2N-3 are intrinsic. With all N(N-1)/2
pairwise distances the problem is OVER-determined for N >= 4, and the residual
is a real check on the tape work rather than a formality. For N = 3 the
residual is identically zero and proves nothing.
"""

import argparse
import itertools
import math
import sys

import numpy as np


def parse_dist(entries, n):
    """Parse "i-j:len" entries into a dict {(i,j): length}."""
    out = {}
    for e in entries:
        try:
            pair, val = e.split(':')
            i, j = (int(x) for x in pair.split('-'))
            d = float(val)
        except ValueError:
            sys.exit(f'ERROR: cannot parse --dist entry "{e}". '
                     'Expected  i-j:length  e.g.  0-1:0.620')
        if not (0 <= i < n and 0 <= j < n):
            sys.exit(f'ERROR: --dist entry "{e}" refers to an anchor outside '
                     f'0..{n-1} (you gave {n} heights)')
        if i == j:
            sys.exit(f'ERROR: --dist entry "{e}" is a self-distance')
        if d <= 0:
            sys.exit(f'ERROR: --dist entry "{e}" is not positive')
        out[(min(i, j), max(i, j))] = d
    return out


def horizontal(d_slant, dz, label):
    """Project a slant measurement onto the horizontal plane."""
    if d_slant < abs(dz) - 1e-9:
        sys.exit(f'ERROR: {label} = {d_slant:.4f} m is shorter than its height '
                 f'difference {abs(dz):.4f} m -- impossible. Check the tape or '
                 'the heights.')
    return math.sqrt(max(0.0, d_slant * d_slant - dz * dz))


def mds_2d(H, present):
    """Classical MDS on a (possibly partial) horizontal distance matrix.

    Returns an n x 2 configuration centred on the centroid. Missing entries are
    filled by shortest-path (metric) completion first, which is a standard and
    well-behaved initialisation; the refinement step afterwards uses only the
    distances actually measured.
    """
    n = H.shape[0]
    D = H.copy()
    # Floyd-Warshall metric completion for any unmeasured pair.
    big = D[present].max() * n if present.any() else 1.0
    D[~present] = big
    np.fill_diagonal(D, 0.0)
    for k in range(n):
        D = np.minimum(D, D[:, [k]] + D[[k], :])

    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:2]
    w2 = np.clip(w[idx], 0.0, None)
    return V[:, idx] * np.sqrt(w2)


def refine(X, H, present, fixed_z):
    """Gauss-Newton on the measured horizontal distances only."""
    n = X.shape[0]
    pairs = [(i, j) for i, j in itertools.combinations(range(n), 2)
             if present[i, j]]
    for _ in range(200):
        r, rows = [], []
        for i, j in pairs:
            v = X[i] - X[j]
            d = np.linalg.norm(v)
            if d < 1e-9:
                d, v = 1e-9, np.array([1e-9, 0.0])
            e = v / d
            r.append(H[i, j] - d)
            row = np.zeros(2 * n)
            row[2*i:2*i+2] = e
            row[2*j:2*j+2] = -e
            rows.append(row)
        J = np.array(rows)
        r = np.array(r)
        # Gauge fixing: A0 pinned, A1 pinned in y. Removes the 3 rigid DOF.
        gauge = np.zeros((3, 2 * n))
        gauge[0, 0] = 1.0      # A0.x
        gauge[1, 1] = 1.0      # A0.y
        gauge[2, 3] = 1.0      # A1.y
        J = np.vstack([J, gauge * 1e3])
        r = np.concatenate([r, np.zeros(3)])
        step, *_ = np.linalg.lstsq(J, r, rcond=None)
        X = X + step.reshape(n, 2)
        if np.abs(step).max() < 1e-12:
            break
    return X


def main():
    ap = argparse.ArgumentParser(
        description='Solve anchor coordinates from pairwise tape measurements.')
    ap.add_argument('--heights', type=float, nargs='+', required=True,
                    help='height of each anchor above A0 [m], in id order. '
                         'A0 itself is 0.0 by definition of the frame.')
    ap.add_argument('--dist', nargs='+', required=True,
                    help='pairwise SLANT distances as i-j:length, e.g. 0-1:0.620')
    ap.add_argument('--radius', type=float, default=None,
                    help='planned orbit radius [m], for a geometry check')
    ap.add_argument('--sigma', type=float, default=0.02,
                    help='UWB range noise, one sigma [m]')
    args = ap.parse_args()

    n = len(args.heights)
    if n < 3:
        sys.exit('ERROR: need at least 3 anchors')
    z = np.array(args.heights, float)
    z = z - z[0]                       # frame origin at A0
    meas = parse_dist(args.dist, n)

    H = np.zeros((n, n))
    present = np.zeros((n, n), bool)
    for (i, j), d in meas.items():
        h = horizontal(d, z[i] - z[j], f'A{i}-A{j}')
        H[i, j] = H[j, i] = h
        present[i, j] = present[j, i] = True

    npairs = len(meas)
    intrinsic = 2 * n - 3
    dof = npairs - intrinsic
    print(f'--- {n} anchors, {npairs} of {n*(n-1)//2} pairwise distances ---')
    print(f'  intrinsic horizontal DOF {intrinsic},  redundancy {dof}')
    if dof < 0:
        sys.exit(f'ERROR: under-determined. Need at least {intrinsic} distances; '
                 f'you gave {npairs}.')
    if dof == 0:
        print('  WARNING: exactly determined -- the residual below will be zero')
        print('  whatever the tape said, so it validates nothing. Measure more.')
    print()

    # Connectivity: every anchor needs at least two measured distances to be
    # placed in the plane at all.
    deg = present.sum(axis=1)
    for i in range(n):
        if deg[i] < 2:
            sys.exit(f'ERROR: A{i} has only {deg[i]} measured distance(s). '
                     'Each anchor needs at least two to be located.')

    X = refine(mds_2d(H, present), H, present, z)

    # --- put into the documented frame -----------------------------------
    X = X - X[0]                                        # A0 at origin
    th = math.atan2(X[1, 1], X[1, 0])                   # A1 onto +x
    R = np.array([[math.cos(-th), -math.sin(-th)],
                  [math.sin(-th),  math.cos(-th)]])
    X = X @ R.T
    if X[2, 1] < 0:                                     # A2 into y > 0
        X[:, 1] = -X[:, 1]

    print('--- anchor positions, template frame ---')
    print(f'{"id":>4} {"x [m]":>10} {"y [m]":>10} {"z [m]":>10}   role')
    print('-' * 52)
    roles = {0: 'origin', 1: 'defines +x', 2: 'defines +y'}
    P = np.column_stack([X, z])
    for i in range(n):
        print(f'{i:>4} {P[i,0]:>10.4f} {P[i,1]:>10.4f} {P[i,2]:>10.4f}   '
              f'{roles.get(i, "")}')
    print()

    print('--- residuals: measured vs reconstructed ---')
    print(f'{"pair":>8} {"measured":>10} {"rebuilt":>10} {"error":>9}')
    print('-' * 40)
    errs = []
    for (i, j), d in sorted(meas.items()):
        rebuilt = float(np.linalg.norm(P[i] - P[j]))
        errs.append(rebuilt - d)
        print(f'{"A"+str(i)+"-A"+str(j):>8} {d:>10.4f} {rebuilt:>10.4f} '
              f'{(rebuilt-d)*1000:>+8.1f} mm')
    errs = np.array(errs)
    rms = math.sqrt((errs ** 2).sum() / dof) if dof > 0 else 0.0
    print('-' * 40)
    print(f'{"rms":>8} {"":>10} {"":>10} {rms*1000:>8.1f} mm   over {dof} dof')
    print(f'{"worst":>8} {"":>10} {"":>10} {np.abs(errs).max()*1000:>8.1f} mm')
    print()
    if dof > 0:
        if rms < 0.005:
            print('  Consistent at tape accuracy. These coordinates are sound.')
        elif rms < 0.02:
            print('  Acceptable, but one measurement may be a few mm out. The')
            print('  worst pair above is where to re-measure.')
        else:
            print('  TOO LARGE. These distances cannot describe a rigid 3D set:')
            print('  re-measure the worst pair, and check the heights -- a wrong')
            print('  height turns a good slant distance into a bad horizontal one.')
    print()

    # Planarity: a flat constellation is still mirror-ambiguous for a tag.
    sv = np.linalg.svd(P - P.mean(axis=0), compute_uv=False)
    print(f'--- shape ---')
    print(f'  singular values {sv[0]:.3f} / {sv[1]:.3f} / {sv[2]:.3f} m')
    if sv[2] < 0.1 * sv[0]:
        print('  NEARLY PLANAR. A tag has a mirror twin reflected through that')
        print('  plane, and vertical accuracy stays poor however good these')
        print('  coordinates are. Raising one anchor is the fix; correcting the')
        print('  coordinates is not.')
    else:
        print('  Genuinely three-dimensional -- no mirror ambiguity.')
    print()

    print('--- what to do with these ---')
    print('  Write them into the ANCHORS (CFClient LPS tab). They are not a')
    print('  Crazyflie parameter: each anchor broadcasts its own position and')
    print('  the tag caches what it hears.')
    print()
    print('  Then  swarmCurve.cx = 0.0,  cy = 0.0,  cz = <orbit height above A0>')
    print()
    print('  NOTE: this fixes WHERE the anchors are, not how well they are')
    print('  spread. Correct coordinates remove a bias; they do not change the')
    print('  dilution of a compact constellation.')

    if args.radius:
        base = min(float(np.linalg.norm(P[i] - P[j]))
                   for i, j in itertools.combinations(range(n), 2))
        span = float(max(np.linalg.norm(P[i] - P[j])
                         for i, j in itertools.combinations(range(n), 2)))
        d = math.hypot(args.radius, args.radius)
        print()
        print('--- geometry check ---')
        print(f'  constellation span   {span:.3f} m')
        print(f'  closest pair         {base:.3f} m')
        print(f'  planned radius       {args.radius:.3f} m')
        print(f'  span / radius        {span/args.radius:.2f}')
        print(f'  single-shot error  ~ {math.sqrt(2)*(d/span)*args.sigma*100:.1f} cm'
              f'  (sigma {args.sigma*100:.0f} mm)')


if __name__ == '__main__':
    main()
