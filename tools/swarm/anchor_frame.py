#!/usr/bin/env python3
"""
Turn three tape measurements of the L-template into anchor positions.

Usage:
    python3 tools/swarm/anchor_frame.py --lx 1.50 --ly 1.00 --l12 1.803
    python3 tools/swarm/anchor_frame.py --lx 1.50 --ly 1.00 --l12 1.803 \
            --z0 0.02 --z1 0.02 --z2 0.05 --radius 1.0

Why measure three sides
-----------------------
The firmware never measures the template: the filter consumes anchor POSITIONS,
and those come from whatever you configure into the anchors. Your tape is the
only source, and at 2-3 mm it is about an order of magnitude better than
anything the UWB could measure about itself.

Three lengths determine the triangle by side-side-side, so the interior angle is
COMPUTED rather than assumed:

    cos(gamma) = (Lx^2 + Ly^2 - L12^2) / (2 Lx Ly)

Nothing in the maths needs the template to be square -- it needs the geometry to
be accurately KNOWN. A rigid 87 degree template with gamma measured is fine. A
nominally-90 template that is really 87, configured as 90, is a systematic error
that never announces itself: it shows up as a slowly rotating bias in the phase.

Heights
-------
Measured lengths are 3D (slant) distances. If the anchors sit at different
heights, pass them with --z0/--z1/--z2 and the horizontal distances are
recovered before the triangle is laid out. Leaving an unmodelled height
difference in tilts the whole frame, and every phase inherits the tilt.

Measure to the ANTENNA, not the enclosure: the UWB phase centre can be several
millimetres from the obvious landmark. Being consistent across all three anchors
matters more than being exactly right, since a common offset largely cancels.
"""

import argparse
import math
import sys


def horizontal(slant, dz, what):
    """Recover the horizontal leg of a slant distance."""
    if slant < abs(dz) - 1e-9:
        sys.exit(f'ERROR: {what} = {slant:.4f} m is shorter than its height '
                 f'difference {abs(dz):.4f} m -- impossible. Check the inputs.')
    return math.sqrt(max(0.0, slant * slant - dz * dz))


def main():
    ap = argparse.ArgumentParser(
        description='Compute L-template anchor positions from tape measurements.')
    ap.add_argument('--lx', type=float, required=True,
                    help='distance A0 -> A1, the x arm [m]')
    ap.add_argument('--ly', type=float, required=True,
                    help='distance A0 -> A2, the y arm [m]')
    ap.add_argument('--l12', type=float, required=True,
                    help='distance A1 -> A2, the hypotenuse [m]')
    ap.add_argument('--z0', type=float, default=0.0, help='height of A0 [m]')
    ap.add_argument('--z1', type=float, default=0.0, help='height of A1 [m]')
    ap.add_argument('--z2', type=float, default=0.0, help='height of A2 [m]')
    ap.add_argument('--radius', type=float, default=None,
                    help='planned encirclement radius [m], for a geometry check')
    ap.add_argument('--sigma', type=float, default=0.02,
                    help='UWB range noise, one sigma [m], for the geometry check')
    args = ap.parse_args()

    for name, v in (('lx', args.lx), ('ly', args.ly), ('l12', args.l12)):
        if v <= 0:
            sys.exit(f'ERROR: --{name} must be positive')

    print('--- measurements ---')
    print(f'  A0 -> A1  {args.lx:.4f} m      (x arm)')
    print(f'  A0 -> A2  {args.ly:.4f} m      (y arm)')
    print(f'  A1 -> A2  {args.l12:.4f} m      (hypotenuse)')
    print(f'  heights   A0 {args.z0:+.3f}  A1 {args.z1:+.3f}  A2 {args.z2:+.3f} m')
    print()

    # Slant -> horizontal, so the triangle can be laid out in the plane.
    hx = horizontal(args.lx, args.z1 - args.z0, 'A0->A1')
    hy = horizontal(args.ly, args.z2 - args.z0, 'A0->A2')
    h12 = horizontal(args.l12, args.z2 - args.z1, 'A1->A2')

    if any(abs(a - b) > 1e-9 for a, b in
           ((hx, args.lx), (hy, args.ly), (h12, args.l12))):
        print('--- horizontal projections ---')
        print(f'  A0 -> A1  {hx:.4f} m')
        print(f'  A0 -> A2  {hy:.4f} m')
        print(f'  A1 -> A2  {h12:.4f} m')
        print()

    # Triangle inequality on the horizontal legs.
    for a, b, c, label in ((hx, hy, h12, 'A0->A1 + A0->A2 vs A1->A2'),
                           (hx, h12, hy, 'A0->A1 + A1->A2 vs A0->A2'),
                           (hy, h12, hx, 'A0->A2 + A1->A2 vs A0->A1')):
        if a + b <= c + 1e-9:
            sys.exit(f'ERROR: triangle inequality violated ({label}). '
                     'At least one measurement is wrong.')

    cos_g = (hx * hx + hy * hy - h12 * h12) / (2.0 * hx * hy)
    if not -1.0 <= cos_g <= 1.0:
        sys.exit(f'ERROR: cos(gamma) = {cos_g:.4f} is out of range. '
                 'The three lengths cannot form a triangle.')
    gamma = math.acos(cos_g)

    print('--- computed geometry ---')
    print(f'  interior angle gamma = {math.degrees(gamma):.2f} deg '
          f'({math.degrees(gamma) - 90.0:+.2f} deg from square)')
    if abs(math.degrees(gamma) - 90.0) > 10.0:
        print('  NOTE: far from square. That is fine for the maths -- only the')
        print('        accuracy of the KNOWN value matters -- but check the tape.')
    print()

    a0 = (0.0, 0.0, args.z0)
    a1 = (hx, 0.0, args.z1)
    a2 = (hy * math.cos(gamma), hy * math.sin(gamma), args.z2)

    print('--- anchor positions, template frame ---')
    print(f'{"anchor":>8} {"x [m]":>10} {"y [m]":>10} {"z [m]":>10}   role')
    print('-' * 56)
    for name, p, role in ((0, a0, 'origin / corner'),
                          (1, a1, 'defines +x  (theta = 0)'),
                          (2, a2, 'defines +y  (CCW sense)')):
        print(f'{name:>8} {p[0]:>10.4f} {p[1]:>10.4f} {p[2]:>10.4f}   {role}')
    print()

    # Round trip: rebuild the distances from the positions. This catches an
    # arithmetic slip here, not a measurement error -- those show up as a
    # triangle-inequality or gamma failure above.
    def dist(p, q):
        return math.sqrt(sum((p[i] - q[i]) ** 2 for i in range(3)))

    worst = max(abs(dist(a0, a1) - args.lx),
                abs(dist(a0, a2) - args.ly),
                abs(dist(a1, a2) - args.l12))
    print(f'  round-trip check: distances reproduced to {worst * 1000:.3f} mm')
    if worst > 1e-6:
        print('  WARNING: round trip does not close; do not use these positions.')
    print()

    print('--- what to do with these ---')
    print('  Configure them into the ANCHORS (cfclient LPS tab, or the anchor')
    print('  config tool). They are not a Crazyflie parameter: each anchor')
    print('  broadcasts its own position, and the filter reads it from there.')
    print()
    print('  A0 is the frame origin, so the encirclement centre sits above it:')
    print('      swarmCurve.cx = 0.0')
    print('      swarmCurve.cy = 0.0')
    print('      swarmCurve.cz = <orbit altitude above the anchor plane>')
    print()

    if args.radius:
        # Instantaneous trilateration error scales as sqrt(2) * d / L. This is
        # a pessimistic bound: the filter fuses many ranges over time and does
        # far better. It is the SCALING that matters when sizing a template.
        d = math.hypot(args.radius, args.radius)   # rough slant to an anchor
        base = min(hx, hy)
        err = math.sqrt(2.0) * (d / base) * args.sigma
        print('--- geometry check ---')
        print(f'  shortest baseline      {base:.3f} m')
        print(f'  planned radius         {args.radius:.3f} m')
        print(f'  baseline / radius      {base / args.radius:.2f}')
        print(f'  single-shot position error ~ {err * 100:.1f} cm '
              f'(sigma = {args.sigma * 100:.0f} mm)')
        print()
        if base < 0.5 * args.radius:
            print('  The baseline is short relative to the orbit. Position error')
            print('  scales as distance / baseline, so a small template multiplies')
            print('  the ranging noise. Aim for a baseline comparable to r.')
        else:
            print('  Baseline is a reasonable fraction of the orbit radius.')
        print()
        print('  This is the INSTANTANEOUS trilateration bound. The filter fuses')
        print('  ranges over time and constrains the agent to the curve, so the')
        print('  figure it achieves is several times better -- treat this as a')
        print('  pessimistic sizing guide, not a prediction.')


if __name__ == '__main__':
    main()
