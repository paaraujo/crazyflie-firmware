#!/usr/bin/env python3
"""
Validate the UWB position estimate, optionally against Vicon ground truth.

Usage
-----
    # Noise only -- no Vicon needed. Drones stationary.
    python3 tools/swarm/validate_position.py record --robots cf252 cf253 cf254 \
            --duration 60 --out /tmp/pose_a.json

    # Same, but also capture Vicon for a later accuracy fit.
    python3 tools/swarm/validate_position.py record --robots cf252 cf253 cf254 \
            --duration 60 --vicon --out /tmp/pose_a.json

    # After >= 2 placements, solve the frame and get the accuracy.
    python3 tools/swarm/validate_position.py align /tmp/pose_*.json

Precision versus accuracy
-------------------------
These are different questions and they need different amounts of work.

PRECISION (noise) is the spread of the UWB estimate about its own mean while the
drone sits still. It needs no ground truth at all: the drone is not moving, so
every deviation from the mean is error. `record` reports it immediately.

ACCURACY (bias) is how far that mean sits from the truth, and it needs Vicon
PLUS the rigid transform between the Vicon frame and the template frame. That
transform is not free: three non-collinear points determine it exactly, so
fitting it to three stationary drones gives a zero residual by construction and
tells you nothing. Use at least four well-spread points -- six to ten is better
-- gathered by placing the drones, recording, moving them, and recording again.
`align` does the fit and reports what is left over, which is the accuracy.

Why the Vicon subjects must NOT share the robot names
-----------------------------------------------------
crazyflie_server subscribes to /poses unconditionally and forwards every pose
whose name matches a configured robot straight to that drone as an external
position measurement, which the on-board EKF then fuses. The per-type setting
`motion_capture.enabled: false` does NOT disable this -- it only affects rate
warnings. Name the Vicon subjects cf252_gt, cf253_gt, ... so the lookup misses
and nothing is transmitted. Otherwise you are validating the UWB estimate
against a Vicon stream that is already inside it, and it will look excellent.
"""

import argparse
import json
import math
import sys
from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------

def cmd_record(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import PoseStamped

    vicon_msg = None
    if args.vicon:
        try:
            from motion_capture_tracking_interfaces.msg import NamedPoseArray
            vicon_msg = NamedPoseArray
        except ImportError:
            sys.exit('ERROR: --vicon needs motion_capture_tracking_interfaces.\n'
                     '       Source your ROS workspace, or drop --vicon to do '
                     'noise only.')

    rclpy.init()
    node = Node('validate_position')

    uwb = defaultdict(list)     # robot -> [(t, x, y, z)]
    gt = defaultdict(list)      # subject -> [(t, x, y, z)]

    def make_cb(name):
        def cb(msg):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            p = msg.pose.position
            uwb[name].append((t, p.x, p.y, p.z))
        return cb

    sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=50)

    subs = []
    for r in args.robots:
        subs.append(node.create_subscription(
            PoseStamped, f'/{r}/pose', make_cb(r), 10))

    if vicon_msg is not None:
        def vicon_cb(msg):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            for np_ in msg.poses:
                p = np_.pose.position
                gt[np_.name].append((t, p.x, p.y, p.z))
        subs.append(node.create_subscription(
            vicon_msg, '/poses', vicon_cb, sensor_qos))

    print(f'Recording {args.duration:.0f} s ... keep the drones STILL.')
    end = node.get_clock().now().nanoseconds * 1e-9 + args.duration
    while rclpy.ok():
        now = node.get_clock().now().nanoseconds * 1e-9
        if now >= end:
            break
        rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, end - now)))
    print()

    if not uwb:
        node.destroy_node()
        rclpy.shutdown()
        sys.exit('ERROR: no /pose messages received. Is crazyflie_server running, '
                 'and is firmware_logging default_topics.pose enabled?')

    record = {'uwb': {}, 'vicon': {}, 'duration': args.duration}

    print('=== UWB position estimate, template frame ===')
    for r in args.robots:
        s = summarise(uwb.get(r, []), r)
        if s:
            record['uwb'][r] = s

    if vicon_msg is not None:
        print()
        print('=== Vicon ground truth ===')
        if not gt:
            print('  no /poses messages received -- is motion_capture_tracking up?')
        for name in sorted(gt):
            s = summarise(gt[name], name)
            if s:
                record['vicon'][name] = s
        clash = set(gt) & set(args.robots)
        if clash:
            print()
            print(f'  *** WARNING: Vicon subject(s) {sorted(clash)} share a robot')
            print('  *** name. crazyflie_server is forwarding these to the drones')
            print('  *** as external position and the EKF is fusing them. This')
            print('  *** measurement is NOT UWB-only. Rename them (e.g. _gt).')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(record, f, indent=2)
        print()
        print(f'wrote {args.out}')
        if vicon_msg is not None:
            print('Move the drones to a new placement and record again; once you')
            print('have >= 2 files (>= 4 points total), run the align subcommand.')

    node.destroy_node()
    rclpy.shutdown()


def summarise(samples, label):
    """Per-axis statistics for one stationary point. Returns a dict or None."""
    if len(samples) < 10:
        print(f'  {label:>10}  only {len(samples)} samples -- skipped')
        return None

    a = np.array(samples, dtype=float)
    t, p = a[:, 0], a[:, 1:4]

    span = t[-1] - t[0]
    rate = (len(t) - 1) / span if span > 0 else float('nan')

    mean = p.mean(axis=0)
    std = p.std(axis=0, ddof=1)
    std3 = float(np.sqrt((std ** 2).sum()))
    ptp = p.max(axis=0) - p.min(axis=0)

    # Split-half drift: white noise averages down, a drifting bias does not.
    half = len(p) // 2
    drift = p[half:].mean(axis=0) - p[:half].mean(axis=0)
    # Expected spread of that difference under pure white noise.
    expect = std * math.sqrt(2.0 / half)
    drifting = np.any(np.abs(drift) > 3.0 * expect)

    print(f'  {label}')
    print(f'      samples {len(t)}  over {span:.1f} s   ({rate:.1f} Hz)')
    print(f'      mean    x {mean[0]:+.4f}  y {mean[1]:+.4f}  z {mean[2]:+.4f}  m')
    print(f'      std     x {std[0]*1000:6.1f}  y {std[1]*1000:6.1f}  '
          f'z {std[2]*1000:6.1f}  mm     3D {std3*1000:.1f} mm')
    print(f'      p-p     x {ptp[0]*1000:6.1f}  y {ptp[1]*1000:6.1f}  '
          f'z {ptp[2]*1000:6.1f}  mm')
    print(f'      drift   x {drift[0]*1000:+6.1f}  y {drift[1]*1000:+6.1f}  '
          f'z {drift[2]*1000:+6.1f}  mm  (1st half -> 2nd half)'
          f'{"   <-- NOT white noise" if drifting else ""}')

    return {'n': len(t), 'rate': rate, 'mean': mean.tolist(),
            'std': std.tolist(), 'std3d': std3, 'ptp': ptp.tolist(),
            'drift': drift.tolist()}


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------

def kabsch(P, Q):
    """Rigid transform (no scale) taking P onto Q, minimising ||R p + t - q||.

    Returns (R, t). Reflections are excluded: a mirrored fit would reduce the
    residual while describing a frame no rigid body can occupy.
    """
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def cmd_align(args):
    pairs, labels = [], []

    for path in args.files:
        with open(path) as f:
            rec = json.load(f)
        if not rec.get('vicon'):
            print(f'  {path}: no Vicon data -- skipped')
            continue
        for robot, u in rec['uwb'].items():
            # Match cf252 <-> cf252_gt, cf252_vicon, Cf252 ...
            match = None
            for subj in rec['vicon']:
                base = subj.lower()
                for suf in ('_gt', '_vicon', '_mocap', '_truth'):
                    if base.endswith(suf):
                        base = base[:-len(suf)]
                        break
                if base == robot.lower():
                    match = subj
                    break
            if match is None:
                print(f'  {path}: no Vicon subject matches {robot} -- skipped')
                continue
            pairs.append((rec['vicon'][match]['mean'], u['mean']))
            labels.append(f'{path.split("/")[-1]}:{robot}')

    n = len(pairs)
    print()
    print(f'=== {n} point pair(s) from {len(args.files)} file(s) ===')
    if n < 4:
        sys.exit(f'\nERROR: need at least 4 point pairs to measure accuracy; have {n}.\n'
                 '       Three points determine a rigid transform exactly, so the\n'
                 '       residual would be zero regardless of the real error.\n'
                 '       Record more placements and pass all the files.')

    P = np.array([p[0] for p in pairs], dtype=float)   # vicon
    Q = np.array([p[1] for p in pairs], dtype=float)   # uwb

    # Degenerate spread check: points on a line or plane leave the transform
    # underdetermined in the missing direction, and the residual understates it.
    sv = np.linalg.svd(P - P.mean(axis=0), compute_uv=False)
    if sv[2] < 0.05 * sv[0]:
        print(f'  WARNING: the points are nearly coplanar (singular values '
              f'{sv[0]:.2f} / {sv[1]:.2f} / {sv[2]:.3f}).')
        print('  The fit is poorly constrained out of that plane. Spread the')
        print('  placements in all three axes, especially height.')
        print()

    R, t = kabsch(P, Q)
    resid = (R @ P.T).T + t - Q
    norms = np.linalg.norm(resid, axis=1)

    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))

    print()
    print('--- Vicon -> template frame ---')
    print(f'  translation   {t[0]:+.4f}  {t[1]:+.4f}  {t[2]:+.4f}  m')
    print(f'  yaw           {yaw:+.2f} deg')
    print('  rotation')
    for row in R:
        print(f'      {row[0]:+.5f}  {row[1]:+.5f}  {row[2]:+.5f}')
    print()

    print('--- residual after alignment = UWB accuracy ---')
    print(f'{"point":>28} {"dx":>8} {"dy":>8} {"dz":>8} {"|d|":>8}   (mm)')
    print('-' * 70)
    for lab, r, nm in zip(labels, resid, norms):
        print(f'{lab:>28} {r[0]*1000:>8.1f} {r[1]*1000:>8.1f} '
              f'{r[2]*1000:>8.1f} {nm*1000:>8.1f}')
    print('-' * 70)
    rms = float(np.sqrt((norms ** 2).mean()))
    print(f'{"RMS":>28} {resid[:,0].std()*1000:>8.1f} '
          f'{resid[:,1].std()*1000:>8.1f} {resid[:,2].std()*1000:>8.1f} '
          f'{rms*1000:>8.1f}')
    print(f'{"worst":>28} {"":>8} {"":>8} {"":>8} {norms.max()*1000:>8.1f}')
    print()

    dof = 3 * n - 6          # 6 parameters consumed by the rigid transform
    print(f'  {n} points, {dof} residual degrees of freedom')
    print(f'  RMS 3D accuracy: {rms*1000:.1f} mm')
    print()
    print('  This is the accuracy AFTER removing the best-fit frame, so it')
    print('  excludes any constant offset or rotation between Vicon and the')
    print('  template -- those are survey error, not UWB error. What is left')
    print('  is anchor-position error plus range bias plus geometry.')


def main():
    ap = argparse.ArgumentParser(
        description='Validate the UWB position estimate against Vicon.')
    sub = ap.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('record', help='record stationary poses and report noise')
    r.add_argument('--robots', nargs='+', required=True,
                   help='robot names as in crazyflies.yaml, e.g. cf252 cf253')
    r.add_argument('--duration', type=float, default=60.0, help='seconds')
    r.add_argument('--vicon', action='store_true',
                   help='also record /poses for a later accuracy fit')
    r.add_argument('--out', help='write a JSON file for the align step')
    r.set_defaults(func=cmd_record)

    a = sub.add_parser('align', help='fit the frame and report accuracy')
    a.add_argument('files', nargs='+', help='JSON files from record --vicon')
    a.set_defaults(func=cmd_align)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
