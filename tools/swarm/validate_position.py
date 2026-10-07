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
import re
import sys
from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------

def collect(robots, duration, want_vicon, msg_prefix='Recording', ranges_topic=None):
    """Subscribe and gather samples. Returns (uwb, gt, had_vicon, rng)."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import PoseStamped

    vicon_msg = None
    if want_vicon:
        try:
            from motion_capture_tracking_interfaces.msg import NamedPoseArray
            vicon_msg = NamedPoseArray
        except ImportError:
            sys.exit('ERROR: --vicon needs motion_capture_tracking_interfaces.\n'
                     '       Source your ROS workspace, or drop --vicon.')

    ranges_msg = None
    if ranges_topic:
        try:
            from crazyflie_interfaces.msg import LogDataGeneric
            ranges_msg = LogDataGeneric
        except ImportError:
            sys.exit('ERROR: needs crazyflie_interfaces. Source your ROS workspace.')

    rclpy.init()
    node = Node('validate_position')
    uwb, gt, rng = defaultdict(list), defaultdict(list), defaultdict(list)

    def make_cb(name):
        def cb(msg):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            q = msg.pose.position
            uwb[name].append((t, q.x, q.y, q.z))
        return cb

    for r in robots:
        node.create_subscription(PoseStamped, f'/{r}/pose', make_cb(r), 10)

    if ranges_msg is not None:
        def make_rng_cb(name):
            def cb(msg):
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                rng[name].append((t,) + tuple(msg.values))
            return cb
        for r in robots:
            node.create_subscription(ranges_msg, f'/{r}/{ranges_topic}',
                                     make_rng_cb(r), 10)

    if vicon_msg is not None:
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=50)
        def vicon_cb(msg):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            for np_ in msg.poses:
                q = np_.pose.position
                gt[np_.name].append((t, q.x, q.y, q.z))
        node.create_subscription(vicon_msg, '/poses', vicon_cb, sensor_qos)

    print(f'{msg_prefix} {duration:.0f} s ...')
    end = node.get_clock().now().nanoseconds * 1e-9 + duration
    while rclpy.ok():
        now = node.get_clock().now().nanoseconds * 1e-9
        if now >= end:
            break
        rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, end - now)))
    print()

    node.destroy_node()
    rclpy.shutdown()
    return uwb, gt, vicon_msg is not None, rng


# --------------------------------------------------------------------------
# hover analysis
# --------------------------------------------------------------------------

BANDS = (('drift  0.05-0.5 Hz', 0.05, 0.5),
         ('loop   0.5 -2.0 Hz', 0.5, 2.0),
         ('noise  2.0+   Hz  ', 2.0, 1e9))


def welch(x, fs, nseg=256):
    """Averaged periodogram. Returns (freqs, psd). No scipy needed."""
    n = len(x)
    nseg = min(nseg, n)
    step = max(nseg // 2, 1)
    win = np.hanning(nseg)
    norm = fs * (win ** 2).sum()
    acc, cnt = None, 0
    for start in range(0, n - nseg + 1, step):
        seg = x[start:start + nseg]
        seg = (seg - seg.mean()) * win
        P = np.abs(np.fft.rfft(seg)) ** 2 / norm
        P[1:-1] *= 2.0
        acc = P if acc is None else acc + P
        cnt += 1
    if cnt == 0:
        return np.array([0.0]), np.array([0.0])
    return np.fft.rfftfreq(nseg, 1.0 / fs), acc / cnt


def analyse_hover(samples, label, plot_path=None):
    if len(samples) < 64:
        print(f'  {label:>10}  only {len(samples)} samples -- need >= 64')
        return None

    a = np.array(samples, dtype=float)
    t, p = a[:, 0], a[:, 1:4]
    t = t - t[0]

    # Uniform grid: log timestamps jitter, and the FFT assumes even spacing.
    fs = (len(t) - 1) / (t[-1] - t[0])
    tg = np.arange(t[0], t[-1], 1.0 / fs)
    pg = np.column_stack([np.interp(tg, t, p[:, i]) for i in range(3)])

    mean = pg.mean(axis=0)
    std = pg.std(axis=0, ddof=1)
    std3 = float(np.sqrt((std ** 2).sum()))

    print(f'  {label}')
    print(f'      samples {len(t)}  over {t[-1]:.1f} s   ({fs:.1f} Hz, '
          f'Nyquist {fs/2:.1f} Hz)')
    print(f'      mean    x {mean[0]:+.4f}  y {mean[1]:+.4f}  z {mean[2]:+.4f}  m')
    print(f'      std     x {std[0]*1000:6.1f}  y {std[1]*1000:6.1f}  '
          f'z {std[2]*1000:6.1f}  mm     3D {std3*1000:.1f} mm')

    out = {'mean': mean.tolist(), 'std': std.tolist(), 'std3d': std3,
           'fs': fs, 'bands': {}, 'peak': {}}
    psds = {}

    print()
    print(f'      {"":5} {"  ".join(n for n,_,_ in BANDS)}   peak')
    for i, ax in enumerate('xyz'):
        f, P = welch(pg[:, i], fs)
        psds[ax] = (f, P)
        tot = P[1:].sum()
        fracs = []
        for _, lo, hi in BANDS:
            m = (f >= lo) & (f < hi)
            fracs.append(float(P[m].sum() / tot) if tot > 0 else 0.0)
        band = (f >= 0.05)
        pk = float(f[band][np.argmax(P[band])]) if band.any() else 0.0
        print(f'      {ax:>5} ' + '  '.join(f'{v*100:16.0f}%' for v in fracs)
              + f'   {pk:.2f} Hz')
        out['bands'][ax] = fracs
        out['peak'][ax] = pk

    # Verdict from where the horizontal energy sits. Vertical is driven by a
    # different loop (thrust, not attitude) so it is reported but not voted.
    loop = max(out['bands']['x'][1], out['bands']['y'][1])
    noise = max(out['bands']['x'][2], out['bands']['y'][2])
    drift = max(out['bands']['x'][0], out['bands']['y'][0])
    print()
    if loop > 0.45:
        print('      VERDICT: energy concentrated at 0.5-2 Hz -> LOOP OSCILLATION.')
        print('      Estimator lag or gains too high. LOWER tdoa3.twrStd (less lag)')
        print('      and/or lower posCtlPid position gains. Smoothing harder will')
        print('      make this worse, not better.')
    elif noise > 0.45:
        print('      VERDICT: energy mostly above 2 Hz -> ESTIMATOR NOISE.')
        print('      The controller is chasing range noise. RAISE tdoa3.twrStd,')
        print('      lower position gains, or widen the anchor baseline.')
    elif drift > 0.45:
        print('      VERDICT: energy mostly below 0.5 Hz -> SLOW WANDER.')
        print('      Not a tuning problem: suspect anchor geometry or survey, or')
        print('      integral wind-up. Compare with the stationary record.')
    else:
        print('      VERDICT: energy spread across bands, no single mechanism.')
        print('      Fix the largest contributor first and re-measure.')

    if plot_path:
        make_plot(tg, pg, psds, label, plot_path)
        print(f'      plot -> {plot_path}')
    return out


def make_plot(t, p, psds, label, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    for i, a in enumerate('xyz'):
        ax[0].plot(t, (p[:, i] - p[:, i].mean()) * 1000, lw=0.7, label=a)
    ax[0].set_xlabel('time [s]'); ax[0].set_ylabel('deviation from mean [mm]')
    ax[0].set_title(f'{label}  hover, position about the mean')
    ax[0].legend(); ax[0].grid(alpha=.3)

    for a, (f, P) in psds.items():
        ax[1].loglog(f[1:], np.sqrt(P[1:]) * 1000, lw=0.9, label=a)
    for _, lo, hi in BANDS[:2]:
        ax[1].axvline(hi, color='k', ls=':', lw=.8)
    ax[1].set_xlabel('frequency [Hz]')
    ax[1].set_ylabel(r'amplitude spectral density [mm/$\sqrt{Hz}$]')
    ax[1].set_title('where the motion lives:  <0.5 Hz drift | 0.5-2 Hz loop | >2 Hz noise')
    ax[1].legend(); ax[1].grid(alpha=.3, which='both')

    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def cmd_hover(args):
    uwb, gt, had_vicon, _ = collect(args.robots, args.duration, args.vicon,
                                    'Hovering -- recording')
    if not uwb:
        sys.exit('ERROR: no /pose messages received.')

    print('=== UWB position in hover ===')
    for r in args.robots:
        plot = f'{args.plot}_{r}.png' if args.plot else None
        analyse_hover(uwb.get(r, []), r, plot)
        print()

    if had_vicon and gt:
        print('=== Vicon: real motion vs estimator error ===')
        print('  Vicon shows how much the VEHICLE actually moved; the UWB std')
        print('  above is that plus estimator error. Compare the two.')
        for name in sorted(gt):
            plot = f'{args.plot}_{name}.png' if args.plot else None
            analyse_hover(gt[name], name, plot)
            print()


def cmd_record(args):
    uwb, gt, had_vicon, _ = collect(args.robots, args.duration, args.vicon,
                                    'Recording (keep the drones STILL)')

    if not uwb:
        sys.exit('ERROR: no /pose messages received. Is crazyflie_server running, '
                 'and is firmware_logging default_topics.pose enabled?')

    record = {'uwb': {}, 'vicon': {}, 'duration': args.duration}

    print('=== UWB position estimate, template frame ===')
    for r in args.robots:
        s = summarise(uwb.get(r, []), r)
        if s:
            record['uwb'][r] = s

    if had_vicon:
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
        if had_vicon:
            print('Move the drones to a new placement and record again; once you')
            print('have >= 2 files (>= 4 points total), run the align subcommand.')


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

    # The EKF output is strongly autocorrelated -- consecutive samples are not
    # independent draws. Treating them as independent makes the expected spread
    # far too small and the test fires on almost any record. Estimate an
    # effective sample size from the lag-1 autocorrelation (AR(1) model):
    #     n_eff = n (1 - rho) / (1 + rho)
    dev = p - p.mean(axis=0)
    neff = np.empty(3)
    for i in range(3):
        v = dev[:, i]
        den = float((v * v).sum())
        rho = float((v[:-1] * v[1:]).sum() / den) if den > 0 else 0.0
        rho = min(max(rho, 0.0), 0.99)
        neff[i] = max(len(v) * (1.0 - rho) / (1.0 + rho), 2.0)

    # sd of a split-half difference of means, with n_eff/2 per half.
    expect = 2.0 * std / np.sqrt(neff)
    zscore = np.abs(drift) / np.maximum(expect, 1e-12)
    # Statistical significance is meaningless below physical relevance. Vicon
    # sits at its quantisation floor (std ~0.0 mm), so a 0.0 mm drift divided by
    # a ~0 expectation reads as tens of sigma. Require a real effect size too.
    DRIFT_FLOOR_M = 1.0e-3
    drifting = bool(np.any((zscore > 3.0) & (np.abs(drift) > DRIFT_FLOOR_M)))

    print(f'  {label}')
    print(f'      samples {len(t)}  over {span:.1f} s   ({rate:.1f} Hz)')
    print(f'      mean    x {mean[0]:+.4f}  y {mean[1]:+.4f}  z {mean[2]:+.4f}  m')
    print(f'      std     x {std[0]*1000:6.1f}  y {std[1]*1000:6.1f}  '
          f'z {std[2]*1000:6.1f}  mm     3D {std3*1000:.1f} mm')
    print(f'      p-p     x {ptp[0]*1000:6.1f}  y {ptp[1]*1000:6.1f}  '
          f'z {ptp[2]*1000:6.1f}  mm')
    print(f'      drift   x {drift[0]*1000:+6.1f}  y {drift[1]*1000:+6.1f}  '
          f'z {drift[2]*1000:+6.1f}  mm  (1st half -> 2nd half)')
    print(f'      drift-z {zscore[0]:6.1f}  {zscore[1]:6.1f}  {zscore[2]:6.1f}'
          f'     sigma, n_eff {neff[0]:.0f}/{neff[1]:.0f}/{neff[2]:.0f}'
          f'{"   <-- SYSTEMATIC" if drifting else ""}')

    # Coplanar anchors admit a mirror solution: a point and its reflection in
    # the anchor plane give identical ranges to every anchor. If the anchors are
    # on the floor and the estimate is below it, the filter picked the wrong
    # branch -- and the altitude loop will then have inverted feedback.
    if mean[2] < -0.05:
        print(f'      *** z is NEGATIVE ({mean[2]:+.3f} m). If the anchors are on')
        print('      *** the floor this is the MIRROR solution: identical ranges,')
        print('      *** wrong branch. Do not fly -- altitude feedback inverts.')
        print('      *** Break the symmetry: raise an anchor off the plane.')

    return {'n': len(t), 'rate': rate, 'mean': mean.tolist(),
            'std': std.tolist(), 'std3d': std3, 'ptp': ptp.tolist(),
            'drift': drift.tolist(), 'neff': neff.tolist(),
            'drift_sigma': zscore.tolist()}


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


# --------------------------------------------------------------------------
# per-anchor range residuals
# --------------------------------------------------------------------------

# Single-sided TWR rescales the reply interval by an estimated clock
# correction, so a clock error eps leaks straight into range:
#
#     range error = c * eps * t_reply / 2
#
# With t_reply in ms and the error in metres that inverts to
#
#     eps [ppm] = 2e9 / c * bias / t_reply = 6.6713 * bias_m / t_ms
#
# Classic TWR (lpsTwrTag.c) uses the double-sided estimator, where the reply
# intervals cancel algebraically and no clock correction appears at all. That
# is why the same hardware and geometry range accurately in TWR mode.
EPS_PPM_PER_M_PER_MS = 2.0e9 / 299792458.0

# Slot convention, matching swarm_task.c: 0-2 are anchors, 3 is the leader and
# 4 the follower. Slot 5 is spare.
N_ANCHOR_SLOTS = 3


def fit_reply_bias(t_ms, bias_m):
    """Fit  bias = a + k*t_reply  over the anchors of one drone.

    The intercept absorbs whatever is COMMON to every link -- the Vicon/template
    frame datum and the antenna-delay calibration, which are not separable from
    each other but are separable from a reply-time effect. The slope is the
    reply-time term, and eps = k scaled into ppm.

    Returns (a, eps_ppm, residual_rms, dof) or None if underdetermined.
    """
    t = np.asarray(t_ms, float)
    b = np.asarray(bias_m, float)
    n = len(t)
    # The slope needs a lever arm. If every link replies at nearly the same
    # time there is no way to separate slope from intercept, and the fit will
    # happily return an enormous eps driven entirely by measurement noise.
    MIN_SPREAD_MS = 1.0
    if n < 3 or not np.all(np.isfinite(t)):
        return None
    if float(np.ptp(t)) < MIN_SPREAD_MS:
        return 'flat', float(np.ptp(t))
    A = np.column_stack([np.ones(n), t])
    (a, k), *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - (a + k * t)
    dof = n - 2
    rms = float(np.sqrt((resid ** 2).sum() / dof)) if dof > 0 else 0.0
    return float(a), float(k * EPS_PPM_PER_M_PER_MS), rms, dof


SUFFIXES = ('_gt', '_vicon', '_mocap', '_truth')


def strip_suffix(name):
    low = name.lower()
    for suf in SUFFIXES:
        if low.endswith(suf):
            return low[:-len(suf)]
    return low


def match_subject(gt, want):
    """Find a Vicon subject matching `want`, ignoring a _gt-style suffix."""
    for subj in gt:
        if strip_suffix(subj) == want.lower():
            return subj
    return None


def parse_slots(spec):
    """Parse a log-block layout like "D0,D1,D2,RT0,RT1,RT2".

    Returns (dcol, rcol): slot index -> column index in msg.values.
    Keeping this explicit means the block can be rearranged for whatever
    question is being asked without the tool guessing from value counts.
    """
    dcol, rcol = {}, {}
    for col, tok in enumerate(t.strip().upper() for t in spec.split(',')):
        m = re.fullmatch(r'(D|RT)(\d)', tok)
        if not m:
            sys.exit(f'ERROR: --slots entry "{tok}" is not D<n> or RT<n>.\n'
                     '       e.g. --slots D0,D1,D2,RT0,RT1,RT2')
        (rcol if m.group(1) == 'RT' else dcol)[int(m.group(2))] = col
    if len(dcol) + len(rcol) > 6:
        print('  NOTE: more than 6 values -- a CRTP log block holds at most 6 '
              'floats.')
    return dcol, rcol


def live_mean(a, col):
    """Mean and std of one column, ignoring the 0.0 no-measurement sentinel."""
    v = a[:, 1 + col]
    v = v[v != 0.0]
    if len(v) < 10:
        return None, None, len(v)
    return float(v.mean()), float(v.std(ddof=1)), len(v)


def solve_pos(anchors, ranges, p0, iters=60):
    """Gauss-Newton position solve from ranges. `anchors`/`ranges` keyed by slot.

    Started from p0, which also resolves the mirror branch that three ranges
    would otherwise leave open.
    """
    ids = [i for i in anchors if i in ranges]
    if len(ids) < 3:
        return None
    p = np.array(p0, float)
    for _ in range(iters):
        r, J = [], []
        for i in ids:
            v = p - anchors[i]
            d = np.linalg.norm(v)
            if d < 1e-9:
                d, v = 1e-9, np.array([1e-9, 0.0, 0.0])
            r.append(ranges[i] - d)
            J.append(v / d)
        step, *_ = np.linalg.lstsq(np.array(J), np.array(r), rcond=None)
        p = p + step
        if np.abs(step).max() < 1e-10:
            break
    return p


def leave_one_out(anchors, ranges, p0):
    """For each anchor: solve from the OTHERS, then predict this one.

    This is the check that redundancy buys and ground truth is not needed for.
    Holding one measurement out keeps its own error from being absorbed into
    the position solution, so an inconsistent link stands out instead of being
    smeared across every residual.
    """
    out = {}
    ids = [i for i in anchors if i in ranges]
    for held in ids:
        sub_a = {i: anchors[i] for i in ids if i != held}
        sub_r = {i: ranges[i] for i in ids if i != held}
        if len(sub_a) < 3:
            continue
        p = solve_pos(sub_a, sub_r, p0)
        if p is None:
            continue
        pred = float(np.linalg.norm(p - anchors[held]))
        # How well do the REMAINING anchors agree among themselves? If holding
        # one out leaves a mutually consistent set, that one is the culprit --
        # excluding any innocent anchor leaves the fault behind and the subset
        # fits badly. This is what separates the bad link from its victims.
        inner = [sub_r[i] - np.linalg.norm(p - sub_a[i]) for i in sub_a]
        inner_rms = float(np.sqrt(np.mean(np.square(inner))))
        out[held] = (pred, ranges[held] - pred, inner_rms)
    return out


def cmd_ranges(args):
    """Compare MEASURED ranges against ranges computed from Vicon.

    Anchor slots (0-2) separate two causes of a distorted position solution:
      residuals ~ 0 but position wrong -> anchor COORDINATES are wrong
      residuals biased per anchor      -> that LINK is biased

    Chord slots (3-4) are the ones the swarm filter actually consumes, and they
    carry a check the anchor ranges cannot: the same physical distance is
    measured independently from both ends. Reciprocity needs no ground truth
    and no frame, so a disagreement there is unambiguous evidence of a ranging
    fault rather than a survey or datum error.
    """
    spec = 'D0,D1,D2,RT0,RT1,RT2' if args.with_reply else args.slots
    dcol, rcol = parse_slots(spec)

    uwb, gt, had_vicon, rng = collect(args.robots, args.duration, True,
                                      'Recording ranges (keep the drones STILL)',
                                      args.topic)
    if not rng or not any(rng.values()):
        want = '  '.join(f'/{r}/{args.topic}' for r in args.robots)
        sys.exit(
            f'ERROR: no messages on   {want}\n\n'
            '  The per-slot ranges are not published by default. Add this to\n'
            '  crazyflies.yaml under  all: firmware_logging: custom_topics:\n\n'
            f'      {args.topic}:\n'
            '        frequency: 10\n'
            f'        vars: [{", ".join(chr(34) + "tdoa3.hm" + ("RT" if k == "RT" else "D") + str(i) + chr(34) for k, i in [("D", i) for i in sorted(dcol)] + [("RT", i) for i in sorted(rcol)])}]\n\n'
            '  then restart crazyflie_server. Check it is up with:\n'
            f'      ros2 topic hz {args.robots[0]}/{args.topic}')
    have_truth = bool(gt)
    if not have_truth:
        print('No Vicon. Reporting NOISE and SELF-CONSISTENCY, which need no')
        print('ground truth. Absolute bias cannot be measured without it.')
        print()

    arr = {r: np.array(v, dtype=float) for r, v in rng.items() if v}

    # The log block's width must match what --slots declares, or every column
    # index is off and the failure shows up as nonsense rather than an error.
    ncols = max(len(dcol) + len(rcol), 0)
    for r, a in arr.items():
        got = a.shape[1] - 1                      # column 0 is the timestamp
        if got != ncols:
            want = ', '.join(
                f'"tdoa3.hm{"RT" if k == "RT" else "D"}{i}"'
                for k, i in ([('D', i) for i in sorted(dcol)]
                             + [('RT', i) for i in sorted(rcol)]))
            sys.exit(
                f'ERROR: /{r}/{args.topic} publishes {got} value(s), but --slots '
                f'declares {ncols}.\n\n'
                f'  --slots {spec}  expects the log block to be exactly:\n'
                f'      vars: [{want}]\n\n'
                '  Update crazyflies.yaml to match (or change --slots to match\n'
                '  the topic), then restart crazyflie_server.')
        if np.all(a[:, 1:] == 0.0):
            print(f'  WARNING: every value on /{r}/{args.topic} is 0.0, the')
            print('  "no measurement" sentinel. The slots are configured for ids')
            print('  that are not being ranged -- check tdoa3.hmLId0..hmLId5.')
            print()

    # ---------------- anchors ----------------------------------------------
    try:
        role_anchor = {int(x) for x in args.anchor_slots.split(',') if x.strip()}
    except ValueError:
        sys.exit('ERROR: --anchor-slots must be comma-separated slot numbers, '
                 'e.g. 0,1,2,3,4')
    anchor_slots = sorted(i for i in dcol if i in role_anchor)
    if anchor_slots:
        anchors = {}
        if args.anchors:
            for i, triple in enumerate(args.anchors):
                parts = triple.split(',')
                if len(parts) != 3:
                    sys.exit(
                        f'ERROR: --anchors entry {i} has {len(parts)} values, '
                        f'not 3: "{triple}"\n'
                        '       Each anchor is ONE argument "x,y,z", separated '
                        'by SPACES:\n'
                        '           --anchors 0,0,0 0.62,0,0 0,0.65,0')
                try:
                    anchors[i] = np.array([float(v) for v in parts])
                except ValueError:
                    sys.exit(f'ERROR: --anchors entry {i} is not numeric')
            if have_truth:
                print('  *** CAVEAT: these are CONFIGURED coordinates in the template')
                print('  *** frame, while the drone positions come from Vicon. A range')
                print('  *** is frame-independent only when BOTH ends share a frame, so')
                print('  *** any datum offset lands in every "bias" below. Read the')
                print('  *** SPREAD, not the mean. Chords and the self-consistency')
                print('  *** check further down are immune to this.')
                print()
            src = 'command line'
        else:
            for i, want in enumerate(args.anchor_subjects):
                subj = match_subject(gt, want)
                if subj is None:
                    sys.exit(f'ERROR: no Vicon subject matches "{want}". Put '
                             'markers on the anchors, or pass --anchors.')
                anchors[i] = np.array(gt[subj][-1][1:4])
            src = 'Vicon'

        print(f'=== anchor positions (from {src}) ===')
        for i in sorted(anchors):
            a = anchors[i]
            print(f'  A{i}  {a[0]:+.4f} {a[1]:+.4f} {a[2]:+.4f}  m')
        if src == 'Vicon':
            print('  Baselines measured by Vicon:')
            for i, j in ((0, 1), (0, 2), (1, 2)):
                if i in anchors and j in anchors:
                    print(f'    A{i}-A{j}  '
                          f'{np.linalg.norm(anchors[i]-anchors[j]):.4f} m')
        print()

        print('=== anchor slots: range residual  measured - true ===')
        for r in args.robots:
            a = arr.get(r)
            subj = match_subject(gt, r) if have_truth else None
            if a is None:
                print(f'  {r}: no range data')
                continue
            p_true = np.array(gt[subj][-1][1:4]) if subj else None
            if p_true is not None:
                print(f'  {r}   Vicon {p_true[0]:+.3f} {p_true[1]:+.3f} '
                      f'{p_true[2]:+.3f}')
            else:
                print(f'  {r}')
            hdr = f'{"t_reply":>9} {"eps ppm":>9}' if rcol else ''
            print(f'      {"slot":>5} {"n":>6} {"true":>8} {"meas":>8} '
                  f'{"bias":>9} {"noise":>8}{hdr}')
            print('      ' + '-' * (62 if rcol else 50))
            biases, treplies = [], []
            live = {}
            for slot in anchor_slots:
                if slot not in anchors:
                    continue
                meas, noise, n = live_mean(a, dcol[slot])
                if meas is None:
                    print(f'      {slot:>5} {n:>6}   -- too few measurements --')
                    continue
                live[slot] = (meas, noise)
                if p_true is None:
                    print(f'      {slot:>5} {n:>6} {"--":>8} {meas:>8.3f} '
                          f'{"--":>9} {noise*1000:>7.1f}mm')
                    continue
                true = float(np.linalg.norm(p_true - anchors[slot]))
                bias = meas - true
                biases.append(bias)
                tail = ''
                if slot in rcol:
                    t_ms, _, _ = live_mean(a, rcol[slot])
                    if t_ms:
                        treplies.append(t_ms)
                        tail = (f'{t_ms:>8.2f}ms '
                                f'{EPS_PPM_PER_M_PER_MS*bias/t_ms:>+9.3f}')
                print(f'      {slot:>5} {n:>6} {true:>8.3f} {meas:>8.3f} '
                      f'{bias*1000:>+8.0f}mm {noise*1000:>7.1f}mm{tail}')

            if len(biases) >= 2:
                b = np.array(biases)
                spread = float(b.std(ddof=1))
                print('      ' + '-' * (62 if rcol else 50))
                print(f'      across anchors:  mean {b.mean()*1000:+.0f} mm   '
                      f'SPREAD {spread*1000:.0f} mm')
                if spread < 0.03:
                    print('        tight -> no per-link problem. The mean is a COMMON')
                    print('        offset: antenna delay and/or frame datum.')
                else:
                    print('        WIDE -> per-link error.')
                    if np.all(b > 0.0):
                        print('        All POSITIVE -> consistent with multipath.')
                    elif np.any(b < -0.03):
                        neg = [anchor_slots[i] for i, v in enumerate(b) if v < -0.03]
                        print(f'        Slot(s) {neg} NEGATIVE: measured SHORTER than')
                        print('        true. Multipath only ADDS path length, so this is')
                        print('        not multipath. Suspect the frame datum, the')
                        print('        configured coordinates, or timestamp pairing.')
                if len(treplies) == len(biases) and treplies:
                    fit = fit_reply_bias(treplies, biases)
                    if isinstance(fit, tuple) and fit[0] == 'flat':
                        print(f'        reply times agree to {fit[1]*1000:.0f} us -- no')
                        print('        lever arm, so the per-link spread is NOT a clock')
                        print('        effect (equal reply, equal reply-time bias).')
                    elif fit:
                        a_off, eps, rms, dof = fit
                        print(f'        fit bias = a + k*t:  a {a_off*1000:+.0f} mm   '
                              f'eps {eps:+.3f} ppm   rms {rms*1000:.0f} mm ({dof} dof)')
                        if abs(eps) > 5.0:
                            print('        IMPLAUSIBLE eps -- not reply-time driven.')
                        elif rms < 0.3 * spread:
                            print('        CONFIRMED: single-sided TWR clock error.')
                        else:
                            print('        NOT explained by reply time.')
            # --- self-consistency: needs redundancy, not truth ------------
            if len(live) >= 4:
                acs = {k: anchors[k] for k in live}
                rng_ = {k: live[k][0] for k in live}
                p0 = anchors[min(anchors)] + np.array([0.0, 0.0, 1.0])
                if p_true is not None:
                    p0 = p_true
                elif r in uwb and uwb[r]:
                    p0 = np.array(uwb[r][-1][1:4])
                p_fit = solve_pos(acs, rng_, p0)
                loo = leave_one_out(acs, rng_, p0)
                print()
                print('      --- self-consistency (no ground truth needed) ---')
                if p_fit is not None:
                    print(f'      position from all {len(live)} ranges: '
                          f'{p_fit[0]:+.3f} {p_fit[1]:+.3f} {p_fit[2]:+.3f}')
                print(f'      {"held out":>9} {"measured":>9} {"predicted":>10} '
                      f'{"residual":>10} {"others fit":>11}')
                print('      ' + '-' * 60)
                for slot in sorted(loo):
                    pred, res, inner = loo[slot]
                    print(f'      {"A"+str(slot):>9} {rng_[slot]:>9.3f} '
                          f'{pred:>10.3f} {res*1000:>+9.0f}mm '
                          f'{inner*1000:>9.0f}mm')
                print('      ' + '-' * 60)
                # The culprit is the anchor whose removal leaves the cleanest set.
                culprit = min(loo, key=lambda k: loo[k][2])
                c_pred, c_res, c_inner = loo[culprit]
                others = [loo[k][2] for k in loo if k != culprit]
                margin = (min(others) / c_inner) if c_inner > 1e-9 and others else 1.0
                if abs(c_res) > 0.08 and margin > 2.0:
                    print(f'      A{culprit} IS THE OUTLIER: {c_res*1000:+.0f} mm off, and')
                    print(f'      removing it leaves the other anchors agreeing to')
                    print(f'      {c_inner*1000:.0f} mm -- {margin:.1f}x better than excluding any')
                    print('      other. Large residuals elsewhere are that fault being')
                    print('      absorbed into the solve, not separate problems.')
                elif max(abs(v[1]) for v in loo.values()) > 0.08:
                    print('      Residuals are large but no single anchor stands out.')
                    print('      Either more than one link is bad, or the configured')
                    print('      coordinates are wrong. Re-measure with anchor_solve.py.')
                else:
                    print('      All anchors mutually consistent. Any remaining error is')
                    print('      COMMON to all of them -- invisible to this test by')
                    print('      construction, and only ground truth can reveal it.')
            elif len(live) == 3:
                print()
                print('      only 3 anchors: zero redundancy, so no self-consistency')
                print('      check is possible. Add D3,D4 to --slots for all five.')
            print()

    if not anchor_slots:
        print('(no anchor slots in --slots; anchor section skipped)')
        print()

    # ---------------- chords ----------------------------------------------
    chord_slots = sorted(i for i in dcol if i not in role_anchor)
    if not chord_slots:
        print('(no chord slots in --slots; add D3,D4 to check the chords)')
        return
    if not args.ring:
        print('=== chord slots ===')
        print('  --ring is needed to say who is in slot 3 and slot 4.')
        print('  e.g.  --ring C23 C24 C25   (slot 3 = next, slot 4 = previous)')
        return

    ring = args.ring
    peer = {}
    for i, ego in enumerate(ring):
        peer[ego] = {3: ring[(i + 1) % len(ring)],
                     4: ring[(i - 1) % len(ring)]}

    print('=== chord slots: ego -> leader (3) and follower (4) ===')
    print('    Chords are frame-independent: both ends are Vicon positions, so')
    print('    no datum or survey error enters. This is the clean measurement.')
    print()
    measured = {}
    chord_fit = []
    for ego in args.robots:
        a = arr.get(ego)
        s_ego = match_subject(gt, ego)
        if a is None or s_ego is None:
            print(f'  {ego}: no data')
            continue
        p_ego = np.array(gt[s_ego][-1][1:4])
        print(f'  {ego}')
        hdr = f'{"t_reply":>9} {"eps ppm":>9}' if rcol else ''
        print(f'      {"slot":>5} {"peer":>6} {"n":>6} {"true":>8} {"meas":>8} '
              f'{"bias":>9} {"noise":>8}{hdr}')
        print('      ' + '-' * (68 if rcol else 56))
        for slot in chord_slots:
            other = peer.get(ego, {}).get(slot)
            if other is None:
                continue
            s_oth = match_subject(gt, other)
            if s_oth is None:
                print(f'      {slot:>5} {other:>6}   no Vicon subject')
                continue
            meas, noise, n = live_mean(a, dcol[slot])
            if meas is None:
                print(f'      {slot:>5} {other:>6} {n:>6}   -- too few --')
                continue
            true = float(np.linalg.norm(p_ego - np.array(gt[s_oth][-1][1:4])))
            bias = meas - true
            measured[(ego, other)] = (meas, noise, true)
            tail = ''
            if slot in rcol:
                t_ms, _, _ = live_mean(a, rcol[slot])
                if t_ms:
                    chord_fit.append((t_ms, bias, f'{ego}->{other}'))
                    tail = (f'{t_ms:>8.2f}ms '
                            f'{EPS_PPM_PER_M_PER_MS*bias/t_ms:>+9.3f}')
            print(f'      {slot:>5} {other:>6} {n:>6} {true:>8.3f} {meas:>8.3f} '
                  f'{bias*1000:>+8.0f}mm {noise*1000:>7.1f}mm{tail}')
        print()

    print('=== reciprocity: the SAME distance, measured from both ends ===')
    print('    No ground truth and no frame needed. The physical distance is')
    print('    identical, so any disagreement beyond the noise is a ranging')
    print('    fault -- and it localises to a specific link.')
    print()
    seen = set()
    print(f'  {"link":>14} {"A->B":>8} {"B->A":>8} {"diff":>9} {"vs noise":>10}')
    print('  ' + '-' * 54)
    worst = 0.0
    for (ego, other) in list(measured):
        key = tuple(sorted((ego, other)))
        if key in seen or (other, ego) not in measured:
            continue
        seen.add(key)
        m1, n1, true = measured[(ego, other)]
        m2, n2, _ = measured[(other, ego)]
        diff = m1 - m2
        comb = math.sqrt(n1 * n1 + n2 * n2)
        worst = max(worst, abs(diff))
        sig = abs(diff) / comb if comb > 0 else float('inf')
        mark = '   <-- INCONSISTENT' if sig > 3.0 else ''
        print(f'  {key[0]+"-"+key[1]:>14} {m1:>8.3f} {m2:>8.3f} '
              f'{diff*1000:>+8.0f}mm {sig:>9.1f}x{mark}')
    if not seen:
        print('  (need both directions of a link -- pass all drones in --robots)')
    else:
        print()
        print(f'  worst disagreement {worst*1000:.0f} mm')
        print('  A reciprocal pair that agrees but disagrees with Vicon means a')
        print('  SHARED bias. A pair that disagrees with ITSELF means one')
        print('  direction is faulty -- the one case calibration cannot fix.')

    if len(chord_fit) >= 3:
        # Pooled across every chord observation. Each drone has only two chord
        # links, too few for a 2-parameter fit on its own, so this assumes the
        # units have similar clock-correction error and asks whether the bias
        # scales with the reply interval at all.
        t = np.array([c[0] for c in chord_fit])
        b = np.array([c[1] for c in chord_fit])
        print()
        print('=== does the chord bias scale with the reply interval? ===')
        print('    Single-sided TWR rescales the reply by an estimated clock')
        print('    correction, so an error eps gives  bias = c*eps*t_reply/2.')
        print('    Crazyflies reply on a contended schedule, so t_reply varies')
        print('    here in a way the anchors\' steady schedule did not.')
        print()
        print(f'  reply-time range {t.min():.2f} - {t.max():.2f} ms  '
              f'(spread {np.ptp(t)*1000:.0f} us)')
        fit = fit_reply_bias(t, b)
        if isinstance(fit, tuple) and fit[0] == 'flat':
            print('  Reply times are effectively equal, so there is still no lever')
            print('  arm. The common bias CANNOT be attributed to the clock from')
            print('  this data -- it is consistent with it, but equally with any')
            print('  other constant offset. A different experiment is needed:')
            print('  vary the anchor TX rate, or compare against classic TWR.')
        elif fit:
            a_off, eps, rms, dof = fit
            spread = float(b.std(ddof=1))
            print(f'  fit  bias = a + k*t_reply   over {len(t)} observations')
            print(f'    a   (reply-independent) {a_off*1000:>+8.0f} mm')
            print(f'    eps (from slope)        {eps:>+8.3f} ppm')
            print(f'    residual rms            {rms*1000:>8.0f} mm  ({dof} dof)')
            print(f'    bias spread before fit  {spread*1000:>8.0f} mm')
            if abs(eps) > 5.0:
                print('    IMPLAUSIBLE eps -- not reply-time driven.')
            elif rms < 0.5 * spread:
                print('    CONFIRMED: the bias scales with reply time. Single-sided')
                print('    TWR clock error. Double-sided TWR would remove it.')
            else:
                print('    NOT explained by reply time: the residual is as large as')
                print('    the spread. The offset is reply-INDEPENDENT, so look at')
                print('    the intercept instead -- a constant in the range')
                print('    computation, not a clock effect.')


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

    h = sub.add_parser('hover', help='analyse a hover: noise vs loop oscillation')
    h.add_argument('--robots', nargs='+', required=True)
    h.add_argument('--duration', type=float, default=60.0, help='seconds')
    h.add_argument('--vicon', action='store_true',
                   help='also record /poses, to separate real motion from error')
    h.add_argument('--plot', help='path prefix for PNG plots')
    h.set_defaults(func=cmd_hover)

    g = sub.add_parser('ranges',
                       help='per-anchor range residual vs Vicon (needs markers '
                            'on the anchors)')
    g.add_argument('--robots', nargs='+', required=True)
    g.add_argument('--duration', type=float, default=60.0, help='seconds')
    g.add_argument('--topic', default='swarm_ranges',
                   help='custom_topics name publishing tdoa3.hmD0..hmD4 '
                        '(default: swarm_ranges)')
    g.add_argument('--slots', default='D0,D1,D2,D3,D4',
                   help='log-block layout, in order. D<n> = tdoa3.hmD<n>, '
                        'RT<n> = tdoa3.hmRT<n>. Max 6. '
                        'default: D0,D1,D2,D3,D4')
    g.add_argument('--ring', nargs='+',
                   help='formation order for the chord slots, e.g. '
                        'C23 C24 C25 -- slot 3 is the NEXT drone, slot 4 the '
                        'PREVIOUS one')
    g.add_argument('--with-reply', action='store_true',
                   help='the log block carries hmD0..2 THEN hmRT0..2 (6 floats, '
                        'exactly one block) -- enables the eps regression')
    g.add_argument('--anchor-slots', default='0,1,2',
                   help='which log slots hold ANCHORS (the rest are chords). '
                        'Use 0,1,2,3,4 when hmLId3/hmLId4 point at anchors '
                        'instead of Crazyflies. default: 0,1,2')
    g.add_argument('--anchor-subjects', nargs='+', default=['A0', 'A1', 'A2'],
                   help='Vicon subject names for the anchors (a _gt-style '
                        'suffix is ignored)')
    g.add_argument('--anchors', nargs='+',
                   help='fallback if the anchors have no markers: positions as '
                        'x,y,z x,y,z x,y,z in the template frame')
    g.set_defaults(func=cmd_ranges)

    a = sub.add_parser('align', help='fit the frame and report accuracy')
    a.add_argument('files', nargs='+', help='JSON files from record --vicon')
    a.set_defaults(func=cmd_align)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
