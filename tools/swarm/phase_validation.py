#!/usr/bin/env python3
"""
Phase 3 validation: compute radius and phase from the EKF position, and check
the chord measurement model against ground truth derived from those positions.

Usage:
    python3 tools/swarm/phase_validation.py [SECONDS]
    python3 tools/swarm/phase_validation.py 8

Edit DRONES and ANCHORS below for your setup.

What this checks, and why
-------------------------
Once three anchors make position directly observable, radius and phase are NOT
estimated -- they are computed from the EKF position:

    r     = sqrt(x^2 + y^2)        horizontal distance from the orbit axis
    theta = atan2(y, x)            phase, CCW from the +x axis

(theta = atan2(y, x) holds while the embedding is the flat circle, R_e = I. A
distorted embedding needs the fixed-point inversion of theory.tex 4.1.)

With all drones' positions known we get an INDEPENDENT ground truth for the
phase separations, so the chord measurement model can be validated before any
filter is written. Three quantities are compared per pair:

    1. measured chord   d_meas          two-way ranging
    2. geometric chord  ||p_i - p_j||   from the two EKF positions
    3. model chord      2*r*sin(D/2)    flat-circle formula

    1 vs 2  validates the RANGING          (do TWR and the EKF positions agree?)
    2 vs 3  validates the MODEL ASSUMPTION (are the drones really on a common
                                            circle, same radius and height?)

If 2 and 3 disagree, the simplified chord formula is not applicable to your
formation and the filter must use the general ||q(theta_i) - q(theta_j)||.

The drones are assumed STATIC. A single Crazyradio serves one link at a time,
so they are visited sequentially and each is averaged over the collection
window; that is only valid while nothing is moving.
"""

import sys
import math
import time
import statistics
import threading

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

HM_LOG_SLOTS = 6

DURATION_S = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0

# Edit for your setup. Order defines the ring: each drone's LEADER is the next
# entry (CCW) and its FOLLOWER is the previous one.
DRONES = {
    'radio://0/80/2M/E7E7E7E718': 254,  # CF24
    'radio://0/80/2M/E7E7E7E719': 253,  # CF25
    'radio://0/80/2M/E7E7E7E717': 252,  # CF23
}

# theory.tex roles -> anchor ids.
#   A0 = origin / encirclement centre
#   A1 = defines +x, i.e. the theta = 0 direction
#   A2 = defines +y, i.e. the CCW sense
ANCHORS = {'A0': 0, 'A1': 1, 'A2': 2}

CONFIG = {
    'tdoa3.hmTdoa': 1,
    'tdoa3.hmTwr': 1,
    'tdoa3.hmTwrEstPos': 1,
    'tdoa3.hmTwrTXPos': 0,
    'tdoa3.hmTofAge': 200,
    'tdoa3.stddev': 0.1,
    'tdoa3.twrStd': 0.1,
    'tdoa3.hmOutTh': 0.5,     # median outlier filter, metres
    'tdoa3.hmMaxReply': 0.0,  # reply-time gate disabled (not the mechanism)
}


def wrap(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def connect(uri, attempts=5):
    last = None
    for i in range(1, attempts + 1):
        try:
            scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache'))
            scf.open_link()
            return scf
        except Exception as e:
            last = e
            print(f'    attempt {i}/{attempts}: {e}')
            time.sleep(2.0)
    raise SystemExit(f'could not connect to {uri}: {last}\n'
                     '  (is cfclient still holding the radio?)')


def read_anchor_positions(scf, timeout=6.0):
    """Read the anchor positions the drone itself is using.

    This is the frame the EKF position is expressed in, so it is the only
    authoritative answer to "where does the drone think the anchors are".
    Returns {} if the memory cannot be read.
    """
    try:
        from cflib.crazyflie.mem import MemoryElement
    except Exception:
        return {}

    try:
        mems = scf.cf.mem.get_mems(MemoryElement.TYPE_LOCO2)
        if not mems:
            return {}
        mem = mems[0]

        got_ids = threading.Event()
        got_data = threading.Event()
        mem.update(lambda m: got_ids.set())
        if not got_ids.wait(timeout):
            return {}
        mem.update_data(lambda m: got_data.set())
        if not got_data.wait(timeout):
            return {}

        out = {}
        for aid, data in mem.anchor_data.items():
            if getattr(data, 'is_valid', True):
                p = data.position
                out[int(aid)] = (float(p[0]), float(p[1]), float(p[2]))
        return out
    except Exception:
        return {}


def collect(uri, hm_id, remote_ids):
    """Configure one drone, then average its position and chord distances."""
    scf = connect(uri)
    try:
        cfg = dict(CONFIG)
        cfg['tdoa3.hmId'] = hm_id
        for s in range(HM_LOG_SLOTS):
            cfg[f'tdoa3.hmLId{s}'] = remote_ids[s] if s < len(remote_ids) else 255
        for name, want in cfg.items():
            scf.cf.param.set_value(name, want)
        time.sleep(1.0)

        anchors = read_anchor_positions(scf)

        pos_rows, chord_rows = [], []
        lg_p = LogConfig(name='pos', period_in_ms=100)
        for v in ('stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z'):
            lg_p.add_variable(v, 'float')
        lg_c = LogConfig(name='chord', period_in_ms=100)
        for s in range(HM_LOG_SLOTS):
            lg_c.add_variable(f'tdoa3.hmD{s}', 'float')

        for cf_cfg, sink in ((lg_p, pos_rows), (lg_c, chord_rows)):
            scf.cf.log.add_config(cf_cfg)
            cf_cfg.data_received_cb.add_callback(lambda t, d, l, s=sink: s.append(d))
            cf_cfg.start()
        time.sleep(DURATION_S)
        for cf_cfg in (lg_p, lg_c):
            cf_cfg.stop()

        if not pos_rows:
            raise SystemExit(f'{uri}: no position data -- is the EKF running?')

        pos = tuple(statistics.mean(r[f'stateEstimate.{a}'] for r in pos_rows)
                    for a in ('x', 'y', 'z'))
        pos_sd = tuple(statistics.pstdev(r[f'stateEstimate.{a}'] for r in pos_rows)
                       for a in ('x', 'y', 'z'))

        chords = {}
        for s, rid in enumerate(remote_ids):
            vals = [r[f'tdoa3.hmD{s}'] for r in chord_rows]
            fresh = [v for v in vals if v != 0.0]   # firmware zeroes stale slots
            chords[rid] = statistics.mean(fresh) if fresh else None

        return {'pos': pos, 'pos_sd': pos_sd, 'chords': chords, 'anchors': anchors}
    finally:
        scf.close_link()


def main():
    cflib.crtp.init_drivers()
    uris = list(DRONES.keys())
    ids = [DRONES[u] for u in uris]
    n = len(uris)

    # Every drone logs the three anchors plus every other drone.
    print(f'visiting {n} drone(s), {DURATION_S:.0f} s each '
          f'(drones must be STATIONARY)\n')

    data = {}
    for i, uri in enumerate(uris):
        others = [ids[j] for j in range(n) if j != i]
        remotes = [ANCHORS['A0'], ANCHORS['A1'], ANCHORS['A2']] + others
        print(f'{uri}  (hmId {ids[i]})')
        data[ids[i]] = collect(uri, ids[i], remotes)
        p, sd = data[ids[i]]['pos'], data[ids[i]]['pos_sd']
        print(f'    position  ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m'
              f'   sd ({sd[0]:.3f}, {sd[1]:.3f}, {sd[2]:.3f})')
        print()

    report_frame(data, uris, ids)
    phases = report_phases(data, ids)
    report_chords(data, ids, phases)


def report_frame(data, uris, ids):
    anchors = {}
    for i in ids:
        if data[i]['anchors']:
            anchors = data[i]['anchors']
            break
    if not anchors:
        print('--- anchor frame ---')
        print('  could not read anchor positions from the drone memory;')
        print('  assuming the configured ANCHORS mapping is correct.\n')
        return

    print('--- anchor frame, as the drone sees it ---')
    for role in ('A0', 'A1', 'A2'):
        aid = ANCHORS[role]
        p = anchors.get(aid)
        if p is None:
            print(f'  {role} = id {aid}: NOT KNOWN to the drone')
            continue
        print(f'  {role} = id {aid}: ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m')

    a0 = anchors.get(ANCHORS['A0'])
    a1 = anchors.get(ANCHORS['A1'])
    a2 = anchors.get(ANCHORS['A2'])
    if a0 and a1 and a2:
        print()
        if max(abs(c) for c in a0) > 0.05:
            print(f'  NOTE: A0 is not at the origin. theta and r are measured about')
            print(f'        the origin of the anchor frame, not about A0.')
        bx = [a1[k] - a0[k] for k in range(3)]
        by = [a2[k] - a0[k] for k in range(3)]
        lx, ly = math.dist(a1, a0), math.dist(a2, a0)
        dot = sum(bx[k] * by[k] for k in range(3))
        gamma = math.degrees(math.acos(max(-1.0, min(1.0, dot / (lx * ly)))))
        print(f'  baseline A0->A1 (x) = {lx:.3f} m')
        print(f'  baseline A0->A2 (y) = {ly:.3f} m')
        print(f'  angle between them  = {gamma:.1f} deg  (90 is ideal, but only')
        print(f'                        accuracy of the KNOWN value matters)')
    print()


def report_phases(data, ids):
    print('--- radius and phase, computed from the EKF position ---')
    print(f'{"hmId":>5} | {"r [m]":>8} {"theta [deg]":>12} {"z [m]":>8}')
    print('-' * 40)
    phases = {}
    for i in ids:
        x, y, z = data[i]['pos']
        r = math.hypot(x, y)
        th = math.atan2(y, x)
        phases[i] = (r, th, z)
        print(f'{i:>5} | {r:>8.3f} {math.degrees(th):>12.1f} {z:>8.3f}')

    rs = [phases[i][0] for i in ids]
    zs = [phases[i][2] for i in ids]
    print()
    print(f'  radius spread {max(rs)-min(rs):.3f} m, height spread {max(zs)-min(zs):.3f} m')
    if max(rs) - min(rs) > 0.15 or max(zs) - min(zs) > 0.15:
        print('  WARNING: the drones are not on a common circle. The simplified')
        print('  chord formula 2*r*sin(D/2) assumes equal radius AND height, so')
        print('  expect the model residual below to be large.')
    print()
    return phases


def report_chords(data, ids, phases):
    print('--- chord model validation ---')
    print(f'{"pair":>11} | {"measured":>9} {"geometric":>10} {"model":>8} | '
          f'{"rng err":>8} {"mdl err":>8} | {"true D":>7} {"est D":>7}')
    print('-' * 82)

    r_mean = statistics.mean(phases[i][0] for i in ids)
    rng_errs, mdl_errs, sep_errs = [], [], []

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            d_meas = data[i]['chords'].get(j)
            if d_meas is None:
                print(f'{i:>4} - {j:<4} |  no chord measurement')
                continue

            pi, pj = data[i]['pos'], data[j]['pos']
            d_geom = math.dist(pi, pj)

            dth = wrap(phases[i][1] - phases[j][1])
            d_model = 2 * r_mean * abs(math.sin(dth / 2))

            rng_err = d_meas - d_geom          # ranging vs EKF positions
            mdl_err = d_geom - d_model         # circle assumption
            rng_errs.append(abs(rng_err))
            mdl_errs.append(abs(mdl_err))

            # Inverse: recover the separation from the measured chord, as the
            # filter will. arcsin saturates if the chord exceeds the diameter.
            ratio = d_meas / (2 * r_mean)
            if ratio > 1.0:
                est_txt = ' >180'
            else:
                est = 2 * math.asin(ratio)
                sep_errs.append(abs(math.degrees(est) - abs(math.degrees(dth))))
                est_txt = f'{math.degrees(est):7.1f}'

            print(f'{i:>4} - {j:<4} | {d_meas:>9.3f} {d_geom:>10.3f} {d_model:>8.3f} | '
                  f'{rng_err:>+8.3f} {mdl_err:>+8.3f} | '
                  f'{math.degrees(dth):>7.1f} {est_txt}')

    print()
    if rng_errs:
        print(f'  ranging error  |measured - geometric|  mean {statistics.mean(rng_errs):.3f} m')
    if mdl_errs:
        print(f'  model error    |geometric - model|     mean {statistics.mean(mdl_errs):.3f} m')
    if sep_errs:
        print(f'  separation error, chord-inverted vs true: mean {statistics.mean(sep_errs):.1f} deg')

    print()
    print('--- interpretation ---')
    if rng_errs and statistics.mean(rng_errs) > 0.10:
        print('  Ranging and the EKF positions disagree by >10 cm. Suspect an')
        print('  antenna-delay bias, or anchor positions that do not match reality.')
    elif rng_errs:
        print('  Ranging agrees with the EKF positions: TWR and the position')
        print('  estimate are consistent.')
    if mdl_errs and statistics.mean(mdl_errs) > 0.10:
        print('  The flat-circle chord formula does not describe this formation.')
        print('  Use the general ||q(theta_i) - q(theta_j)|| in the filter rather')
        print('  than 2*r*sin(D/2), or level the drones onto a common circle.')
    elif mdl_errs:
        print('  The flat-circle chord formula fits: 2*r*sin(D/2) is usable as the')
        print('  measurement model for the differential filter.')
    print()
    print('  NOTE: the chord is even in the phase difference, so the inversion')
    print('  recovers |D| only. Ahead vs behind must come from the ring ordering')
    print('  and filter continuity (theory.tex 7.7).')


if __name__ == '__main__':
    main()
