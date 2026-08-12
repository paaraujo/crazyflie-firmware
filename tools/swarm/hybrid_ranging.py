#!/usr/bin/env python3
"""
Log TWR distances to several remotes simultaneously, using the multi-slot
hybrid-mode logging added to lpsTdoa3Tag.c.

Usage:
    python3 tools/swarm/hybrid_ranging.py [URI] [IDS] [HM_ID] [SECONDS]

    python3 tools/swarm/hybrid_ranging.py
    python3 tools/swarm/hybrid_ranging.py radio://0/80/1M/E7E7E7E718 0,1,2 254 10
    python3 tools/swarm/hybrid_ranging.py radio://0/80/1M/E7E7E7E718 0,1,2,253,252 254 20

IDS may list up to HM_LOG_SLOTS (6) remote ids -- typically the three template
anchors, plus the leader and follower Crazyflies.

Requires firmware built with:
    CONFIG_DECK_LOCO_TDOA3_HYBRID_MODE=y
    CONFIG_DECK_LOCO_ALGORITHM_TDOA3=y
and anchors running in TDoA3 mode.

Unlike the old single-slot approach (sweeping tdoa3.hmAnchLog one id at a time)
every range here is sampled at the same instant, which is what the phase filter
actually needs. The firmware zeroes any slot not refreshed within 500 ms, so a
reading of exactly 0.0 means "no recent measurement" rather than a stale value
left over from a previous id.
"""

import sys
import time
import statistics

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

HM_LOG_SLOTS = 6  # must match HM_LOG_SLOTS in lpsTdoa3Tag.c

URI = sys.argv[1] if len(sys.argv) > 1 else 'radio://0/80/1M/E7E7E7E718'
IDS = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['0', '1', '2'])]
HM_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 254
DURATION_S = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0

if len(IDS) > HM_LOG_SLOTS:
    raise SystemExit(f'at most {HM_LOG_SLOTS} ids can be logged at once (got {len(IDS)})')
if HM_ID in IDS:
    raise SystemExit(f'hmId {HM_ID} must not appear in the logged ids -- a node cannot range to itself')
if HM_ID == 255:
    raise SystemExit('hmId 255 is the broadcast address and must not be used')

# Static configuration. These params are volatile and reset on every reboot.
CONFIG = {
    'tdoa3.hmId': HM_ID,        # this drone's UWB id, unique per drone
    'tdoa3.hmTdoa': 1,          # use received packets for TDoA positioning
    'tdoa3.hmTwr': 1,           # transmit TWR packets
    'tdoa3.hmTwrEstPos': 1,     # feed anchor ranges into the estimator
    'tdoa3.hmTwrTXPos': 0,      # do NOT broadcast own position (avoids neighbours
                                # using us as an anchor -> circular error propagation)
    'tdoa3.hmTofAge': 200,
    'tdoa3.stddev': 0.15,
    'tdoa3.twrStd': 0.25,
}


def connect(uri, attempts=5):
    last = None
    for i in range(1, attempts + 1):
        try:
            scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache'))
            scf.open_link()
            return scf
        except Exception as e:
            last = e
            print(f'  attempt {i}/{attempts} failed: {e}')
            time.sleep(2.0)
    raise SystemExit(f'could not connect: {last}\n'
                     '  (is cfclient still holding the radio?)')


def apply_config(scf):
    """Push static config plus the per-slot id mapping, then read it back."""
    cfg = dict(CONFIG)
    # Map requested ids onto slots; park unused slots on 255 (broadcast) so they
    # can never match a real remote and stay at 0.0.
    for slot in range(HM_LOG_SLOTS):
        cfg[f'tdoa3.hmLId{slot}'] = IDS[slot] if slot < len(IDS) else 255

    print('--- applying configuration ---')
    for name, want in cfg.items():
        scf.cf.param.set_value(name, want)
    time.sleep(0.5)

    ok = True
    for name, want in cfg.items():
        got = scf.cf.param.get_value(name)
        try:
            match = abs(float(got) - float(want)) < 1e-6
        except (TypeError, ValueError):
            match = str(got) == str(want)
        ok &= match
        if not match:
            print(f'    {name:<22} = {got}   <-- MISMATCH, wanted {want}')
    if ok:
        print(f'    all {len(cfg)} parameters accepted')
    print(f'    slot mapping: ' +
          ', '.join(f'hmD{s}->id {IDS[s]}' for s in range(len(IDS))))
    print()
    return ok


def collect(scf, duration_s):
    """Log all six distance slots plus the rate counters, concurrently."""
    dist_rows, rate_rows = [], []

    lg_d = LogConfig(name='hmDist', period_in_ms=100)
    for s in range(HM_LOG_SLOTS):
        lg_d.add_variable(f'tdoa3.hmD{s}', 'float')

    lg_r = LogConfig(name='hmRate', period_in_ms=200)
    for v in ('tdoa3.hmTx', 'tdoa3.hmSeqOk', 'tdoa3.hmEst'):
        lg_r.add_variable(v, 'float')

    scf.cf.log.add_config(lg_d)
    scf.cf.log.add_config(lg_r)
    lg_d.data_received_cb.add_callback(lambda t, d, l: dist_rows.append(d))
    lg_r.data_received_cb.add_callback(lambda t, d, l: rate_rows.append(d))

    print(f'collecting for {duration_s:.0f} s ...')
    lg_d.start()
    lg_r.start()
    time.sleep(duration_s)
    lg_d.stop()
    lg_r.stop()
    print(f'  {len(dist_rows)} distance samples, {len(rate_rows)} rate samples\n')

    return dist_rows, rate_rows


def report(dist_rows, rate_rows):
    print(f'{"id":>4} {"slot":>5} | {"dist [m]":>10} {"std [m]":>9} | '
          f'{"valid":>7} {"n":>5}')
    print('-' * 52)

    summary = {}
    for slot, anchor_id in enumerate(IDS):
        vals = [r[f'tdoa3.hmD{slot}'] for r in dist_rows]
        # The firmware writes exactly 0.0 when a slot has not been refreshed
        # within HM_LOG_MAX_AGE_MS, so 0.0 means "stale/absent", not "0 metres".
        fresh = [v for v in vals if v != 0.0]
        frac = len(fresh) / len(vals) if vals else 0.0

        if fresh:
            mean = statistics.mean(fresh)
            std = statistics.pstdev(fresh) if len(fresh) > 1 else 0.0
            summary[anchor_id] = (mean, std, frac)
            print(f'{anchor_id:>4} {slot:>5} | {mean:>10.3f} {std:>9.4f} | '
                  f'{frac*100:>6.1f}% {len(fresh):>5}')
        else:
            summary[anchor_id] = (None, None, 0.0)
            print(f'{anchor_id:>4} {slot:>5} | {"--":>10} {"--":>9} | '
                  f'{0.0:>6.1f}% {0:>5}   NO DATA')

    if rate_rows:
        tx = statistics.mean(r['tdoa3.hmTx'] for r in rate_rows)
        seq = statistics.mean(r['tdoa3.hmSeqOk'] for r in rate_rows)
        est = statistics.mean(r['tdoa3.hmEst'] for r in rate_rows)
        print(f'\nrates [Hz]:  hmTx {tx:.1f}   hmSeqOk {seq:.1f}   hmEst {est:.1f}')
    else:
        tx = seq = est = 0.0

    diagnose(summary, tx, seq, est)


def diagnose(summary, tx, seq, est):
    print('\n--- diagnosis ---')
    if tx < 0.5:
        print('  hmTx == 0 : not transmitting. Is hmTwr set? Is the Loco deck detected?')
        return
    if seq < 0.5:
        print('  hmTx > 0 but hmSeqOk == 0 : nothing echoes our packets back.')
        print('    -> anchors not in TDoA3 mode, out of range, or hmId collides with an anchor id.')
        return

    missing = [i for i, (m, _, _) in summary.items() if m is None]
    partial = [i for i, (m, _, f) in summary.items() if m is not None and f < 0.9]

    for anchor_id, (mean, std, frac) in summary.items():
        if mean is None:
            print(f'  id {anchor_id}: NO DATA -- this id is not ranging. '
                  f'Wrong id? Run tools/swarm/scan_anchors.py to find the real ones.')
        else:
            print(f'  id {anchor_id}: {mean:.3f} m +/- {std:.3f} m  ({frac*100:.0f}% fresh)')

    if partial:
        print(f'\n  ids {partial} dropped below 90% freshness -- marginal link,')
        print('  obstruction, or contention. Check geometry and anchor spacing.')
    if est < 0.5:
        print('\n  hmEst == 0 : ranges are measured but none reach the estimator.')
        print('  The anchors have not broadcast their positions (LPP). Configure them.')
    if not missing and not partial:
        print('\n  All requested ids ranging cleanly.')


def main():
    cflib.crtp.init_drivers()
    print(f'connecting to {URI} ...')
    scf = connect(URI)
    try:
        print('connected\n')
        try:
            print(f'loco.mode = {scf.cf.param.get_value("loco.mode")}   (expect 3 = TDoA3)\n')
        except Exception:
            print('loco.mode not readable -- is the Loco deck attached?\n')

        apply_config(scf)
        time.sleep(1.0)
        dist_rows, rate_rows = collect(scf, DURATION_S)
        if not dist_rows:
            raise SystemExit('no log data received -- is the firmware built with hybrid mode?')
        report(dist_rows, rate_rows)

        print('\nNOTE: hmTwr is left ENABLED. All tdoa3 params are volatile;')
        print('      rerun this script after every reboot.')
    finally:
        scf.close_link()


if __name__ == '__main__':
    main()
