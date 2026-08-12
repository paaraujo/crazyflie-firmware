#!/usr/bin/env python3
"""
Discover which UWB anchor ids are actually ranging, and their distances.

Usage:
    python3 scan_anchors.py [URI] [MAX_ID] [HM_ID]
    python3 scan_anchors.py radio://0/80/1M/E7E7E7E718 15 254

Why this exists
---------------
`tdoa3.hmAnchLog` selects which anchor id is reported in `tdoa3.hmDist`.
In the firmware (lpsTdoa3Tag.c):

    if (tdoaStorageGetId(anchorCtx) == ctx.logDistAnchorId) {
        ctx.logDistance = distance;
    }

`ctx.logDistance` is ONLY written when a two-way exchange completes for the
selected id. If you select an id that is not present, the variable is never
updated and simply retains the previous anchor's distance -- a stale reading
that looks like a perfectly plausible measurement.

This script distinguishes the two by checking whether the value actually
CHANGES while sampling:
    live   -> the id exists and is ranging   (samples vary, std > 0)
    FROZEN -> no such id / not ranging       (value never changes)
"""

import sys
import time
import statistics

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger

URI = sys.argv[1] if len(sys.argv) > 1 else 'radio://0/80/1M/E7E7E7E718'
MAX_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 15
HM_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 254

SETTLE_S = 1.2
SAMPLES = 25


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


def sample(scf, anchor_id):
    scf.cf.param.set_value('tdoa3.hmAnchLog', anchor_id)
    time.sleep(SETTLE_S)

    lg = LogConfig(name='scan', period_in_ms=100)
    lg.add_variable('tdoa3.hmDist', 'float')
    lg.add_variable('tdoa3.hmSeqOk', 'float')

    dist, seq = [], []
    with SyncLogger(scf, lg) as logger:
        for i, (_, data, _) in enumerate(logger):
            dist.append(data['tdoa3.hmDist'])
            seq.append(data['tdoa3.hmSeqOk'])
            if i >= SAMPLES:
                break

    return {
        'mean': statistics.mean(dist),
        'std': statistics.pstdev(dist) if len(dist) > 1 else 0.0,
        'unique': len(set(dist)),
        'seq': statistics.mean(seq),
    }


def main():
    cflib.crtp.init_drivers()
    print(f'connecting to {URI} ...')
    scf = connect(URI)
    try:
        print('connected\n')

        scf.cf.param.set_value('tdoa3.hmId', HM_ID)
        scf.cf.param.set_value('tdoa3.hmTwr', 1)
        scf.cf.param.set_value('tdoa3.hmTdoa', 1)
        time.sleep(1.5)

        print(f'sweeping hmAnchLog over ids 0..{MAX_ID}')
        print(f'{"id":>4} | {"dist [m]":>10} {"std [m]":>9} {"uniq":>5} | {"hmSeqOk":>8} | verdict')
        print('-' * 68)

        live = {}
        for a in range(MAX_ID + 1):
            r = sample(scf, a)
            # A live id produces changing values; an absent id leaves hmDist frozen.
            is_live = r['std'] > 1e-6 and r['unique'] > 1
            if is_live:
                live[a] = r
                verdict = 'LIVE'
            else:
                verdict = 'frozen (id not ranging)'
            print(f'{a:>4} | {r["mean"]:>10.3f} {r["std"]:>9.4f} {r["unique"]:>5} | '
                  f'{r["seq"]:>8.1f} | {verdict}')

        print()
        print('--- anchors actually ranging ---')
        if not live:
            print('  NONE. Either hmTwr is off, no anchors are in range,')
            print('  or every anchor id is above the scanned range.')
        else:
            for a, r in sorted(live.items()):
                print(f'  id {a:>3} : {r["mean"]:.3f} m  (std {r["std"]:.3f})')
            print()
            print('  Compare these against your tape-measured distances to map')
            print('  physical anchor -> UWB id. The ids are whatever each anchor')
            print('  was configured with; they need not match your physical labels.')
    finally:
        scf.close_link()


if __name__ == '__main__':
    main()
