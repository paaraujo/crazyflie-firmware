#!/usr/bin/env python3
"""
Configure every Crazyflie in the swarm for hybrid-mode ranging.

Usage:
    python3 tools/swarm/setup_swarm.py                       # use DRONES below
    python3 tools/swarm/setup_swarm.py URI=ID [URI=ID ...]   # override on the CLI

    python3 tools/swarm/setup_swarm.py \
        radio://0/80/2M/E7E7E7E718=254 \
        radio://0/80/2M/E7E7E7E719=253 \
        radio://0/80/2M/E7E7E7E71A=252

Each drone is visited in turn, given a unique hmId, and left transmitting.
The tdoa3 parameters are volatile, so they survive until that drone reboots --
which is what lets a single Crazyradio configure the whole swarm sequentially
and then have every drone ranging simultaneously.

Logging slots are assigned per drone as:
    hmD0..hmD2  ->  the three template anchors
    hmD3        ->  this drone's LEADER   (next drone, CCW)
    hmD4        ->  this drone's FOLLOWER (previous drone)
    hmD5        ->  unused (parked on 255)

Leader/follower follow the order of the drone list, wrapping around, so the
same script works for any swarm size up to the number of logging slots.
"""

import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

HM_LOG_SLOTS = 6  # must match HM_LOG_SLOTS in lpsTdoa3Tag.c

# Edit for your setup, or override on the command line.
DRONES = {
    'radio://0/80/2M/E7E7E7E718': 254,  # CF24
    'radio://0/80/2M/E7E7E7E719': 253,  # CF25
    'radio://0/80/2M/E7E7E7E717': 252,  # CF23
}

ANCHORS = [0, 1, 2]     # the three L-template anchor ids

# Static config, identical on every drone. Mirrors hybrid_ranging.py.
CONFIG = {
    'tdoa3.hmTdoa': 1,        # use received packets for TDoA positioning
    'tdoa3.hmTwr': 1,         # transmit TWR packets
    'tdoa3.hmTwrEstPos': 1,   # feed anchor ranges into the estimator
    'tdoa3.hmTwrTXPos': 0,    # do NOT broadcast own position -- prevents neighbours
                              # using us as an anchor (circular error propagation)
    'tdoa3.hmTofAge': 200,
    'tdoa3.stddev': 0.1,
    'tdoa3.twrStd': 0.1,
}


def parse_args(argv):
    if not argv:
        return dict(DRONES)
    drones = {}
    for arg in argv:
        if '=' not in arg:
            raise SystemExit(f'expected URI=ID, got "{arg}"')
        uri, sid = arg.rsplit('=', 1)
        drones[uri] = int(sid)
    return drones


def validate(drones):
    ids = list(drones.values())
    if len(set(ids)) != len(ids):
        raise SystemExit(f'hmId values must be unique, got {ids}')
    for i in ids:
        if i == 255:
            raise SystemExit('hmId 255 is the broadcast address and must not be used')
        if i in ANCHORS:
            raise SystemExit(f'hmId {i} collides with an anchor id {ANCHORS}')
    if len(ANCHORS) + 2 > HM_LOG_SLOTS:
        raise SystemExit(f'{len(ANCHORS)} anchors + leader + follower exceeds '
                         f'{HM_LOG_SLOTS} logging slots')


def slot_map(uris, index, drones):
    """Logging slots for the drone at position `index` in the ring."""
    n = len(uris)
    leader = drones[uris[(index + 1) % n]]      # ahead, CCW
    follower = drones[uris[(index - 1) % n]]    # behind

    slots = list(ANCHORS) + [leader, follower]
    slots += [255] * (HM_LOG_SLOTS - len(slots))   # park unused slots
    return slots, leader, follower


def configure(uri, hm_id, slots):
    scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache'))
    scf.open_link()
    try:
        cfg = dict(CONFIG)
        cfg['tdoa3.hmId'] = hm_id
        for s, remote in enumerate(slots):
            cfg[f'tdoa3.hmLId{s}'] = remote

        for name, want in cfg.items():
            scf.cf.param.set_value(name, want)
        time.sleep(0.4)

        bad = []
        for name, want in cfg.items():
            got = scf.cf.param.get_value(name)
            try:
                if abs(float(got) - float(want)) > 1e-6:
                    bad.append((name, got, want))
            except (TypeError, ValueError):
                if str(got) != str(want):
                    bad.append((name, got, want))
        return bad
    finally:
        scf.close_link()


def main():
    drones = parse_args(sys.argv[1:])
    validate(drones)
    uris = list(drones.keys())

    cflib.crtp.init_drivers()
    print(f'configuring {len(uris)} drone(s)\n')

    failed = []
    for i, uri in enumerate(uris):
        hm_id = drones[uri]
        slots, leader, follower = slot_map(uris, i, drones)
        print(f'{uri}')
        print(f'    hmId {hm_id}   leader {leader}   follower {follower}')
        print(f'    slots: ' + ', '.join(
            f'hmD{s}->{r}' for s, r in enumerate(slots) if r != 255))
        try:
            bad = configure(uri, hm_id, slots)
            if bad:
                print('    FAILED to set:')
                for name, got, want in bad:
                    print(f'      {name} = {got}, wanted {want}')
                failed.append(uri)
            else:
                print('    OK')
        except Exception as e:
            print(f'    ERROR: {e}')
            failed.append(uri)
        print()

    if failed:
        print(f'{len(failed)} drone(s) failed: {failed}')
        print('  If a param is "not in TOC", that drone is not running hybrid-mode')
        print('  firmware. Build with: make cf21bl_swarm_defconfig && make')
        raise SystemExit(1)

    print('All drones configured and transmitting.')
    print()
    print('Parameters are VOLATILE -- rerun this after any drone reboots.')
    print('Now log one drone\'s five ranges with, for example:')
    first = uris[0]
    slots, leader, follower = slot_map(uris, 0, drones)
    ids = ','.join(str(r) for r in slots if r != 255)
    print(f'    python3 tools/swarm/hybrid_ranging.py {first} {ids} {drones[first]} 20')


if __name__ == '__main__':
    main()
