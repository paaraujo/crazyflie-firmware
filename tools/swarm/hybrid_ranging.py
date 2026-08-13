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
    """Log distances, per-slot reply times, and the rate counters concurrently."""
    dist_rows, reply_rows, rate_rows = [], [], []

    lg_d = LogConfig(name='hmDist', period_in_ms=100)
    for s in range(HM_LOG_SLOTS):
        lg_d.add_variable(f'tdoa3.hmD{s}', 'float')

    # Reply time per slot. Recorded by the firmware BEFORE the accept/reject
    # decision, so rejected samples appear here too.
    lg_t = LogConfig(name='hmReply', period_in_ms=100)
    for s in range(HM_LOG_SLOTS):
        lg_t.add_variable(f'tdoa3.hmRT{s}', 'float')

    lg_r = LogConfig(name='hmRate', period_in_ms=200)
    for v in ('tdoa3.hmTx', 'tdoa3.hmSeqOk', 'tdoa3.hmEst',
              'tdoa3.hmRejR', 'tdoa3.hmRejO', 'tdoa3.hmCcPpm'):
        lg_r.add_variable(v, 'float')

    for cfg, sink in ((lg_d, dist_rows), (lg_t, reply_rows), (lg_r, rate_rows)):
        scf.cf.log.add_config(cfg)
        cfg.data_received_cb.add_callback(lambda t, d, l, s=sink: s.append(d))

    print(f'collecting for {duration_s:.0f} s ...')
    for cfg in (lg_d, lg_t, lg_r):
        cfg.start()
    time.sleep(duration_s)
    for cfg in (lg_d, lg_t, lg_r):
        cfg.stop()
    print(f'  {len(dist_rows)} distance, {len(reply_rows)} reply, '
          f'{len(rate_rows)} rate samples\n')

    return dist_rows, reply_rows, rate_rows


def percentile(values, p):
    """Linear-interpolated percentile; avoids a numpy dependency."""
    if not values:
        return float('nan')
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def reply_report(reply_rows):
    """Compare reply-time distributions per remote.

    The hypothesis under test: clock-correction error is amplified by the reply
    time (~3 m of range error per ppm per 20 ms), so Crazyflie-to-Crazyflie
    links -- which have a randomised, contended transmit schedule -- should show
    markedly longer replies than anchors, and that is where the outliers live.
    """
    print('reply time per remote [ms] -- long replies amplify clock error')
    print(f'{"id":>4} {"slot":>5} | {"median":>8} {"p90":>8} {"p99":>8} {"max":>8}')
    print('-' * 50)

    stats = {}
    for slot, remote in enumerate(IDS):
        vals = [r[f'tdoa3.hmRT{slot}'] for r in reply_rows]
        vals = [v for v in vals if v > 0.0]
        if not vals:
            print(f'{remote:>4} {slot:>5} |     no reply-time samples')
            continue
        med, p90 = percentile(vals, 50), percentile(vals, 90)
        p99, mx = percentile(vals, 99), max(vals)
        stats[remote] = (med, p90, p99, mx)
        print(f'{remote:>4} {slot:>5} | {med:>8.2f} {p90:>8.2f} {p99:>8.2f} {mx:>8.2f}')

    if stats:
        print()
        print('  predicted range error at 1 ppm clock error '
              '(error = c * ppm * t_reply / 2):')
        for remote, (med, _, p99, _) in stats.items():
            err_med = 3e8 * 1e-6 * (med * 1e-3) / 2.0
            err_p99 = 3e8 * 1e-6 * (p99 * 1e-3) / 2.0
            print(f'    id {remote:>3}: {err_med:6.2f} m at median, '
                  f'{err_p99:6.2f} m at p99')
    return stats


def correlate(dist_rows, reply_rows, reply_stats):
    """Test whether long replies actually coincide with bad distances.

    Splits each remote's samples into a low-reply-time group and a high-reply-time
    group (below median vs above p90) and compares the distance spread in each.
    If the reply-time hypothesis holds, the high group should be markedly noisier.

    Caveat: distances and reply times arrive in separate log blocks, so they are
    only approximately aligned (both are logged at 100 ms). This is good enough
    to expose a strong effect, not to measure it precisely.
    """
    if not reply_stats:
        return

    print('\n--- does reply time explain the outliers? ---')
    print(f'{"id":>4} | {"spread, fast replies":>21} | {"spread, slow replies":>21} | ratio')
    print('-' * 74)

    verdicts = []
    for slot, remote in enumerate(IDS):
        if remote not in reply_stats:
            continue
        n = min(len(dist_rows), len(reply_rows))
        pairs = [(reply_rows[i][f'tdoa3.hmRT{slot}'],
                  dist_rows[i][f'tdoa3.hmD{slot}'])
                 for i in range(n)
                 if dist_rows[i][f'tdoa3.hmD{slot}'] != 0.0
                 and reply_rows[i][f'tdoa3.hmRT{slot}'] > 0.0]
        if len(pairs) < 20:
            continue

        med = reply_stats[remote][0]
        p90 = reply_stats[remote][1]
        fast = [d for rt, d in pairs if rt <= med]
        slow = [d for rt, d in pairs if rt >= p90]
        if len(fast) < 5 or len(slow) < 5:
            continue

        sd_fast = statistics.pstdev(fast)
        sd_slow = statistics.pstdev(slow)
        ratio = (sd_slow / sd_fast) if sd_fast > 1e-9 else float('inf')
        verdicts.append((remote, ratio))
        print(f'{remote:>4} | {sd_fast:>18.4f} m | {sd_slow:>18.4f} m | {ratio:>5.1f}x')

    if verdicts:
        worst = max(r for _, r in verdicts)
        print()
        if worst > 3.0:
            print('  CONFIRMED: slow replies are markedly noisier. The reply-time gate')
            print('  (tdoa3.hmMaxReply) should remove most outliers. Suggested starting')
            print('  point: set it just above the median reply time of your ANCHOR links.')
        else:
            print('  NOT CONFIRMED: noise does not track reply time. The outliers are')
            print('  likely multipath or NLOS instead, so rely on the median filter')
            print('  (tdoa3.hmOutTh) rather than the reply-time gate.')


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
        rej_r = statistics.mean(r['tdoa3.hmRejR'] for r in rate_rows)
        rej_o = statistics.mean(r['tdoa3.hmRejO'] for r in rate_rows)
        cc = statistics.mean(abs(r['tdoa3.hmCcPpm']) for r in rate_rows)
        print(f'\nrates [Hz]:  hmTx {tx:.1f}   hmSeqOk {seq:.1f}   hmEst {est:.1f}')
        print(f'rejected  :  reply-time {rej_r:.1f}/s   outlier {rej_o:.1f}/s')
        print(f'clock corr:  |{cc:.3f}| ppm mean deviation')
        if rej_r == 0.0 and rej_o == 0.0:
            print('  (both rejection thresholds are 0 = disabled; '
                  'set tdoa3.hmMaxReply / tdoa3.hmOutTh to enable)')
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
        dist_rows, reply_rows, rate_rows = collect(scf, DURATION_S)
        if not dist_rows:
            raise SystemExit('no log data received -- is the firmware built with hybrid mode?')
        report(dist_rows, rate_rows)
        print()
        reply_stats = reply_report(reply_rows)
        correlate(dist_rows, reply_rows, reply_stats)

        print('\nNOTE: hmTwr is left ENABLED. All tdoa3 params are volatile;')
        print('      rerun this script after every reboot.')
    finally:
        scf.close_link()


if __name__ == '__main__':
    main()
