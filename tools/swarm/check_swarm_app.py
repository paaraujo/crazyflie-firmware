#!/usr/bin/env python3
"""
Verify the onboard swarm_phase app against the same computation done offboard.

Usage:
    python3 tools/swarm/check_swarm_app.py [URI] [SECONDS]
    python3 tools/swarm/check_swarm_app.py radio://0/80/2M/E7E7E7E718 5

Logs stateEstimate.x/y/z together with the app's swarmPhase.r / swarmPhase.th outputs,
recomputes r and theta in Python from the same positions, and compares.

This isolates ONE thing: is the firmware arithmetic correct? It says nothing
about whether the underlying position estimate is any good -- that is what
phase_validation.py is for. A clean pass here plus a bad position still means a
bad phase; the point is to rule the app code in or out.
"""

import sys
import math
import time
import statistics

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

URI = sys.argv[1] if len(sys.argv) > 1 else 'radio://0/80/2M/E7E7E7E718'
DURATION_S = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0


def main():
    cflib.crtp.init_drivers()
    print(f'connecting to {URI} ...')
    scf = SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache'))
    scf.open_link()
    try:
        # If the swarm group is absent, the app is not in the flashed binary.
        groups = scf.cf.log.toc.toc
        if 'swarmPhase' not in groups:
            raise SystemExit(
                'ERROR: no "swarmPhase" log group on this drone.\n'
                '  The app is not in the flashed firmware. Build and flash:\n'
                '    cd examples/app_swarm_encircle\n'
                '    make cf21bl_swarm_defconfig && make -j$(nproc)\n'
                '    cfloader flash build/cf21bl.bin stm32-fw -w ' + URI)
        print('swarmPhase log group found:', sorted(groups['swarmPhase']))

        cx = scf.cf.param.get_value('swarmCurve.cx')
        cy = scf.cf.param.get_value('swarmCurve.cy')
        cz = scf.cf.param.get_value('swarmCurve.cz')
        print(f'orbit centre ({float(cx):+.3f}, {float(cy):+.3f}), cz {float(cz):.3f} m\n')
        cx, cy, cz = float(cx), float(cy), float(cz)

        rows = []
        lg = LogConfig(name='chk', period_in_ms=100)
        for v in ('stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
                  'swarmPhase.r', 'swarmPhase.th', 'swarmPhase.valid'):
            lg.add_variable(v, 'float')
        scf.cf.log.add_config(lg)
        lg.data_received_cb.add_callback(lambda t, d, l: rows.append(d))

        print(f'collecting for {DURATION_S:.0f} s ...')
        lg.start()
        time.sleep(DURATION_S)
        lg.stop()
        if not rows:
            raise SystemExit('no data received')
        print(f'  {len(rows)} samples\n')

        dr, dth, invalid = [], [], 0
        for row in rows:
            x, y = row['stateEstimate.x'], row['stateEstimate.y']
            z = row['stateEstimate.z']
            if row['swarmPhase.valid'] < 0.5:
                invalid += 1
                continue
            # 3D distance from the encirclement centre, matching the firmware.
            # The curve lies on the SPHERE of radius r about c, so the radius is
            # the full norm; the horizontal distance would only agree for an
            # agent at exactly z = cz.
            r_exp = math.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
            th_exp = math.atan2(y - cy, x - cx)
            dr.append(abs(row['swarmPhase.r'] - r_exp))
            # wrap the phase difference so the +/-pi boundary is not an error
            d = (row['swarmPhase.th'] - th_exp + math.pi) % (2 * math.pi) - math.pi
            dth.append(abs(d))

        last = rows[-1]
        print(f'last sample: pos ({last["stateEstimate.x"]:+.3f}, '
              f'{last["stateEstimate.y"]:+.3f}, {last["stateEstimate.z"]:+.3f})')
        print(f'             onboard r {last["swarmPhase.r"]:.3f} m, '
              f'theta {math.degrees(last["swarmPhase.th"]):+.1f} deg\n')

        if invalid:
            print(f'  {invalid} sample(s) had swarmPhase.valid = 0 '
                  f'(agent on the orbit axis, phase undefined)')
        if not dr:
            raise SystemExit('no valid samples to compare')

        print(f'onboard vs offboard:')
        print(f'  radius mismatch  max {max(dr)*1000:.2f} mm   mean {statistics.mean(dr)*1000:.2f} mm')
        print(f'  phase  mismatch  max {math.degrees(max(dth))*1000:.2f} mdeg '
              f'  mean {math.degrees(statistics.mean(dth))*1000:.2f} mdeg')

        # Both sides use float32-ish arithmetic on values logged at 100 ms, so a
        # small mismatch is expected; anything large means a real disagreement.
        ok = max(dr) < 0.002 and math.degrees(max(dth)) < 0.5
        print()
        if ok:
            print('  PASS: the onboard computation matches the offboard one.')
            print('  NOTE: this only validates the arithmetic. Whether the phase is')
            print('  physically correct depends on the EKF position -- use')
            print('  phase_validation.py for that.')
        else:
            print('  MISMATCH. Likely causes:')
            print('    - swarmCurve.cx/cy differ from what this script read (changed mid-run)')
            print('    - the two log blocks are not sampling the same instant while moving')
            print('      (re-run with the drone stationary)')
    finally:
        scf.close_link()


if __name__ == '__main__':
    main()
