/**
 * Encirclement controller.
 *
 * Consumes the curve-constrained filter output (theta, r, dz) and emits
 * velocity setpoints. It deliberately does NOT use the free 3D position
 * estimate: with a compact anchor constellation that estimate is amplified by
 * the geometric dilution (GDOP ~4-8 here), while theta and r are constrained
 * to a one-dimensional curve and are an order of magnitude better conditioned.
 *
 * Horizontal axes run in modeVelocity so the stock position loop -- which
 * closes on stateEstimate.x/y -- is bypassed entirely. Altitude runs in
 * modeAbs because the vertical axis is handled well by the existing controller.
 */
#ifndef SWARM_CONTROL_H
#define SWARM_CONTROL_H

#include <stdbool.h>
#include "swarm_curve.h"
#include "swarm_ekf.h"

/** Register params and logs. Call once from appMain before the loop. */
void swarmControlInit(void);

/**
 * One control cycle. Safe to call every loop iteration regardless of state:
 * it does nothing unless armed by swarmCtrl.en AND the filter is healthy.
 *
 * @param f       filter, read-only
 * @param curve   active curve
 * @param dt      seconds since the previous call
 */
void swarmControlUpdate(const swarmEkf_t* f, const swarmCurve_t* curve, float dt);

/** True while the controller is actively commanding. */
bool swarmControlIsEngaged(void);

#endif // SWARM_CONTROL_H
