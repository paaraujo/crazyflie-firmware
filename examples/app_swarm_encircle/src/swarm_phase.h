/**
 * Radius and phase derived from the onboard state estimate.
 *
 * This is an INDEPENDENT CROSS-CHECK on the range-driven filter, and the source
 * used to seed it. It comes from a different measurement path -- the IMU and the
 * anchor ranges, through the vehicle's own EKF -- so a disagreement with the
 * filter is evidence that the filter has diverged or latched onto the wrong
 * branch of the chord's ahead/behind ambiguity.
 *
 * The phase is exact rather than approximate: for the tangential embedding of
 * swarm_curve.h, atan2(q_y - c_y, q_x - c_x) = theta identically, however severe
 * the elevation profile.
 */

#ifndef __SWARM_PHASE_H__
#define __SWARM_PHASE_H__

#include <stdbool.h>

#include "swarm_curve.h"

/** Resolve the state-estimate log ids. False if they are unavailable. */
bool swarmPhaseInit(void);

/** Recompute from the current state estimate. Call periodically. */
void swarmPhaseUpdate(const swarmCurve_t* curve);

/**
 * Latest phase [rad] and distance from the encirclement centre [m].
 * False when the agent is on the orbit axis, where azimuth is undefined.
 */
bool swarmPhaseGet(float* theta, float* r);

#endif // __SWARM_PHASE_H__
