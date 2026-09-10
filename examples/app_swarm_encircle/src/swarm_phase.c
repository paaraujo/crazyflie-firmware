/**
 * Swarm encirclement -- radius and phase derived from the state estimate.
 *
 * ROLE: this is an INDEPENDENT CROSS-CHECK, not the primary phase source.
 *
 * The recommended estimator constrains the state to the curve and fuses the UWB
 * ranges directly, so it never consults the EKF position. That is the right
 * architecture -- collapsing three degrees of freedom to one is far better
 * conditioned than trilaterating from three marginal ranges -- but it also
 * leaves nothing independently checking the filter's phase. This module supplies
 * that second opinion, from a completely different measurement path (EKF
 * position, driven by the IMU and the anchor ranges).
 *
 * Disagreement between swarmPhase.th here and the filter's phase is the signal that
 * the filter has locked onto the wrong branch of the chord ambiguity, or has
 * diverged.
 *
 * The phase is exact, not approximate: for the tangential embedding of
 * swarm_curve.h, atan2(q_y - c_y, q_x - c_x) = theta identically, for any
 * elevation profile. The radius uses the FULL 3D distance from the encirclement
 * centre, which equals r on the curve for any embedding -- the horizontal
 * distance would only be correct for a level circle.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "log.h"
#include "param.h"

#include "swarm_curve.h"
#include "swarm_phase.h"

static struct {
  float r;         // distance from the encirclement centre [m]
  float theta;     // phase, CCW from +x [rad]
  float thetaDeg;  // same, in degrees, for convenience when reading logs
  float zErr;      // height error relative to the curve at this phase [m]
  float bDeg;      // elevation of the curve at this phase [deg]
  uint8_t valid;   // 1 when the phase is defined
} ctx;

static logVarId_t idX, idY, idZ;

bool swarmPhaseInit(void) {
  idX = logGetVarId("stateEstimate", "x");
  idY = logGetVarId("stateEstimate", "y");
  idZ = logGetVarId("stateEstimate", "z");

  const bool ok = logVarIdIsValid(idX) && logVarIdIsValid(idY) && logVarIdIsValid(idZ);
  if (!ok) {
    ctx.valid = 0;
  }
  return ok;
}

void swarmPhaseUpdate(const swarmCurve_t* curve) {
  const float p[3] = {
    logGetFloat(idX),
    logGetFloat(idY),
    logGetFloat(idZ),
  };

  // Distance from the encirclement centre. Equals r exactly when the agent is
  // on the curve, whatever the elevation profile, because a rotation preserves
  // length. The horizontal distance would only be correct for a level circle.
  ctx.r = swarmCurveRadius(curve, p);

  const float dx = p[0] - curve->cx;
  const float dy = p[1] - curve->cy;
  const float horiz = sqrtf(dx * dx + dy * dy);

  // atan2 is undefined on the orbit axis; hold the previous phase there. The
  // guard is on the HORIZONTAL distance, since that is what atan2 consumes --
  // an agent directly above the centre has a large 3D radius but no azimuth.
  if (horiz > 0.01f) {
    ctx.theta = swarmCurveTheta(curve, p);
    ctx.thetaDeg = ctx.theta * 180.0f / (float)M_PI;

    // Height error against the curve's OWN height at this phase, which varies
    // with the elevation profile. A constant reference would be wrong.
    float b;
    swarmCurveEvalB(curve, ctx.theta, &b, NULL);
    ctx.bDeg = b * 180.0f / (float)M_PI;
    ctx.zErr = p[2] - (curve->cz - ctx.r * sinf(b));

    ctx.valid = 1;
  } else {
    ctx.valid = 0;
  }
}

bool swarmPhaseGet(float* theta, float* r) {
  if (!ctx.valid) {
    return false;
  }
  if (theta) { *theta = ctx.theta; }
  if (r)     { *r = ctx.r; }
  return true;
}

/**
 * Encirclement state derived from the state estimate, for cross-checking the
 * range-driven filter.
 */
LOG_GROUP_START(swarmPhase)

/**
 * @brief Distance from the encirclement centre [m]. Equals the curve radius
 * when the agent is on the curve.
 */
LOG_ADD(LOG_FLOAT, r, &ctx.r)

/**
 * @brief Phase on the encirclement curve, CCW from the +x axis [rad]
 */
LOG_ADD(LOG_FLOAT, th, &ctx.theta)

/**
 * @brief Phase on the encirclement curve, CCW from the +x axis [deg]
 */
LOG_ADD(LOG_FLOAT, thDeg, &ctx.thetaDeg)

/**
 * @brief Elevation of the curve at the current phase [deg]. Positive is below
 * the encirclement centre.
 */
LOG_ADD(LOG_FLOAT, bDeg, &ctx.bDeg)

/**
 * @brief Height error against the curve at the current phase [m]
 */
LOG_ADD(LOG_FLOAT, zErr, &ctx.zErr)

/**
 * @brief 1 when the phase is defined, 0 when the agent is on the orbit axis
 */
LOG_ADD(LOG_UINT8, valid, &ctx.valid)

LOG_GROUP_STOP(swarmPhase)
