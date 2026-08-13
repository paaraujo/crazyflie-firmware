/**
 * Swarm encirclement -- onboard radius and phase.
 *
 * Computes the ego agent's position on the encirclement curve from the state
 * estimate and publishes it as log variables.
 *
 * Why these are COMPUTED and not ESTIMATED
 * ----------------------------------------
 * Three non-collinear anchors make 3D position directly observable, so radius
 * and phase are a reduced description of the position the EKF already provides:
 *
 *     r     = || p_xy - c_xy ||         distance from the orbit axis
 *     theta = atan2(y - cy, x - cx)     phase, CCW from +x
 *
 * Carrying them as filter states alongside x, y, z would use five numbers for
 * three degrees of freedom. Beyond being redundant, constraining the position
 * estimate onto the reference curve makes tracking error partly unobservable to
 * the controller -- the estimator would report "on trajectory" while the agent
 * is not. So the EKF owns position, and this module derives phase from it.
 *
 * theta = atan2() is exact while the embedding is the flat circle (R_e = I). A
 * distorted embedding requires the fixed-point inversion of theory.tex 4.1,
 * which is not implemented here yet.
 *
 * The orbit centre defaults to the origin of the anchor frame, which is where
 * the corner anchor A0 of the L-template sits. Override with the swarm.cx/cy
 * parameters if the template origin is elsewhere.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "app.h"
#include "FreeRTOS.h"
#include "task.h"

#include "log.h"
#include "param.h"

#define DEBUG_MODULE "SWARMPHASE"
#include "debug.h"

#define UPDATE_PERIOD_MS 20   // 50 Hz, matched to the ranging rate

static struct {
  // Orbit centre in the anchor frame. A0 of the L-template is at the origin,
  // so these default to zero.
  float cx;
  float cy;

  // Nominal orbit height above the anchor plane. Only reported as an error
  // here; keeping the agents clear of the anchor plane matters because the
  // vertical direction is ill-conditioned there.
  float hc;

  // Outputs
  float r;         // distance from the orbit axis [m]
  float theta;     // phase, CCW from +x [rad]
  float thetaDeg;  // same, in degrees, for convenience when reading logs
  float zErr;      // height error relative to hc [m]
  uint8_t valid;   // 1 when the state estimate is usable
} ctx = {
  .cx = 0.0f,
  .cy = 0.0f,
  .hc = 0.5f,
};

void appMain(void) {
  DEBUG_PRINT("Swarm encirclement: onboard phase\n");

  logVarId_t idX = logGetVarId("stateEstimate", "x");
  logVarId_t idY = logGetVarId("stateEstimate", "y");
  logVarId_t idZ = logGetVarId("stateEstimate", "z");

  if (!logVarIdIsValid(idX) || !logVarIdIsValid(idY) || !logVarIdIsValid(idZ)) {
    DEBUG_PRINT("ERROR: stateEstimate x/y/z not found, phase disabled\n");
    ctx.valid = 0;
    while (true) {
      vTaskDelay(M2T(1000));
    }
  }

  TickType_t lastWake = xTaskGetTickCount();

  while (true) {
    vTaskDelayUntil(&lastWake, M2T(UPDATE_PERIOD_MS));

    const float x = logGetFloat(idX);
    const float y = logGetFloat(idY);
    const float z = logGetFloat(idZ);

    const float dx = x - ctx.cx;
    const float dy = y - ctx.cy;

    ctx.r = sqrtf(dx * dx + dy * dy);
    ctx.zErr = z - ctx.hc;

    // atan2f is undefined at the origin; hold the previous phase there rather
    // than emitting a meaningless value. This happens if the estimate collapses
    // to the orbit axis, where the phase genuinely is not defined.
    if (ctx.r > 0.01f) {
      ctx.theta = atan2f(dy, dx);
      ctx.thetaDeg = ctx.theta * 180.0f / (float)M_PI;
      ctx.valid = 1;
    } else {
      ctx.valid = 0;
    }
  }
}

/**
 * Onboard encirclement state, derived from the state estimate.
 */
LOG_GROUP_START(swarm)

/**
 * @brief Distance from the orbit axis [m]
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
 * @brief Height error relative to the nominal orbit height swarm.hc [m]
 */
LOG_ADD(LOG_FLOAT, zErr, &ctx.zErr)

/**
 * @brief 1 when the phase is defined, 0 when the agent is on the orbit axis
 */
LOG_ADD(LOG_UINT8, valid, &ctx.valid)

LOG_GROUP_STOP(swarm)

PARAM_GROUP_START(swarm)

/**
 * @brief X coordinate of the orbit centre in the anchor frame [m]
 */
PARAM_ADD(PARAM_FLOAT, cx, &ctx.cx)

/**
 * @brief Y coordinate of the orbit centre in the anchor frame [m]
 */
PARAM_ADD(PARAM_FLOAT, cy, &ctx.cy)

/**
 * @brief Nominal orbit height above the anchor plane [m]
 */
PARAM_ADD(PARAM_FLOAT, hc, &ctx.hc)

PARAM_GROUP_STOP(swarm)
