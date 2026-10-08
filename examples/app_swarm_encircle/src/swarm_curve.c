/**
 * Encirclement curve evaluation. See swarm_curve.h for the derivation.
 */

#include <math.h>
#include <stddef.h>

#include "swarm_curve.h"

#ifdef SWARM_CURVE_HOST_TEST
// Host build for the unit test: no param system.
static swarmCurve_t curve;
void swarmCurveSetForTest(const swarmCurve_t* c) { curve = *c; }
#else
#include "log.h"
#include "param.h"

// Defaults describe the flat circle at 1 m: K = 0 and b0 = 0 give b(theta) = 0,
// so the vehicle flies a level ring until an elevation profile is loaded.
static swarmCurve_t curve = {
  .cx = 0.0f,
  .cy = 0.0f,
  .cz = 1.0f,
  .r = 0.6f,
  .K = 0,
  .b0 = 0.0f,
};
#endif

const swarmCurve_t* swarmCurveGet(void) {
  return &curve;
}

void swarmCurveEvalB(const swarmCurve_t* c, const float theta, float* b, float* db) {
  float bb = c->b0;
  float dd = 0.0f;

  uint8_t K = c->K;
  if (K > SWARM_CURVE_KMAX) {
    K = SWARM_CURVE_KMAX;
  }

  if (K > 0) {
    const float c1 = cosf(theta);
    const float s1 = sinf(theta);
    float ck = c1;   // cos(k theta), k = 1
    float sk = s1;   // sin(k theta), k = 1

    for (uint8_t k = 1; k <= K; k++) {
      const float a = c->alpha[k - 1];
      const float be = c->beta[k - 1];

      bb += a * ck + be * sk;
      dd += (float)k * (be * ck - a * sk);

      // Advance to k+1 by angle addition, so the loop costs no further
      // transcendental calls.
      const float cn = ck * c1 - sk * s1;
      const float sn = sk * c1 + ck * s1;
      ck = cn;
      sk = sn;
    }
  }

  if (b != NULL) {
    *b = bb;
  }
  if (db != NULL) {
    *db = dd;
  }
}

void swarmCurveQ(const swarmCurve_t* c, const float theta, const float r, float q[3]) {
  float b;
  swarmCurveEvalB(c, theta, &b, NULL);

  const float cb = cosf(b);
  const float sb = sinf(b);
  const float ct = cosf(theta);
  const float st = sinf(theta);

  q[0] = c->cx + r * cb * ct;
  q[1] = c->cy + r * cb * st;
  q[2] = c->cz - r * sb;
}

void swarmCurveDq(const swarmCurve_t* c, const float theta, const float r, float dq[3]) {
  float b, db;
  swarmCurveEvalB(c, theta, &b, &db);

  const float cb = cosf(b);
  const float sb = sinf(b);
  const float ct = cosf(theta);
  const float st = sinf(theta);

  // Differentiating the normal form directly. No right Jacobian of SO(3) and
  // no finite differences are needed.
  dq[0] = -r * (db * sb * ct + cb * st);
  dq[1] =  r * (cb * ct - db * sb * st);
  dq[2] = -r * cb * db;
}

void swarmCurveU(const swarmCurve_t* c, const float theta, float u[3]) {
  float b;
  swarmCurveEvalB(c, theta, &b, NULL);

  const float cb = cosf(b);

  u[0] = cb * cosf(theta);
  u[1] = cb * sinf(theta);
  u[2] = -sinf(b);
}

float swarmCurveTheta(const swarmCurve_t* c, const float q[3]) {
  return atan2f(q[1] - c->cy, q[0] - c->cx);
}

float swarmCurveRadius(const swarmCurve_t* c, const float q[3]) {
  const float dx = q[0] - c->cx;
  const float dy = q[1] - c->cy;
  const float dz = q[2] - c->cz;
  return sqrtf(dx * dx + dy * dy + dz * dz);
}

#ifndef SWARM_CURVE_HOST_TEST

/**
 * Encirclement curve definition. The elevation coefficients come from
 * tools/swarm/fit_elevation.py; K = 0 with b0 = 0 is the flat circle.
 */
PARAM_GROUP_START(swarmCurve)

/** @brief X coordinate of the encirclement centre in the anchor frame [m] */
PARAM_ADD(PARAM_FLOAT, cx, &curve.cx)

/** @brief Y coordinate of the encirclement centre in the anchor frame [m] */
PARAM_ADD(PARAM_FLOAT, cy, &curve.cy)

/** @brief Altitude of the encirclement centre above the anchor plane [m].
 *  This is the h of the theory; it must be strictly positive. */
PARAM_ADD(PARAM_FLOAT, cz, &curve.cz)

/**
 * @brief Nominal orbit radius [m]
 *
 * The radius the controller drives towards when swarmCtrl.rTgt is left at 0.
 *
 * b(theta) is an angle, so the harmonic coefficients are scale-free: changing
 * r rescales the curve uniformly rather than distorting the elevation profile.
 * Refit only if you want different elevation ANGLES; the same coefficients
 * remain valid at any radius, describing a similar curve of a different size.
 */
PARAM_ADD(PARAM_FLOAT, r, &curve.r)

/** @brief Harmonic order of the elevation profile in use, 0..4. 0 = flat. */
PARAM_ADD(PARAM_UINT8, K, &curve.K)

/** @brief Elevation profile constant term beta_0 [rad] */
PARAM_ADD(PARAM_FLOAT, b0, &curve.b0)

/** @brief Elevation profile alpha_1 (cos theta) [rad] */
PARAM_ADD(PARAM_FLOAT, a1, &curve.alpha[0])
/** @brief Elevation profile beta_1 (sin theta) [rad] */
PARAM_ADD(PARAM_FLOAT, b1, &curve.beta[0])

/** @brief Elevation profile alpha_2 (cos 2theta) [rad] */
PARAM_ADD(PARAM_FLOAT, a2, &curve.alpha[1])
/** @brief Elevation profile beta_2 (sin 2theta) [rad] */
PARAM_ADD(PARAM_FLOAT, b2, &curve.beta[1])

/** @brief Elevation profile alpha_3 (cos 3theta) [rad] */
PARAM_ADD(PARAM_FLOAT, a3, &curve.alpha[2])
/** @brief Elevation profile beta_3 (sin 3theta) [rad] */
PARAM_ADD(PARAM_FLOAT, b3, &curve.beta[2])

/** @brief Elevation profile alpha_4 (cos 4theta) [rad] */
PARAM_ADD(PARAM_FLOAT, a4, &curve.alpha[3])
/** @brief Elevation profile beta_4 (sin 4theta) [rad] */
PARAM_ADD(PARAM_FLOAT, b4, &curve.beta[3])

PARAM_GROUP_STOP(swarmCurve)

/**
 * =====================================================================
 * Curve configuration, mirrored as logs.
 * =====================================================================
 *
 * These duplicate the parameters above. Parameters are write-mostly from the
 * host's point of view: reading one back needs the server's
 * firmware_params.query_all_values_on_connect, which floods the CRTP receive
 * queue at connect. Logs are always readable, so mirroring the curve here is
 * the cheap way to answer "did my configuration actually land", which is
 * otherwise invisible until the vehicle flies somewhere unexpected.
 *
 * The harmonic coefficients a1..b4 are deliberately NOT mirrored: eight more
 * floats would not fit a log block, and for a flat circle (K = 0) they are
 * unused. Verify them from the parameter TOC if an elevation profile is loaded.
 */
LOG_GROUP_START(swarmCurve)

/** @brief Configured encirclement centre X [m] */
LOG_ADD(LOG_FLOAT, cx, &curve.cx)

/** @brief Configured encirclement centre Y [m] */
LOG_ADD(LOG_FLOAT, cy, &curve.cy)

/** @brief Configured centre altitude above the anchor plane [m] */
LOG_ADD(LOG_FLOAT, cz, &curve.cz)

/** @brief Configured nominal orbit radius [m] */
LOG_ADD(LOG_FLOAT, r, &curve.r)

/** @brief Configured elevation constant term b0 [rad] */
LOG_ADD(LOG_FLOAT, b0, &curve.b0)

/** @brief Configured harmonic order in use, 0..4 */
LOG_ADD(LOG_UINT8, K, &curve.K)

LOG_GROUP_STOP(swarmCurve)

#endif // SWARM_CURVE_HOST_TEST
