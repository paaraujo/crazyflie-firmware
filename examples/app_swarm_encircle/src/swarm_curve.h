/**
 * Encirclement curve: the embedded trajectory, evaluated forward only.
 *
 * The embedding generator decomposes in the moving frame of the virtual agent
 * as omega = a*p_hat + b*t_hat. The radial component a is a no-op -- it rotates
 * the point about the very radius it lies on -- so a purely tangential
 * generator loses nothing, and the construction collapses to its normal form:
 *
 *     q(theta) = c + r * [ cos b cos theta,  cos b sin theta,  -sin b ]
 *
 * which is spherical coordinates on the sphere of radius r about the centre c,
 * with azimuth theta and elevation -b(theta). Two consequences matter here:
 *
 *   1. No matrix exponential, no rotation matrix, no Rodrigues. The whole curve
 *      is a handful of cos/sin calls, which is why this runs comfortably on the
 *      vehicle.
 *
 *   2. atan2(q_y - c_y, q_x - c_x) = theta EXACTLY, for any elevation profile
 *      however severe. The embedding cannot move a point in azimuth, only in
 *      elevation. There is no fixed-point inversion to perform.
 *
 * The elevation profile is a truncated Fourier series,
 *
 *     b(theta) = b0 + sum_k [ alpha_k cos(k theta) + beta_k sin(k theta) ]
 *
 * whose coefficients are fitted OFFLINE from waypoints by
 * tools/swarm/fit_elevation.py and delivered as parameters. Nothing here
 * solves anything; it only evaluates.
 *
 * Sign convention: b > 0 is BELOW the centre, since q_z = c_z - r sin b.
 */

#ifndef __SWARM_CURVE_H__
#define __SWARM_CURVE_H__

#include <stdint.h>

/** Maximum harmonic order. Sized for the trajectories the fitter produces. */
#define SWARM_CURVE_KMAX 4

typedef struct {
  // Encirclement centre c. Note this is NOT the corner anchor: it sits at
  // altitude cz above the anchor plane, and conflating the two collapses the
  // geometry into its degenerate configuration.
  float cx;
  float cy;
  float cz;   // the altitude h of the theory

  uint8_t K;                        // harmonic order in use, 0..SWARM_CURVE_KMAX
  float b0;                         // beta_0
  float alpha[SWARM_CURVE_KMAX];    // alpha_1 .. alpha_K
  float beta[SWARM_CURVE_KMAX];     // beta_1  .. beta_K
} swarmCurve_t;

/** The live configuration, written by the swarm.* parameters. */
const swarmCurve_t* swarmCurveGet(void);

/**
 * Elevation and its derivative in one pass.
 *
 * Both share the same cos(k theta) / sin(k theta) sequence, which is generated
 * by angle addition from a single cos/sin pair -- so the cost is two
 * transcendental calls regardless of harmonic order.
 *
 * Either output pointer may be NULL.
 */
void swarmCurveEvalB(const swarmCurve_t* c, float theta, float* b, float* db);

/** Position on the curve, q(theta). */
void swarmCurveQ(const swarmCurve_t* c, float theta, float r, float q[3]);

/** Tangent dq/dtheta, for the measurement Jacobians. */
void swarmCurveDq(const swarmCurve_t* c, float theta, float r, float dq[3]);

/** dq/dr: the unit vector from the centre toward the curve point. */
void swarmCurveU(const swarmCurve_t* c, float theta, float u[3]);

/**
 * Phase of a Cartesian position, exact for this family of curves.
 * Equivalent to atan2(q_y - c_y, q_x - c_x); provided for symmetry with the
 * forward evaluation and to keep the centre offset in one place.
 */
float swarmCurveTheta(const swarmCurve_t* c, const float q[3]);

/** Distance from the encirclement centre. Equals r exactly when on the curve. */
float swarmCurveRadius(const swarmCurve_t* c, const float q[3]);

#endif // __SWARM_CURVE_H__
