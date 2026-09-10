/**
 * Encirclement EKF in common and differential coordinates.
 *
 * The filter constrains the agent to the curve from the outset, so each agent
 * contributes ONE degree of freedom rather than three. That is why it can work
 * where trilateration from three marginal ranges does not: the same
 * measurements, a far better conditioned problem.
 *
 * State
 * -----
 *     x = [ theta_i,  delta_k,  delta_j,  omega_e,  r ]
 *           common     differential        rate     radius
 *
 * with the neighbours defined as deviations from nominal spacing,
 *
 *     theta_k = theta_i + D_nom + delta_k     (leader,   ahead, CCW)
 *     theta_j = theta_i - D_nom + delta_j     (follower, behind)
 *
 * so delta_k, delta_j are formation ERRORS, zero at perfect spacing, and the
 * separations the controller consumes follow directly.
 *
 * Why these coordinates: the transformation is an exact linear change of basis
 * and adds no information. What it buys is expressibility of the process noise.
 * Disturbances -- wind, a mis-scaled rate command, a slow control loop --
 * displace the formation AS A WHOLE, so the common mode should carry large
 * process noise and the differential modes small. In the naive
 * [theta_i, theta_k, theta_j] coordinates that belief requires a dense, almost
 * perfectly correlated Q that nobody reaches by tuning diagonal entries; here it
 * is one large number and two small ones.
 *
 * Two further properties of the prediction matter. The neighbours are not
 * propagated at an unverifiable nominal rate -- holding the differences constant
 * carries them along with whatever the ego agent is actually doing, so
 * common-mode rate error cancels and never enters the separation estimate. And
 * omega_e is learned rather than assumed, absorbing a persistent
 * commanded-versus-achieved discrepancy that would otherwise appear as a lag
 * bias no amount of process noise can shift.
 *
 * Measurement structure (the reason both sensor types are needed):
 *
 *     anchor ranges  ->  common mode, strongly (all three, including the corner)
 *     chords         ->  differential modes, strongly; common mode only weakly,
 *                        and not at all on a flat circle
 *
 * Ranges are applied as independent SCALAR updates as they arrive. No
 * simultaneity is required and no matrix is ever inverted.
 */

#ifndef __SWARM_EKF_H__
#define __SWARM_EKF_H__

#include <stdbool.h>
#include <stdint.h>

#include "swarm_curve.h"

#define SWARM_EKF_DIM 5

// State indices
#define SWARM_EKF_TH 0   // theta_i, common phase
#define SWARM_EKF_DK 1   // delta_k, leader formation error
#define SWARM_EKF_DJ 2   // delta_j, follower formation error
#define SWARM_EKF_WE 3   // omega_e, angular rate error
#define SWARM_EKF_R  4   // r, encirclement radius

typedef struct {
  float x[SWARM_EKF_DIM];
  float P[SWARM_EKF_DIM][SWARM_EKF_DIM];

  // Process noise, per second. qDelta << qTheta encodes common-mode dominance.
  float qTheta;
  float qDelta;
  float qOmega;
  float qR;

  // Initial covariance, applied by swarmEkfInit. Diagonal only: there is no
  // reason to assume the states are correlated before any measurement.
  //
  // These are VARIANCES -- the square of the 1-sigma uncertainty you believe
  // the SEED has. That is a different question from the process noise: P0 says
  // how wrong the starting guess may be, Q says how fast the truth drifts
  // afterwards.
  //
  // A non-positive entry is replaced by a safe default at init. Zero would
  // claim perfect knowledge of the seed, and the filter would then reject every
  // measurement disagreeing with it -- silently, and permanently.
  float p0Theta;
  float p0Delta;
  float p0Omega;
  float p0R;

  float dNom;        // nominal spacing 2*pi/n [rad]
  bool initialised;

  // Diagnostics
  float lastNis;     // normalised innovation squared of the last accepted update
  uint32_t nAccept;
  uint32_t nReject;
} swarmEkf_t;

/**
 * Populate the process noise and initial covariance with sane defaults.
 *
 * MUST be called before swarmEkfInit unless every q* and p0* field is set
 * explicitly: swarmEkf_t carries its own configuration, so a struct on the
 * stack starts as uninitialised memory. Garbage that happens to be small and
 * positive passes the guard in swarmEkfInit and produces a filter that is
 * confident in a seed it has no right to trust -- which looks like convergence
 * to the wrong answer rather than like an error.
 */
void swarmEkfDefaults(swarmEkf_t* f);

/** Seed the filter. delta_k/delta_j start at zero, which is what selects the
 *  correct branch of the chord's ahead/behind ambiguity. */
void swarmEkfInit(swarmEkf_t* f, float theta0, float r0, float dNom);

/** Propagate. wz is the commanded angular rate [rad/s]. */
void swarmEkfPredict(swarmEkf_t* f, float wz, float dt);

/** Range to a fixed anchor at @p anchor. Returns false if gated out. */
bool swarmEkfUpdateAnchor(swarmEkf_t* f, const swarmCurve_t* curve,
                          const float anchor[3], float dist, float sigma,
                          float nisGate);

/** Chord to the leader (@p isLeader true) or follower. */
bool swarmEkfUpdateChord(swarmEkf_t* f, const swarmCurve_t* curve,
                         bool isLeader, float dist, float sigma,
                         float nisGate);

/** Separations the controller consumes: phi_ki and phi_ji [rad]. */
void swarmEkfSeparations(const swarmEkf_t* f, float* phiK, float* phiJ);

/** Wrap an angle to [-pi, pi). */
float swarmEkfWrap(float a);

#endif // __SWARM_EKF_H__
