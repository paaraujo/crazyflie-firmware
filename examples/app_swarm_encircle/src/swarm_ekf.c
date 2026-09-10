/**
 * Encirclement EKF. See swarm_ekf.h for the derivation and coordinate choice.
 */

#include <math.h>
#include <string.h>

#include "swarm_ekf.h"

#define DIM SWARM_EKF_DIM

float swarmEkfWrap(float a) {
  while (a >= (float)M_PI)  { a -= 2.0f * (float)M_PI; }
  while (a < -(float)M_PI)  { a += 2.0f * (float)M_PI; }
  return a;
}

void swarmEkfDefaults(swarmEkf_t* f) {
  // Process noise, per second. qDelta << qTheta is the whole point of these
  // coordinates: disturbances displace the formation as a whole.
  f->qTheta = 1e-3f;
  f->qDelta = 1e-5f;
  f->qOmega = 1e-4f;
  f->qR     = 1e-5f;

  // Initial covariance (variances). theta and r are seeded from an independent
  // source and trusted moderately; the formation errors are unknown until the
  // first chords arrive, so they start wide.
  f->p0Theta = 0.25f;   // (0.5 rad)^2  ~ 29 deg one sigma
  f->p0Delta = 0.50f;   // (0.71 rad)^2 ~ 40 deg one sigma
  f->p0Omega = 0.25f;   // (0.5 rad/s)^2
  f->p0R     = 0.09f;   // (0.3 m)^2
}

/** Accept a configured variance only if it is positive, finite and sane. */
static float p0OrDefault(const float v, const float fallback) {
  return (v > 0.0f && v < 1.0e6f) ? v : fallback;
}

void swarmEkfInit(swarmEkf_t* f, const float theta0, const float r0, const float dNom) {
  memset(f->x, 0, sizeof(f->x));
  memset(f->P, 0, sizeof(f->P));

  f->x[SWARM_EKF_TH] = swarmEkfWrap(theta0);
  f->x[SWARM_EKF_R] = r0;
  // delta_k, delta_j and omega_e start at zero. Zero formation error is not
  // merely a convenient guess: it places the neighbours on the CCW-ordered
  // branch of the chord ambiguity, which prediction continuity then maintains.

  // Initial uncertainty. theta is seeded from an independent source and is
  // trusted moderately; the formation errors are not known at all.
  // A zero variance claims perfect knowledge of the seed: the Kalman gain for
  // that state becomes zero and no measurement can ever move it again. The
  // guard makes that unreachable by configuration.
  // The guard catches zero, negative, NaN and absurd values. It cannot catch
  // garbage that merely looks plausible, which is why swarmEkfDefaults() exists.
  const float pTh = p0OrDefault(f->p0Theta, 0.25f);   // (0.5 rad)^2
  const float pD  = p0OrDefault(f->p0Delta, 0.50f);
  const float pW  = p0OrDefault(f->p0Omega, 0.25f);
  const float pR  = p0OrDefault(f->p0R,     0.09f);   // (0.3 m)^2

  f->P[SWARM_EKF_TH][SWARM_EKF_TH] = pTh;
  f->P[SWARM_EKF_DK][SWARM_EKF_DK] = pD;
  f->P[SWARM_EKF_DJ][SWARM_EKF_DJ] = pD;
  f->P[SWARM_EKF_WE][SWARM_EKF_WE] = pW;
  f->P[SWARM_EKF_R][SWARM_EKF_R]   = pR;

  f->dNom = dNom;
  f->lastNis = 0.0f;
  f->nAccept = 0;
  f->nReject = 0;
  f->initialised = true;
}

void swarmEkfPredict(swarmEkf_t* f, const float wz, const float dt) {
  if (!f->initialised || dt <= 0.0f) {
    return;
  }

  // x' = F x + u.  F is the identity except for the omega_e -> theta coupling,
  // so the propagation is exact rather than linearised.
  f->x[SWARM_EKF_TH] = swarmEkfWrap(
      f->x[SWARM_EKF_TH] + (wz + f->x[SWARM_EKF_WE]) * dt);
  // delta_k, delta_j, omega_e and r are all constant under the model; the
  // differences being constant is what carries the neighbours along with the
  // ego agent instead of at an assumed nominal rate.

  // P = F P F' + Q, exploiting F = I + dt * e_TH e_WE': only row TH and
  // column TH change. Applying the row pass and then the column pass IN THAT
  // ORDER reproduces F P F' exactly -- the column pass reads the already
  // updated P[TH][WE], which is what supplies the dt^2 P_WE,WE term at (TH,TH).
  // Adding that term separately would double-count it.
  const int T = SWARM_EKF_TH;
  const int W = SWARM_EKF_WE;

  for (int j = 0; j < DIM; j++) {
    f->P[T][j] += dt * f->P[W][j];
  }
  for (int i = 0; i < DIM; i++) {
    f->P[i][T] += dt * f->P[i][W];
  }

  f->P[SWARM_EKF_TH][SWARM_EKF_TH] += f->qTheta * dt;
  f->P[SWARM_EKF_DK][SWARM_EKF_DK] += f->qDelta * dt;
  f->P[SWARM_EKF_DJ][SWARM_EKF_DJ] += f->qDelta * dt;
  f->P[SWARM_EKF_WE][SWARM_EKF_WE] += f->qOmega * dt;
  f->P[SWARM_EKF_R][SWARM_EKF_R]   += f->qR * dt;
}

/**
 * Sequential scalar update. Never inverts a matrix: the innovation covariance
 * is a scalar, which is what makes asynchronous ranges cheap to fuse.
 */
static bool scalarUpdate(swarmEkf_t* f, const float H[DIM],
                         const float innovation, const float sigma,
                         const float nisGate) {
  float PH[DIM];
  for (int i = 0; i < DIM; i++) {
    float s = 0.0f;
    for (int j = 0; j < DIM; j++) {
      s += f->P[i][j] * H[j];
    }
    PH[i] = s;
  }

  float S = sigma * sigma;
  for (int i = 0; i < DIM; i++) {
    S += H[i] * PH[i];
  }
  if (!(S > 1e-12f)) {
    f->nReject++;
    return false;
  }

  // Innovation gate. A measurement this far from the prediction is more likely
  // a multipath return than evidence about the state.
  const float nis = innovation * innovation / S;
  if (nisGate > 0.0f && nis > nisGate) {
    f->nReject++;
    return false;
  }
  f->lastNis = nis;

  for (int i = 0; i < DIM; i++) {
    const float k = PH[i] / S;
    f->x[i] += k * innovation;
    // P -= K (H P);  H P is (P H')' = PH' because P is symmetric.
    for (int j = 0; j < DIM; j++) {
      f->P[i][j] -= k * PH[j];
    }
  }

  // Re-symmetrise: rounding makes P drift out of symmetry, and an asymmetric
  // covariance eventually produces a negative innovation variance.
  for (int i = 0; i < DIM; i++) {
    for (int j = i + 1; j < DIM; j++) {
      const float m = 0.5f * (f->P[i][j] + f->P[j][i]);
      f->P[i][j] = m;
      f->P[j][i] = m;
    }
  }

  f->x[SWARM_EKF_TH] = swarmEkfWrap(f->x[SWARM_EKF_TH]);
  f->x[SWARM_EKF_DK] = swarmEkfWrap(f->x[SWARM_EKF_DK]);
  f->x[SWARM_EKF_DJ] = swarmEkfWrap(f->x[SWARM_EKF_DJ]);
  if (f->x[SWARM_EKF_R] < 0.05f) {
    f->x[SWARM_EKF_R] = 0.05f;   // a non-positive radius is unphysical
  }

  f->nAccept++;
  return true;
}

bool swarmEkfUpdateAnchor(swarmEkf_t* f, const swarmCurve_t* curve,
                          const float anchor[3], const float dist,
                          const float sigma, const float nisGate) {
  if (!f->initialised) {
    return false;
  }

  const float th = f->x[SWARM_EKF_TH];
  const float r = f->x[SWARM_EKF_R];

  float q[3], dq[3], u[3];
  swarmCurveQ(curve, th, r, q);
  swarmCurveDq(curve, th, r, dq);
  swarmCurveU(curve, th, u);

  const float v[3] = {q[0] - anchor[0], q[1] - anchor[1], q[2] - anchor[2]};
  const float h = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  if (h < 1e-6f) {
    return false;
  }
  const float e[3] = {v[0]/h, v[1]/h, v[2]/h};

  float H[DIM] = {0};
  H[SWARM_EKF_TH] = e[0]*dq[0] + e[1]*dq[1] + e[2]*dq[2];
  H[SWARM_EKF_R]  = e[0]*u[0]  + e[1]*u[1]  + e[2]*u[2];
  // The corner anchor is NOT special-cased: with the encirclement centre above
  // the anchor plane its phase sensitivity is h*q_z'/d_0, which is non-zero.

  return scalarUpdate(f, H, dist - h, sigma, nisGate);
}

bool swarmEkfUpdateChord(swarmEkf_t* f, const swarmCurve_t* curve,
                         const bool isLeader, const float dist,
                         const float sigma, const float nisGate) {
  if (!f->initialised) {
    return false;
  }

  const float th = f->x[SWARM_EKF_TH];
  const float r = f->x[SWARM_EKF_R];

  const int idx = isLeader ? SWARM_EKF_DK : SWARM_EKF_DJ;
  const float sign = isLeader ? +1.0f : -1.0f;
  const float thN = th + sign * f->dNom + f->x[idx];

  float qi[3], qn[3], dqi[3], dqn[3];
  swarmCurveQ(curve, th, r, qi);
  swarmCurveQ(curve, thN, r, qn);
  swarmCurveDq(curve, th, r, dqi);
  swarmCurveDq(curve, thN, r, dqn);

  const float v[3] = {qi[0] - qn[0], qi[1] - qn[1], qi[2] - qn[2]};
  const float h = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  if (h < 1e-6f) {
    return false;
  }
  const float e[3] = {v[0]/h, v[1]/h, v[2]/h};

  float H[DIM] = {0};
  // Advancing theta_i drags the neighbour with it, so the common-mode entry is
  // a DIFFERENCE of tangents -- and vanishes identically on a flat circle.
  H[SWARM_EKF_TH] = e[0]*(dqi[0]-dqn[0]) + e[1]*(dqi[1]-dqn[1]) + e[2]*(dqi[2]-dqn[2]);
  H[idx] = -(e[0]*dqn[0] + e[1]*dqn[1] + e[2]*dqn[2]);
  // Both endpoints share r, so q_i - q_n = r (u_i - u_n) and dh/dr = h/r.
  H[SWARM_EKF_R] = (r > 1e-6f) ? (h / r) : 0.0f;

  return scalarUpdate(f, H, dist - h, sigma, nisGate);
}

void swarmEkfSeparations(const swarmEkf_t* f, float* phiK, float* phiJ) {
  if (phiK) {
    *phiK = -(f->dNom + f->x[SWARM_EKF_DK]);
  }
  if (phiJ) {
    *phiJ = +(f->dNom - f->x[SWARM_EKF_DJ]);
  }
}
