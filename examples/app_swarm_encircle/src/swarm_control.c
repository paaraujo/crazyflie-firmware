/**
 * Encirclement controller. See swarm_control.h for why it avoids the free 3D
 * position estimate.
 *
 * Control decomposition, all in the template frame (A0 origin, A0->A1 = +x):
 *
 *   tangential   v_t = r * (w_nom + k_phi * e_phi)      drives the orbit
 *   radial       v_r = -k_r * (r_hat - r_target)        holds the radius
 *   vertical     handled by the stock altitude loop at cz
 *
 * e_phi is the SEPARATION error, not an absolute phase error. Each agent
 * compares its measured separations to its two neighbours against the nominal
 * spacing and corrects the imbalance. No agent needs to know its neighbours'
 * phases, which is what keeps the scheme communication-free.
 */

#include <math.h>
#include <string.h>

#include "FreeRTOS.h"
#include "task.h"

#include "commander.h"
#include "log.h"
#include "param.h"

#include "swarm_control.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static struct {
  uint8_t en;        // 1 = engage
  float rTarget;     // commanded orbit radius [m]
  float wNom;        // commanded orbit rate [rad/s]
  float kPhi;        // separation gain [1/s]
  float kR;          // radius gain [1/s]
  float vMaxT;       // tangential speed limit [m/s]
  float vMaxR;       // radial speed limit [m/s]
  float rampS;       // seconds to ramp w_nom in on engage
  float maxSigTh;    // refuse to engage above this theta sigma [deg]
  float maxSigR;     // refuse to engage above this r sigma [m]
  uint8_t needChords;// 1 = require both separations to be informed
  uint8_t zRef;      // altitude reference: 0 = cz on the EKF, 1 = this filter
} cfg = {
  .en = 0,
  .rTarget = 0.0f,   // 0 = follow swarmCurve.r
  .wNom = 0.0f,
  .kPhi = 0.5f,
  .kR = 0.8f,
  .vMaxT = 0.35f,
  .vMaxR = 0.25f,
  .rampS = 3.0f,
  .maxSigTh = 15.0f,
  .maxSigR = 0.25f,
  .needChords = 1,
  .zRef = 0,
};

static struct {
  float vx, vy;      // commanded velocity, template frame [m/s]
  float vt, vr;      // tangential / radial components [m/s]
  float ePhi;        // separation error [rad]
  float eR;          // radius error [m]
  float rUse;        // radius target actually applied [m]
  float ramp;        // 0..1
  uint8_t engaged;
  uint8_t blocked;   // 1 = armed but refused (see reason)
  uint8_t reason;    // 0 ok, 1 not init, 2 sigma theta, 3 sigma r, 4 no chords
} out;

static bool engaged = false;
static logVarId_t idStateZ = 0;
static float ramp = 0.0f;

static float clampf(const float v, const float lim) {
  if (v > lim) return lim;
  if (v < -lim) return -lim;
  return v;
}

bool swarmControlIsEngaged(void) {
  return engaged;
}

/**
 * Decide whether it is safe to command.
 *
 * Engaging on a bad estimate is the failure mode that matters: the controller
 * would drive confidently to a wrong phase. The covariance is the filter's own
 * statement about how much it should be trusted, so it gates engagement.
 */
static bool healthy(const swarmEkf_t* f) {
  const float R2D = 180.0f / (float)M_PI;

  if (!f->initialised) {
    out.reason = 1;
    return false;
  }
  if (sqrtf(f->P[SWARM_EKF_TH][SWARM_EKF_TH]) * R2D > cfg.maxSigTh) {
    out.reason = 2;
    return false;
  }
  if (sqrtf(f->P[SWARM_EKF_R][SWARM_EKF_R]) > cfg.maxSigR) {
    out.reason = 3;
    return false;
  }
  if (cfg.needChords) {
    // delta_k and delta_j only shrink below their priors once chords arrive.
    // Without that, the separation error is a guess and correcting it would
    // move the vehicle on no evidence.
    const float pD = 0.9f * f->p0Delta;
    if (f->P[SWARM_EKF_DK][SWARM_EKF_DK] > pD ||
        f->P[SWARM_EKF_DJ][SWARM_EKF_DJ] > pD) {
      out.reason = 4;
      return false;
    }
  }
  out.reason = 0;
  return true;
}

/** Hand control back cleanly. */
static void disengage(void) {
  if (engaged) {
    // Tells the high-level planner where the vehicle actually is, so a
    // subsequent ROS land() starts from the current state rather than from
    // wherever the planner last believed it was.
    commanderRelaxPriority();
    engaged = false;
  }
  ramp = 0.0f;
  // Preserve the reason. healthy() sets it, then this memset used to clear it
  // back to 0 ("ok") before the caller could set blocked -- so every refusal
  // reported itself as healthy and the actual cause was unreportable.
  const uint8_t keep = out.reason;
  memset(&out, 0, sizeof(out));
  out.reason = keep;
}

void swarmControlUpdate(const swarmEkf_t* f, const swarmCurve_t* curve, float dt) {
  // Health is evaluated on EVERY cycle, armed or not, so a host can see
  // whether the controller WOULD accept command before it asks for it.
  // Reporting "not armed" in the same field made readiness unobservable until
  // after arming, which is a chicken-and-egg a careful host cannot escape.
  const bool ok = healthy(f);

  if (!cfg.en) {
    disengage();         // reason now says whether it would engage, not that it has
    return;
  }
  if (!ok) {
    // Armed but unsafe: stop commanding rather than command badly. Releasing
    // priority lets whatever the host last asked for (a hover) take over.
    disengage();
    out.blocked = 1;
    return;
  }
  out.blocked = 0;

  const float th = f->x[SWARM_EKF_TH];
  const float r = f->x[SWARM_EKF_R];

  // --- separation error -------------------------------------------------
  // The filter defines  theta_k = theta_i + dNom + delta_k  (leader, ahead)
  //                     theta_j = theta_i - dNom + delta_j  (follower, behind)
  // so the two gaps either side of this agent are
  //     gap_lead = theta_k - theta_i = dNom + delta_k
  //     gap_foll = theta_i - theta_j = dNom - delta_j
  // and their imbalance is
  //     gap_lead - gap_foll = delta_k + delta_j.
  // Positive means the leader is further away than the follower, so this agent
  // sits too close behind its follower and must SPEED UP. Note it is the SUM,
  // not the difference: delta_k = -delta_j leaves the agent centred between
  // neighbours that have both moved outward, which needs no correction, while
  // delta_k = delta_j means the whole formation has slid ahead of this agent
  // and it must catch up.
  const float ePhi = f->x[SWARM_EKF_DK] + f->x[SWARM_EKF_DJ];

  // --- radius error -----------------------------------------------------
  // The curve owns the nominal radius, because the elevation coefficients were
  // fitted about it. rTgt overrides only when set positive, so a single number
  // in swarmCurve drives both the geometry and the control target and the two
  // cannot silently disagree.
  const float rTarget = (cfg.rTarget > 0.0f) ? cfg.rTarget : curve->r;
  const float eR = r - rTarget;

  // --- ramp -------------------------------------------------------------
  // Stepping w_nom from zero would ask for an instantaneous lateral velocity.
  if (cfg.rampS > 0.0f) {
    ramp += dt / cfg.rampS;
    if (ramp > 1.0f) ramp = 1.0f;
  } else {
    ramp = 1.0f;
  }

  // --- control law ------------------------------------------------------
  const float vt = clampf(r * (ramp * cfg.wNom + cfg.kPhi * ePhi), cfg.vMaxT);
  const float vr = clampf(-cfg.kR * eR, cfg.vMaxR);

  // Rotate into the template frame. Radial unit vector is (cos th, sin th);
  // tangential, in the direction of increasing theta, is (-sin th, cos th).
  const float ct = cosf(th), st = sinf(th);
  const float vx = vr * ct - vt * st;
  const float vy = vr * st + vt * ct;

  // --- emit -------------------------------------------------------------
  setpoint_t sp;
  memset(&sp, 0, sizeof(sp));
  sp.timestamp = xTaskGetTickCount();

  sp.mode.x = modeVelocity;
  sp.mode.y = modeVelocity;
  sp.velocity.x = vx;
  sp.velocity.y = vy;
  sp.velocity_body = false;          // template frame, not body frame

  // Altitude stays on the stock absolute loop, which closes on
  // stateEstimate.z. The setpoint must therefore be expressed in THAT
  // estimate's terms:
  //
  //   zRef 0: setpoint = cz. Trust the EKF's altitude. Correct whenever the
  //           EKF altitude is good -- with mocap, or with a height sensor.
  //   zRef 1: setpoint = stateEstimate.z - dz. Use this filter as the
  //           altitude reference: dz is our estimate of (true z - cz), so this
  //           drives the TRUE altitude to cz even if the EKF's z is biased.
  //
  // The earlier form, cz - dz, was wrong: the EKF already sees an altitude
  // error, and subtracting dz corrected it a second time through the filter's
  // lag -- a double-counted loop that invites vertical oscillation.
  sp.mode.z = modeAbs;
  if (cfg.zRef == 1 && logVarIdIsValid(idStateZ)) {
    sp.position.z = logGetFloat(idStateZ) - f->x[SWARM_EKF_DZ];
  } else {
    sp.position.z = curve->cz;
  }

  // Yaw is not part of the task, but leaving it modeDisable would hand the
  // rate loop an uncommanded axis. Hold whatever heading the vehicle has.
  sp.mode.yaw = modeVelocity;
  sp.attitudeRate.yaw = 0.0f;

  // CRTP priority outranks the high-level planner, so this call also stops any
  // takeoff/goTo still running -- which is what makes the ROS handoff work.
  commanderSetSetpoint(&sp, COMMANDER_PRIORITY_CRTP);
  engaged = true;

  out.vx = vx; out.vy = vy; out.vt = vt; out.vr = vr;
  out.ePhi = ePhi; out.eR = eR; out.ramp = ramp; out.engaged = 1;
  out.rUse = rTarget;
}

void swarmControlInit(void) {
  idStateZ = logGetVarId("stateEstimate", "z");
  memset(&out, 0, sizeof(out));
  engaged = false;
  ramp = 0.0f;
}

/**
 * Commanded velocities and the errors driving them.
 */
LOG_GROUP_START(swarmCtrl)

/** @brief Commanded velocity, template frame x [m/s] */
LOG_ADD(LOG_FLOAT, vx, &out.vx)

/** @brief Commanded velocity, template frame y [m/s] */
LOG_ADD(LOG_FLOAT, vy, &out.vy)

/** @brief Tangential component, drives the orbit [m/s] */
LOG_ADD(LOG_FLOAT, vt, &out.vt)

/** @brief Radial component, holds the radius [m/s] */
LOG_ADD(LOG_FLOAT, vr, &out.vr)

/** @brief Separation imbalance, delta_k + delta_j [rad]. Positive = speed up. */
LOG_ADD(LOG_FLOAT, ePhi, &out.ePhi)

/** @brief Radius error, r_hat - target [m] */
LOG_ADD(LOG_FLOAT, eR, &out.eR)

/** @brief Radius target actually in use [m]: swarmCurve.r unless rTgt overrides */
LOG_ADD(LOG_FLOAT, rUse, &out.rUse)

/** @brief Rate ramp, 0 at engage rising to 1 over swarmCtrl.rampS */
LOG_ADD(LOG_FLOAT, ramp, &out.ramp)

/** @brief 1 while commanding setpoints */
LOG_ADD(LOG_UINT8, eng, &out.engaged)

/** @brief 1 when armed but refusing to command; see swarmCtrl.why */
LOG_ADD(LOG_UINT8, blk, &out.blocked)

/**
 * @brief Why the controller is refusing
 *
 * 0 = would accept command, 1 = filter not initialised,
 * 2 = theta sigma above maxSigTh, 3 = r sigma above maxSigR,
 * 4 = chords have not informed the separations.
 *
 * This reports READINESS, not engagement -- it reads 0 whenever the filter is
 * good enough, whether or not swarmCtrl.en is set. Pair it with swarmCtrl.eng
 * to tell the four cases apart:
 *
 *   why 0, eng 0 : ready, waiting to be armed
 *   why 0, eng 1 : commanding
 *   why >0, eng 0, blk 0 : not ready, not armed
 *   why >0, eng 0, blk 1 : armed but refused -- why says what is missing
 */
LOG_ADD(LOG_UINT8, why, &out.reason)

LOG_GROUP_STOP(swarmCtrl)

/**
 * Encirclement controller configuration.
 */
PARAM_GROUP_START(swarmCtrl)

/**
 * @brief Engage the controller (0/1)
 *
 * Set to 1 after takeoff to begin encircling; clear to hand control back to
 * the high-level commander so a ROS land() works. Clearing it also calls
 * commanderRelaxPriority(), so the planner resumes from the actual state.
 */
PARAM_ADD(PARAM_UINT8, en, &cfg.en)

/**
 * @brief Commanded orbit radius override [m]
 *
 * 0 (the default) means follow swarmCurve.r, which is where the fitting tool
 * puts the radius and where the elevation coefficients were fitted. Set this
 * positive only to command a radius that deliberately differs from the curve,
 * accepting that b(theta) is then evaluated off its fitted surface.
 *
 * Keep the radius comparable to the anchor constellation span either way:
 * position accuracy degrades as distance/baseline, so a larger orbit costs
 * phase accuracy directly.
 */
PARAM_ADD(PARAM_FLOAT, rTgt, &cfg.rTarget)

/**
 * @brief Commanded orbit rate [rad/s]
 *
 * Positive is counter-clockwise, matching increasing theta. Ramped in over
 * rampS on engage. 0.2 rad/s is about 31 s per revolution.
 */
PARAM_ADD(PARAM_FLOAT, wNom, &cfg.wNom)

/**
 * @brief Separation gain [1/s]
 *
 * How hard to correct an uneven formation. The correction enters as a
 * tangential rate, so the effective time constant is roughly 1/kPhi.
 */
PARAM_ADD(PARAM_FLOAT, kPhi, &cfg.kPhi)

/**
 * @brief Radius gain [1/s]
 */
PARAM_ADD(PARAM_FLOAT, kR, &cfg.kR)

/**
 * @brief Tangential speed limit [m/s]
 */
PARAM_ADD(PARAM_FLOAT, vMaxT, &cfg.vMaxT)

/**
 * @brief Radial speed limit [m/s]
 */
PARAM_ADD(PARAM_FLOAT, vMaxR, &cfg.vMaxR)

/**
 * @brief Seconds to ramp wNom in on engage. 0 disables ramping.
 */
PARAM_ADD(PARAM_FLOAT, rampS, &cfg.rampS)

/**
 * @brief Refuse to engage while theta's one-sigma exceeds this [deg]
 */
PARAM_ADD(PARAM_FLOAT, sigTh, &cfg.maxSigTh)

/**
 * @brief Refuse to engage while r's one-sigma exceeds this [m]
 */
PARAM_ADD(PARAM_FLOAT, sigR, &cfg.maxSigR)

/**
 * @brief Require chords before engaging (0/1)
 *
 * With 1, the controller waits until both separations have been informed by
 * chord measurements. Set to 0 for single-drone testing, where there are no
 * neighbours and the separation terms are meaningless -- combine with
 * kPhi = 0 so only the orbit and radius are commanded.
 */
PARAM_ADD(PARAM_UINT8, needC, &cfg.needChords)

/**
 * @brief Altitude reference (0/1)
 *
 * 0 (default): hold swarmCurve.cz on the EKF's altitude. Use whenever the EKF
 * altitude is trustworthy -- mocap-stabilised flight, or a height sensor.
 * 1: setpoint = stateEstimate.z - dz, i.e. use this filter's vertical offset
 * as the reference, for UWB-only flight where the EKF's z may be biased.
 */
PARAM_ADD(PARAM_UINT8, zRef, &cfg.zRef)

PARAM_GROUP_STOP(swarmCtrl)
