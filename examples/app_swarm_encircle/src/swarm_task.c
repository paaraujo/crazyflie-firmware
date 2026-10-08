/**
 * Swarm encirclement app task.
 *
 * Runs two independent estimates of the same quantity, on purpose:
 *
 *   swarmPhase.thDeg   phase derived from the EKF POSITION (swarm_phase.c)
 *   swarmFilter.th     phase from the range-driven filter  (swarm_ekf.c)
 *
 * The filter is the primary source -- it constrains the agent to the curve and
 * fuses the UWB ranges directly, which is far better conditioned than
 * trilaterating three degrees of freedom from three marginal ranges. But that
 * also leaves nothing checking it. The position-derived value comes from a
 * different measurement path (IMU plus the anchor ranges, through the onboard
 * EKF), so a disagreement between the two is the signal that the filter has
 * locked onto the wrong branch of the chord ambiguity or diverged. Log
 * swarmDiagnostics.dTh to watch it.
 *
 * Range consumption
 * -----------------
 * Two-way ranging is asynchronous and the filter applies scalar updates, so no
 * simultaneity is needed. What IS needed is applying each measurement exactly
 * once: re-using one shrinks the covariance without new evidence, and repeated
 * often enough the gain collapses and the filter stops responding to data while
 * still reporting confidence. hybridGetRange() returns the update timestamp for
 * precisely this, and a slot is consumed only when that timestamp advances.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "app.h"
#include "FreeRTOS.h"
#include "task.h"

#include "log.h"
#include "param.h"
#include "locodeck.h"
#include "lpsTdoa3Tag.h"

#include "swarm_curve.h"
#include "swarm_ekf.h"
#include "swarm_phase.h"
#include "swarm_control.h"

#define DEBUG_MODULE "SWARM"
#include "debug.h"

#define UPDATE_PERIOD_MS 20
#define DT ((float)UPDATE_PERIOD_MS / 1000.0f)

// Slot convention, matching tools/swarm/setup_swarm.py.
//
// Slots 0..N_ANCHOR_SLOTS-1 carry anchor ranges, the next two carry the chords
// to the leader and follower. The anchor count is a compile-time constant
// rather than a parameter because the slot layout has to agree with the
// hmLId* configuration pushed from the host; a mismatch shows up as ranges
// attributed to the wrong role, which is silent and hard to see. Keep this in
// step with tools/swarm/setup_swarm.py and the crazyflies.yaml hmLId* block.
#define N_ANCHOR_SLOTS 5
#define SLOT_LEADER   (N_ANCHOR_SLOTS)        // 5
#define SLOT_FOLLOWER (N_ANCHOR_SLOTS + 1)    // 6

_Static_assert(SLOT_FOLLOWER < HYBRID_RANGE_SLOTS,
               "not enough hybrid range slots for 5 anchors plus 2 chords -- "
               "raise HM_LOG_SLOTS in lpsTdoa3Tag.c and HYBRID_RANGE_SLOTS in "
               "its header");

static swarmEkf_t ekf;

static struct {
  uint8_t enable;    // 0 = filter idle
  uint8_t reset;     // write 1 to force re-initialisation
  uint8_t n;         // swarm size, sets the nominal spacing 2*pi/n
  float wz;          // commanded angular rate [rad/s]
  float sigAnchor;   // anchor range noise [m]
  float sigChord;    // chord noise [m]
  float nisGate;     // innovation gate, normalised; 0 disables
} cfg = {
  .enable = 1,
  .reset = 0,
  .n = 3,
  .wz = 0.0f,
  .sigAnchor = 0.03f,
  .sigChord = 0.03f,
  .nisGate = 25.0f,   // 5 sigma
};

static struct {
  float th;           // phase [rad]
  float dk, dj;       // formation errors [rad]
  float we, r;
  float dz;           // vertical offset from the modelled curve [m]
  float phiK, phiJ;   // separations [rad]
  float dTheta;       // filter vs position-derived phase [deg]
  float nis;
  float sTh, sDk, sDj, sWe, sR, sDz;  // 1-sigma, natural units
  float pTh, pDk, pDj, pWe, pR, pDz;  // raw P diagonal, SI (rad^2 .. m^2)
  uint8_t init;
  uint16_t nAcc, nRej;
  uint16_t seq;       // increments once per publish; pairs the state and
                      // covariance blocks, which are sampled independently
} out;

static uint32_t lastSeen[HYBRID_RANGE_SLOTS];

/** Seed from the position-derived phase, which is exact for this curve family. */
static bool tryInit(const swarmCurve_t* curve) {
  float theta, r;
  if (!swarmPhaseGet(&theta, &r)) {
    return false;
  }
  if (r < 0.05f) {
    return false;
  }
  swarmEkfInit(&ekf, theta, r, 2.0f * (float)M_PI / (float)(cfg.n > 0 ? cfg.n : 3));
  for (int s = 0; s < HYBRID_RANGE_SLOTS; s++) {
    lastSeen[s] = 0;
  }
  DEBUG_PRINT("filter init: theta %.1f deg, r %.2f m\n",
              (double)(theta * 180.0f / (float)M_PI), (double)r);
  return true;
}

static void consumeRanges(const swarmCurve_t* curve) {
  for (uint8_t s = 0; s < HYBRID_RANGE_SLOTS; s++) {
    uint8_t id;
    float d;
    uint32_t t;

    if (!hybridGetRange(s, &id, &d, &t)) {
      continue;
    }
    // d == 0 is the firmware's stale marker; a real range is never zero.
    if (d == 0.0f || t == lastSeen[s]) {
      continue;
    }
    lastSeen[s] = t;

    if (s < N_ANCHOR_SLOTS) {
      point_t p;
      if (!locoDeckGetAnchorPosition(id, &p)) {
        continue;   // position not advertised yet; the range is unusable
      }
      const float a[3] = {p.x, p.y, p.z};
      swarmEkfUpdateAnchor(&ekf, curve, a, d, cfg.sigAnchor, cfg.nisGate);
    } else if (s == SLOT_LEADER || s == SLOT_FOLLOWER) {
      swarmEkfUpdateChord(&ekf, curve, s == SLOT_LEADER, d,
                          cfg.sigChord, cfg.nisGate);
    }
  }
}

static void publish(void) {
  const float R2D = 180.0f / (float)M_PI;

  // States are published in SI so they pair element-for-element with swarmCov.
  // The human-readable view lives in swarmDiagnostics, in degrees.
  out.th = ekf.x[SWARM_EKF_TH];
  out.dk = ekf.x[SWARM_EKF_DK];
  out.dj = ekf.x[SWARM_EKF_DJ];
  out.we = ekf.x[SWARM_EKF_WE];
  out.r = ekf.x[SWARM_EKF_R];
  out.dz = ekf.x[SWARM_EKF_DZ];

  swarmEkfSeparations(&ekf, &out.phiK, &out.phiJ);

  float thPos, rPos;
  if (swarmPhaseGet(&thPos, &rPos)) {
    out.dTheta = swarmEkfWrap(out.th - thPos) * R2D;
  }

  // Report the covariance as one-sigma in natural units rather than raw
  // variances: 3 deg is readable, 0.0027 rad^2 is not.
  // Raw P diagonal, in the states' own SI units. This is what a downstream
  // covariance message wants: no conversion, no squaring, no unit guessing.
  out.pTh = ekf.P[SWARM_EKF_TH][SWARM_EKF_TH];
  out.pDk = ekf.P[SWARM_EKF_DK][SWARM_EKF_DK];
  out.pDj = ekf.P[SWARM_EKF_DJ][SWARM_EKF_DJ];
  out.pWe = ekf.P[SWARM_EKF_WE][SWARM_EKF_WE];
  out.pR  = ekf.P[SWARM_EKF_R][SWARM_EKF_R];
  out.pDz = ekf.P[SWARM_EKF_DZ][SWARM_EKF_DZ];

  // The same information as one sigma in natural units, for reading by eye:
  // "3 deg" is interpretable, "0.0027 rad^2" is not.
  out.sTh = sqrtf(out.pTh) * R2D;
  out.sDk = sqrtf(out.pDk) * R2D;
  out.sDj = sqrtf(out.pDj) * R2D;
  out.sWe = sqrtf(out.pWe);
  out.sR  = sqrtf(out.pR);
  out.sDz = sqrtf(out.pDz);

  out.nis = ekf.lastNis;
  out.init = ekf.initialised ? 1 : 0;
  out.nAcc = (uint16_t)ekf.nAccept;
  out.nRej = (uint16_t)ekf.nReject;

  // Stamped last, so a reader that sees a new seq has seen the whole update.
  out.seq++;
}

void appMain(void) {
  DEBUG_PRINT("Swarm encirclement: curve + phase + filter\n");

  if (!swarmPhaseInit()) {
    DEBUG_PRINT("ERROR: state estimate unavailable, app idle\n");
    while (true) {
      vTaskDelay(M2T(1000));
    }
  }

  // Defaults live in swarm_ekf.c so there is one source of truth; the
  // swarmFilter.q* and p* parameters override them at runtime.
  swarmEkfDefaults(&ekf);
  swarmControlInit();

  TickType_t lastWake = xTaskGetTickCount();

  while (true) {
    vTaskDelayUntil(&lastWake, M2T(UPDATE_PERIOD_MS));

    const swarmCurve_t* curve = swarmCurveGet();

    swarmPhaseUpdate(curve);

    if (!cfg.enable) {
      continue;
    }

    if (cfg.reset) {
      cfg.reset = 0;
      ekf.initialised = false;
    }

    if (!ekf.initialised) {
      if (!tryInit(curve)) {
        continue;
      }
    }

    ekf.dNom = 2.0f * (float)M_PI / (float)(cfg.n > 0 ? cfg.n : 3);

    swarmEkfPredict(&ekf, cfg.wz, DT);
    consumeRanges(curve);
    publish();

    // The controller reads the filter after it has been updated this cycle,
    // and gates itself on the covariance, so it is safe to call unconditionally.
    swarmControlUpdate(&ekf, curve, DT);
  }
}

/**
 * =====================================================================
 * Range-driven encirclement filter: outputs.
 * =====================================================================
 *
 * UNITS: everything in this group is SI -- angles in RADIANS, rates in rad/s.
 * That is deliberate: swarmCov publishes the matching variances in rad^2, so
 * the two groups pair element-for-element with no conversion. The degree-valued
 * view for reading by eye is in swarmDiagnostics.
 *
 * Sign conventions, once, so the individual entries can be terse:
 *
 *   theta   phase on the curve, CCW from the +x axis of the anchor frame,
 *           wrapped to [-pi, pi). +x is the direction from the corner anchor
 *           A0 toward the x-arm anchor A1.
 *
 *   delta   FORMATION ERROR, not a separation. Zero means perfect nominal
 *           spacing. The neighbours are defined as
 *               theta_leader   = theta + 360/n + delta_k
 *               theta_follower = theta - 360/n + delta_j
 *           so delta_k > 0 means the leader is FURTHER ahead than nominal, and
 *           delta_j > 0 means the follower has closed up (is less far behind).
 *
 *   phi     what the controller consumes, phi = theta_ego - theta_neighbour.
 *           The leader is ahead, so phi_ki is NEGATIVE (-2pi/3 for n=3 at
 *           perfect spacing) and phi_ji is POSITIVE (+2pi/3).
 */
LOG_GROUP_START(swarmFilter)

/**
 * @brief Filter estimate of the ego phase theta_i [rad]
 *
 * CCW from the +x axis, wrapped to [-pi, pi). This is the primary phase output;
 * compare against swarmDiagnostics.dTh (in degrees) to check it against the
 * independent position-derived value.
 */
LOG_ADD(LOG_FLOAT, th, &out.th)

/**
 * @brief Leader formation error delta_k [rad]
 *
 * Deviation of the leader from its nominal slot, NOT the separation itself.
 * Zero is perfect spacing; positive means the leader is further ahead than
 * nominal. Expected to stay small once the formation controller is closed.
 */
LOG_ADD(LOG_FLOAT, dk, &out.dk)

/**
 * @brief Follower formation error delta_j [rad]
 *
 * Zero is perfect spacing; positive means the follower has closed up.
 */
LOG_ADD(LOG_FLOAT, dj, &out.dj)

/**
 * @brief Separation to the leader, phi_ki [rad]
 *
 * phi_ki = theta_ego - theta_leader = -(2pi/n + delta_k). NEGATIVE by
 * convention because the leader is ahead: -2.094 rad for three agents at
 * nominal spacing. This is a controller input.
 */
LOG_ADD(LOG_FLOAT, phiK, &out.phiK)

/**
 * @brief Separation to the follower, phi_ji [rad]
 *
 * phi_ji = +(2pi/n - delta_j). POSITIVE: +2.094 rad for three agents at nominal
 * spacing. This is a controller input.
 */
LOG_ADD(LOG_FLOAT, phiJ, &out.phiJ)

/**
 * @brief Estimated angular rate error omega_e [rad/s]
 *
 * The discrepancy between the commanded rate swarmFilter.wz and the rate the vehicle
 * actually achieves, learned by the filter rather than assumed. A persistent
 * non-zero value means the formation is running fast or slow against its
 * command; without this state that mismatch would appear as a permanent lag in
 * theta that no amount of process noise could remove.
 */
LOG_ADD(LOG_FLOAT, we, &out.we)

/**
 * @brief Estimated encirclement radius r [m]
 *
 * Distance from the encirclement centre, constant along the curve whatever the
 * elevation profile. It is part of the state vector, hence in this group; its
 * variance is swarmCov.r.
 */
LOG_ADD(LOG_FLOAT, r, &out.r)

/**
 * @brief Vertical offset from the modelled curve, dz [m]
 *
 * How far above (+) or below (-) swarmCurve.cz the filter believes the
 * vehicle actually is. Near zero once cz matches the flown altitude. It is
 * ESTIMATED rather than assumed because with a compact anchor set an
 * unmodelled altitude offset biases theta by 26-43 deg per metre, so 10 cm
 * would be several degrees of phase error.
 */
LOG_ADD(LOG_FLOAT, dz, &out.dz)

/**
 * @brief Publish sequence number, increments once per filter cycle
 *
 * swarmFilter and swarmCov are separate log blocks driven by independent
 * timers, so they are NOT sampled atomically. Both carry this counter: when the
 * two agree, the state and its covariance came from the same filter cycle.
 * Wraps at 65535, which at 50 Hz is about 22 minutes -- far longer than needed
 * to pair adjacent messages.
 */
LOG_ADD(LOG_UINT16, seq, &out.seq)

LOG_GROUP_STOP(swarmFilter)

/**
 * =====================================================================
 * Filter health and diagnostics.
 * =====================================================================
 * Watch dTh and nis. Between them they catch the two ways this filter can be
 * wrong: diverged (dTh), or internally inconsistent (nis).
 */
LOG_GROUP_START(swarmDiagnostics)

/**
 * @brief CROSS-CHECK: filter phase minus position-derived phase [deg]
 *
 * The filter fuses UWB ranges directly and never consults the vehicle's EKF
 * position, so swarmPhase.thDeg is an INDEPENDENT second opinion computed from a
 * different measurement path (IMU plus anchor ranges). This is their
 * difference, and it is the only thing that can catch the filter latching onto
 * the wrong branch of the chord's ahead/behind ambiguity.
 *
 * Expect a few degrees of disagreement: the position-derived value inherits
 * whatever error the EKF position has. A PERSISTENT large value (say beyond
 * 20 deg) means the filter has diverged or picked the wrong branch -- force a
 * re-seed with swarmFilter.rst.
 */
LOG_ADD(LOG_FLOAT, dTh, &out.dTheta)

/**
 * @brief Normalised innovation squared of the last accepted update
 *
 * NIS = innovation^2 / S, where S is the innovation variance. For a correctly
 * tuned filter this is chi-squared with one degree of freedom, so it should
 * AVERAGE ABOUT 1.
 *
 *   persistently >> 1  the filter is overconfident: swarmFilter.sigA / sigC are too
 *                      small, the process noise is too small, or there is an
 *                      unmodelled bias (antenna delay, wrong anchor position).
 *   persistently << 1  over-conservative; the noise settings are pessimistic
 *                      and the estimate is sluggish for no reason.
 */
LOG_ADD(LOG_FLOAT, nis, &out.nis)

/**
 * @brief 1 once the filter has been seeded and is running
 *
 * The filter self-seeds from the position-derived phase as soon as that becomes
 * valid. If this stays 0, the state estimate is unavailable or the vehicle is
 * on the orbit axis where azimuth is undefined.
 */
LOG_ADD(LOG_UINT8, init, &out.init)

/**
 * @brief Count of accepted measurements (wraps at 65535)
 *
 * Read together with nRej: the RATIO is what matters, not the absolute value.
 */
LOG_ADD(LOG_UINT16, nAcc, &out.nAcc)

/**
 * @brief Count of measurements rejected by the innovation gate (wraps)
 *
 * A few percent is healthy -- that is the gate doing its job on multipath. A
 * large fraction means either swarmFilter.gate is too tight, or sigA / sigC are too
 * small so that ordinary noise looks like an outlier.
 */
LOG_ADD(LOG_UINT16, nRej, &out.nRej)

/**
 * @brief Current one-sigma uncertainty in the phase [deg]
 *
 * The square root of the theta entry of P, in degrees. Watch it settle after a
 * reset: it should fall from the seeded swarmFilter.pTh and then hold steady.
 * A collapse toward zero while the estimate is still wrong is the signature of
 * an overconfident filter -- measurements being counted more than once, or a
 * measurement noise set far below the truth.
 */
LOG_ADD(LOG_FLOAT, sTh, &out.sTh)

/**
 * @brief Current one-sigma uncertainty in the leader formation error [deg]
 *
 * Starts wide (delta is unknown at init) and should shrink once chords arrive.
 * If it stays wide, the chords are not reaching the filter -- check that the
 * leader slot is populated and its ranges are fresh.
 */
LOG_ADD(LOG_FLOAT, sDk, &out.sDk)

/**
 * @brief Current one-sigma uncertainty in the radius [m]
 */
LOG_ADD(LOG_FLOAT, sR, &out.sR)

/**
 * @brief Current one-sigma uncertainty in the follower formation error [deg]
 */
LOG_ADD(LOG_FLOAT, sDj, &out.sDj)

/**
 * @brief Current one-sigma uncertainty in the rate error [rad/s]
 */
LOG_ADD(LOG_FLOAT, sWe, &out.sWe)

LOG_GROUP_STOP(swarmDiagnostics)

/**
 * =====================================================================
 * Covariance: the diagonal of P, in the states' own SI units.
 * =====================================================================
 * Ordered to match the state vector, so element i here is the variance of
 * state i in the swarmFilter group:
 *
 *     th  dk  dj  we  r     <- swarmFilter (states)
 *     th  dk  dj  we  r     <- swarmCov    (variances)
 *
 * Publish the two together and a subscriber can assemble a state-with-
 * covariance message without converting or squaring anything.
 *
 * Only the diagonal is published. The off-diagonals are non-zero -- the
 * strongest is theta against omega_e, which the prediction creates directly --
 * so this is not the full covariance and should not be treated as one where
 * correlations matter.
 */
LOG_GROUP_START(swarmCov)

/** @brief Variance of the phase theta_i [rad^2] */
LOG_ADD(LOG_FLOAT, th, &out.pTh)

/** @brief Variance of the leader formation error delta_k [rad^2] */
LOG_ADD(LOG_FLOAT, dk, &out.pDk)

/** @brief Variance of the follower formation error delta_j [rad^2] */
LOG_ADD(LOG_FLOAT, dj, &out.pDj)

/** @brief Variance of the rate error omega_e [(rad/s)^2] */
LOG_ADD(LOG_FLOAT, we, &out.pWe)

/** @brief Variance of the radius r [m^2] */
LOG_ADD(LOG_FLOAT, r, &out.pR)

/**
 * @brief Publish sequence number; matches swarmFilter.seq
 *
 * Compare against swarmFilter.seq to confirm a state and its covariance came
 * from the same filter cycle. See the note there.
 */

/**
 * @brief Variance of dz [m^2]
 */
LOG_ADD(LOG_FLOAT, dz, &out.pDz)
LOG_ADD(LOG_UINT16, seq, &out.seq)

LOG_GROUP_STOP(swarmCov)

/**
 * =====================================================================
 * Filter configuration.
 * =====================================================================
 *
 * The process-noise entries (qTh, qD, qW, qR) are variances added PER SECOND,
 * so a value of 1e-3 on a phase state means the phase is allowed to wander with
 * a standard deviation of about sqrt(1e-3) ~ 0.03 rad after one second of
 * dead reckoning. Bigger means the filter trusts its own prediction less and
 * the measurements more: faster to react, noisier at rest.
 *
 * Tune against swarmDiagnostics.nis, not by eye: it should average about 1.
 */
PARAM_GROUP_START(swarmFilter)

/**
 * @brief Run the filter (1) or leave it idle (0)
 *
 * When set to 0 the state FREEZES rather than resetting -- no prediction and no
 * updates. Re-enabling therefore resumes from a stale estimate, so follow it
 * with swarmFilter.rst unless the vehicle has not moved.
 */
PARAM_ADD(PARAM_UINT8, en, &cfg.enable)

/**
 * @brief Write 1 to force re-initialisation; self-clearing
 *
 * Re-seeds theta and r from the position-derived phase, zeroes the formation
 * errors and the rate error, and restores the initial covariance. Use after
 * physically moving the drones, after changing swarmFilter.n, or when swarmDiagnostics.dTh
 * shows the filter has diverged.
 *
 * Zeroing the formation errors is also what selects the correct branch of the
 * chord's ahead/behind ambiguity, so a reset always comes back on the
 * CCW-ordered branch.
 */
PARAM_ADD(PARAM_UINT8, rst, &cfg.reset)

/**
 * @brief Number of agents in the ring, n
 *
 * Sets the nominal spacing to 360/n degrees. This MUST match both the real
 * number of drones and the ring order used by tools/swarm/setup_swarm.py to
 * assign the leader and follower slots. If it is wrong the formation errors
 * silently absorb the mismatch as a constant offset and every separation is
 * wrong by the same amount. Change it, then trigger swarmFilter.rst.
 */
PARAM_ADD(PARAM_UINT8, n, &cfg.n)

/**
 * @brief Commanded angular rate of the formation [rad/s]
 *
 * Feeds the prediction: theta advances by (wz + omega_e) * dt. This is what you
 * are COMMANDING, not a measurement -- set it to 0 for stationary bench tests,
 * and to the controller's commanded rate in flight.
 *
 * Getting it wrong is not fatal: omega_e absorbs a constant error within a few
 * seconds. A large mismatch simply means the prediction contributes less and
 * the filter leans harder on the ranges.
 */
PARAM_ADD(PARAM_FLOAT, wz, &cfg.wz)

/**
 * @brief Anchor range measurement noise, one standard deviation [m]
 *
 * Set it to the ranging noise you actually observe (tools/swarm/hybrid_ranging.py
 * reports it per remote); roughly 0.02 m on this hardware when static. Too
 * small and ordinary noise trips the innovation gate, so good measurements are
 * thrown away; too large and the filter is sluggish.
 */
PARAM_ADD(PARAM_FLOAT, sigA, &cfg.sigAnchor)

/**
 * @brief Crazyflie-to-Crazyflie chord measurement noise, one sigma [m]
 *
 * Kept separate from sigA because the chords are typically noisier: neighbours
 * transmit on a randomised, contended schedule, and their ranging degrades as
 * more agents share the airtime. Raise this rather than sigA if only the chord
 * innovations look large.
 */
PARAM_ADD(PARAM_FLOAT, sigC, &cfg.sigChord)

/**
 * @brief Innovation gate, in units of normalised innovation squared; 0 disables
 *
 * A measurement is rejected when innovation^2 / S exceeds this. Because that
 * quantity is chi-squared with one degree of freedom, the threshold is the
 * SQUARE of a sigma multiple: 9 is 3 sigma, 25 is 5 sigma (the default), 100 is
 * 10 sigma.
 *
 * This is the filter's own defence against multipath, and it is independent of
 * the median filter in the ranging layer (tdoa3.hmOutTh) -- that one rejects a
 * range that disagrees with the recent history for that remote, this one
 * rejects a range that disagrees with the STATE. Both are worth having.
 *
 * Watch swarmDiagnostics.nRej after changing it.
 */
PARAM_ADD(PARAM_FLOAT, gate, &cfg.nisGate)

/**
 * @brief Process noise on the common phase theta_i [rad^2 per second]
 *
 * How far the true phase may drift away from the (wz + omega_e) * dt
 * prediction: unmodelled wind, control lag, tracking error. This is the LARGE
 * one of the four; see qD.
 */
PARAM_ADD(PARAM_FLOAT, qTh, &ekf.qTheta)

/**
 * @brief Process noise on the formation errors delta_k, delta_j [rad^2/s]
 *
 * KEEP THIS WELL BELOW qTh -- it is the entire reason for these coordinates.
 * The dominant disturbances (wind, a mis-scaled rate command, a systematically
 * slow loop) displace the formation AS A WHOLE, leaving the spacing between
 * agents comparatively undisturbed. Encoding that means a large common-mode
 * noise and small differential noise.
 *
 * Setting qD close to qTh throws the benefit away and is equivalent to
 * filtering each agent's phase independently. Raise it only if you expect
 * agent-SPECIFIC disturbances, such as one vehicle with a failing motor.
 */
PARAM_ADD(PARAM_FLOAT, qD, &ekf.qDelta)

/**
 * @brief Process noise on the rate error omega_e [rad^2 per second^3]
 *
 * How quickly omega_e is allowed to change. It is meant to capture a slowly
 * varying bias between commanded and achieved rate, so keep it small: too large
 * and omega_e chases measurement noise instead of settling on the bias.
 */
PARAM_ADD(PARAM_FLOAT, qW, &ekf.qOmega)

/**
 * @brief Process noise on the radius r [m^2 per second]
 *
 * The radius is nominally constant and actively regulated, so this should be
 * small. Raise it only if the vehicle is expected to move radially, for example
 * while the encirclement radius is being commanded to change.
 */
PARAM_ADD(PARAM_FLOAT, qR, &ekf.qR)

/**
 * @brief Process noise for dz, the vertical offset [m^2/s]
 *
 * How fast the filter lets its altitude belief wander. Larger than qR:
 * altitude is actively controlled and drifts, whereas the orbit radius is
 * a commanded constant. Default 1e-4.
 */
PARAM_ADD(PARAM_FLOAT, qZ, &ekf.qZ)

/**
 * @brief Initial variance of the phase theta [rad^2]
 *
 * How wrong the SEED may be, not how fast the truth drifts -- that is qTh. It
 * is a variance, so 0.25 means a one-sigma initial uncertainty of 0.5 rad
 * (about 29 deg).
 *
 * Applied only at initialisation: change it, then trigger swarmFilter.rst for
 * it to take effect. Setting it to 0 is ignored and a safe default substituted,
 * because zero variance claims perfect knowledge of the seed and would stop the
 * filter from ever correcting it.
 */
PARAM_ADD(PARAM_FLOAT, pTh, &ekf.p0Theta)

/**
 * @brief Initial variance of the formation errors delta_k, delta_j [rad^2]
 *
 * Unlike qD, this should be LARGE: the formation errors are genuinely unknown
 * at initialisation, since they are seeded to zero and only the chords can
 * reveal them. Starting narrow here would make the filter reluctant to believe
 * the first chord measurements.
 *
 * Takes effect on swarmFilter.rst.
 */
PARAM_ADD(PARAM_FLOAT, pD, &ekf.p0Delta)

/**
 * @brief Initial variance of the rate error omega_e [(rad/s)^2]
 *
 * Takes effect on swarmFilter.rst.
 */
PARAM_ADD(PARAM_FLOAT, pW, &ekf.p0Omega)

/**
 * @brief Initial variance of the radius r [m^2]
 *
 * 0.09 is a one-sigma of 0.3 m. Takes effect on swarmFilter.rst.
 */
PARAM_ADD(PARAM_FLOAT, pR, &ekf.p0R)

/**
 * @brief Initial variance of dz [m^2]
 *
 * How wrong swarmCurve.cz might be at reset. Default 0.04 = (0.2 m)^2.
 */
PARAM_ADD(PARAM_FLOAT, pZ, &ekf.p0Z)

PARAM_GROUP_STOP(swarmFilter)
