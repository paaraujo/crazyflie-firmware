/**
 * Host test for the encirclement control law.
 *
 * swarm_control.c cannot be linked on the host (commander, FreeRTOS, log,
 * param), so the geometry and saturation are reimplemented here from the same
 * equations and checked against independently derived expectations. What this
 * pins is the part where sign errors hide: the tangential/radial rotation into
 * the template frame, and the direction each correction pushes.
 */
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int failures = 0, checks = 0;

static void check(const char* what, double got, double want, double tol) {
  checks++;
  const int ok = fabs(got - want) <= tol;
  if (!ok) {
    failures++;
    printf("  FAIL  %-44s got %+10.5f  want %+10.5f\n", what, got, want);
  }
}

static float clampf(float v, float lim) {
  if (v > lim) return lim;
  if (v < -lim) return -lim;
  return v;
}

/* The law under test, copied from swarm_control.c. */
static void law(float th, float r, float dk, float dj,
                float rTarget, float wNom, float kPhi, float kR,
                float vMaxT, float vMaxR, float ramp,
                float* vx, float* vy, float* vt, float* vr, float* ePhi) {
  *ePhi = dk + dj;
  const float eR = r - rTarget;
  *vt = clampf(r * (ramp * wNom + kPhi * (*ePhi)), vMaxT);
  *vr = clampf(-kR * eR, vMaxR);
  const float ct = cosf(th), st = sinf(th);
  *vx = (*vr) * ct - (*vt) * st;
  *vy = (*vr) * st + (*vt) * ct;
}

int main(void) {
  printf("swarm_control host test\n\n");
  const float R = 0.6f, RT = 0.6f, VT = 0.35f, VR = 0.25f;
  float vx, vy, vt, vr, ep;

  /* 1. Pure orbit: velocity is purely tangential, perpendicular to the radius,
   *    and in the CCW direction (increasing theta). */
  for (int k = 0; k < 8; k++) {
    const float th = (float)(2.0 * M_PI * k / 8.0);
    law(th, R, 0, 0, RT, 0.2f, 0.5f, 0.8f, VT, VR, 1.0f,
        &vx, &vy, &vt, &vr, &ep);
    const float radial = vx * cosf(th) + vy * sinf(th);
    const float tang  = -vx * sinf(th) + vy * cosf(th);
    char buf[64];
    snprintf(buf, sizeof buf, "orbit: no radial component (phase %d)", k);
    check(buf, radial, 0.0, 1e-5);
    snprintf(buf, sizeof buf, "orbit: tangential = r*w (phase %d)", k);
    check(buf, tang, R * 0.2f, 1e-5);
    snprintf(buf, sizeof buf, "orbit: speed = r*w (phase %d)", k);
    check(buf, sqrt(vx*vx + vy*vy), R * 0.2f, 1e-5);
  }

  /* 2. Radius too LARGE -> velocity points INWARD (negative radial). */
  law(0.7f, R + 0.1f, 0, 0, RT, 0.0f, 0.5f, 0.8f, VT, VR, 1.0f,
      &vx, &vy, &vt, &vr, &ep);
  {
    const float th = 0.7f;
    const float radial = vx * cosf(th) + vy * sinf(th);
    check("radius too large -> inward", radial < 0.0f ? -1.0 : 1.0, -1.0, 0.5);
    check("radius error magnitude", radial, -0.8f * 0.1f, 1e-5);
  }

  /* 3. Radius too SMALL -> outward. */
  law(2.3f, R - 0.1f, 0, 0, RT, 0.0f, 0.5f, 0.8f, VT, VR, 1.0f,
      &vx, &vy, &vt, &vr, &ep);
  {
    const float th = 2.3f;
    const float radial = vx * cosf(th) + vy * sinf(th);
    check("radius too small -> outward", radial, +0.8f * 0.1f, 1e-5);
  }

  /* 4-5. Separation, derived from ACTUAL neighbour phases rather than from
   *      the formula. Build theta_i, theta_k, theta_j, measure the two gaps,
   *      and require the correction to push towards whichever side is wider.
   *      Writing the expectation independently of the law is the whole point:
   *      an earlier version of this test restated the law and therefore agreed
   *      with a law that had both the sign and the combination wrong. */
  {
    const float D = (float)(2.0 * M_PI / 3.0);
    struct { float dk, dj; const char* what; } tc[] = {
      { +0.20f,  0.00f, "leader further" },
      {  0.00f, +0.20f, "follower closer" },
      { +0.30f, +0.30f, "whole formation ahead" },
      { -0.15f,  0.00f, "leader closer" },
      {  0.00f, -0.25f, "follower further" },
      { +0.20f, -0.20f, "symmetric expansion" },
      { -0.20f, +0.20f, "symmetric contraction" },
    };
    for (unsigned i = 0; i < sizeof tc / sizeof tc[0]; i++) {
      const float thI = 0.37f;                      /* arbitrary ego phase */
      const float thK = thI + D + tc[i].dk;         /* leader, ahead  */
      const float thJ = thI - D + tc[i].dj;         /* follower, behind */
      const float gapLead = thK - thI;
      const float gapFoll = thI - thJ;
      const float imbalance = gapLead - gapFoll;    /* independent of the law */

      law(thI, R, tc[i].dk, tc[i].dj, RT, 0.0f, 0.5f, 0.8f, VT, VR, 1.0f,
          &vx, &vy, &vt, &vr, &ep);

      char buf[96];
      /* the correction must carry the same sign as the imbalance */
      snprintf(buf, sizeof buf, "sep: %s -> correct direction", tc[i].what);
      /* Threshold at 1e-5, not 1e-7: gapLead is computed as
       * (thI + D + dk) - thI, and that cancellation leaves ~1e-7 of float
       * error on quantities of order 2, so a genuinely symmetric case does not
       * land exactly on zero. 1e-5 is well above the noise and far below any
       * imbalance worth correcting (1e-5 rad is 0.0006 deg). */
      const double ZTOL = 1e-5;
      const double sgnGot  = (fabs(vt) < ZTOL) ? 0.0 : (vt > 0 ? 1.0 : -1.0);
      const double sgnWant = (fabs(imbalance) < ZTOL) ? 0.0
                                                      : (imbalance > 0 ? 1.0 : -1.0);
      check(buf, sgnGot, sgnWant, 0.5);

      /* and the magnitude must be r * kPhi * imbalance */
      snprintf(buf, sizeof buf, "sep: %s -> correct magnitude", tc[i].what);
      check(buf, vt, R * 0.5f * imbalance, 1e-5);

      /* a separation correction is purely tangential: it must not change r */
      snprintf(buf, sizeof buf, "sep: %s -> no radial component", tc[i].what);
      check(buf, vx * cosf(thI) + vy * sinf(thI), 0.0, 1e-6);
    }
  }

  /* 6. Saturation binds and does not change sign. */
  law(0.4f, R, -4.0f, +4.0f, RT, 2.0f, 0.5f, 0.8f, VT, VR, 1.0f,
      &vx, &vy, &vt, &vr, &ep);
  check("tangential saturates at +vMaxT", vt, VT, 1e-6);
  law(0.4f, R + 5.0f, 0, 0, RT, 0.0f, 0.5f, 0.8f, VT, VR, 1.0f,
      &vx, &vy, &vt, &vr, &ep);
  check("radial saturates at -vMaxR", vr, -VR, 1e-6);

  /* 7. The ramp scales only the feedforward, never the corrections: a radius
   *    error must be corrected at full gain from the first cycle. */
  law(0.0f, R + 0.1f, 0, 0, RT, 0.2f, 0.5f, 0.8f, VT, VR, 0.0f,
      &vx, &vy, &vt, &vr, &ep);
  check("ramp 0: no feedforward", vt, 0.0, 1e-6);
  check("ramp 0: radial still acts", vr, -0.8f * 0.1f, 1e-6);

  printf("\n%d checks, %d failures\n", checks, failures);
  printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
