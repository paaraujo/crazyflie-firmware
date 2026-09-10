/**
 * Host unit test for swarm_curve.c.
 *
 * Build and run without hardware:
 *     cd examples/app_swarm_encircle
 *     gcc -DSWARM_CURVE_HOST_TEST -Isrc -o /tmp/tc test/test_swarm_curve.c src/swarm_curve.c -lm
 *     /tmp/tc
 *
 * Expected values are generated independently in Python from the same formulae
 * (see the generator comment in each case), so this checks the C against a
 * separate implementation rather than against itself.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "swarm_curve.h"

void swarmCurveSetForTest(const swarmCurve_t* c);

static int failures = 0;
static int checks = 0;

static void check(const char* what, double got, double want, double tol) {
  checks++;
  const double err = fabs(got - want);
  if (err > tol || isnan(got)) {
    printf("  FAIL  %-34s got %+.6f  want %+.6f  (err %.2e)\n", what, got, want, err);
    failures++;
  }
}

/* Curve A from tools/swarm/fit_elevation.py: the saddle, b(theta) = -20deg * cos(2 theta).
 * Coefficients: K=2, b0=0, a1=0, b1=0, a2=-0.34906585, b2=0. */
static swarmCurve_t curveA(void) {
  swarmCurve_t c = {0};
  c.cx = 0.0f; c.cy = 0.0f; c.cz = 1.0f;
  c.K = 2;
  c.b0 = 0.0f;
  c.alpha[1] = -0.34906585f;   /* alpha_2 = -20 deg in radians */
  return c;
}

int main(void) {
  printf("swarm_curve host test\n\n");

  /* ---------------------------------------------------------------
   * 1. Flat circle: K = 0, b0 = 0. The degenerate case the vehicle
   *    flies today, and the base case everything else reduces to.
   * --------------------------------------------------------------- */
  {
    swarmCurve_t c = {0};
    c.cx = 0.0f; c.cy = 0.0f; c.cz = 1.0f; c.K = 0; c.b0 = 0.0f;

    float b, db, q[3], dq[3], u[3];
    swarmCurveEvalB(&c, 0.9f, &b, &db);
    check("flat: b", b, 0.0, 1e-6);
    check("flat: db", db, 0.0, 1e-6);

    swarmCurveQ(&c, 0.0f, 1.0f, q);
    check("flat: q(0).x", q[0], 1.0, 1e-6);
    check("flat: q(0).y", q[1], 0.0, 1e-6);
    check("flat: q(0).z", q[2], 1.0, 1e-6);

    /* tangent of a level circle is purely tangential, magnitude r */
    swarmCurveDq(&c, 0.0f, 1.0f, dq);
    check("flat: dq(0).x", dq[0], 0.0, 1e-6);
    check("flat: dq(0).y", dq[1], 1.0, 1e-6);
    check("flat: dq(0).z", dq[2], 0.0, 1e-6);

    swarmCurveU(&c, 0.0f, u);
    check("flat: u(0).x", u[0], 1.0, 1e-6);
    check("flat: u(0).z", u[2], 0.0, 1e-6);
  }

  /* ---------------------------------------------------------------
   * 2. Saddle. Python reference:
   *      b(t)  = -0.34906585*cos(2t)
   *      q     = [r cos b cos t, r cos b sin t, cz - r sin b]
   *    at t = 0.7 rad, r = 1.0, cz = 1.0:
   *      b     = -0.34906585*cos(1.4)      = -0.05932973
   *      q     = [0.76349645, 0.64308419, 1.05929492]
   * --------------------------------------------------------------- */
  {
    const swarmCurve_t c = curveA();
    float b, db, q[3];

    swarmCurveEvalB(&c, 0.7f, &b, &db);
    check("saddle: b(0.7)", b, -0.05932973, 1e-6);
    /* db/dt = +2*0.34906585*sin(2t) = 0.69813170*sin(1.4) */
    check("saddle: db(0.7)", db, 0.68797370, 1e-6);

    swarmCurveQ(&c, 0.7f, 1.0f, q);
    check("saddle: q.x", q[0], 0.76349645, 1e-5);
    check("saddle: q.y", q[1], 0.64308419, 1e-5);
    check("saddle: q.z", q[2], 1.05929492, 1e-5);

    /* waypoint check: at theta = 0 the curve should sit 20 deg ABOVE centre,
       i.e. z = cz + r sin(20deg) = 1.342020 */
    swarmCurveQ(&c, 0.0f, 1.0f, q);
    check("saddle: q(0).z", q[2], 1.34202014, 1e-5);
    check("saddle: q(0).x", q[0], 0.93969262, 1e-5);
  }

  /* ---------------------------------------------------------------
   * 3. Proposition: the curve lies on the sphere of radius r about c,
   *    for every theta and any elevation profile.
   * --------------------------------------------------------------- */
  {
    const swarmCurve_t c = curveA();
    double worst = 0.0;
    for (int i = 0; i < 360; i++) {
      const float t = (float)(i * M_PI / 180.0);
      float q[3];
      swarmCurveQ(&c, t, 1.0f, q);
      const double e = fabs((double)swarmCurveRadius(&c, q) - 1.0);
      if (e > worst) worst = e;
    }
    check("sphere: max |r - 1| over orbit", worst, 0.0, 1e-6);
  }

  /* ---------------------------------------------------------------
   * 4. Corollary (exactness): azimuth equals phase EXACTLY, for any
   *    elevation profile. This is what removes the fixed-point inversion.
   * --------------------------------------------------------------- */
  {
    swarmCurve_t c = curveA();
    c.b0 = 0.30f;            /* deliberately severe, and asymmetric */
    c.alpha[0] = 0.20f;
    c.beta[2] = -0.25f;
    c.K = 3;

    double worst = 0.0;
    for (int i = 0; i < 720; i++) {
      const float t = (float)(i * M_PI / 360.0) - (float)M_PI;  /* -pi .. pi */
      float q[3];
      swarmCurveQ(&c, t, 1.3f, q);
      double d = (double)swarmCurveTheta(&c, q) - (double)t;
      while (d >  M_PI) d -= 2 * M_PI;
      while (d < -M_PI) d += 2 * M_PI;
      if (fabs(d) > worst) worst = fabs(d);
    }
    check("exactness: max |atan2 - theta|", worst, 0.0, 1e-5);
  }

  /* ---------------------------------------------------------------
   * 5. Tangent agrees with a central difference of q. Guards the
   *    analytic derivative used by the measurement Jacobians.
   * --------------------------------------------------------------- */
  {
    swarmCurve_t c = curveA();
    c.b0 = 0.12f;
    c.beta[0] = 0.18f;
    c.K = 2;

    const float r = 1.15f;
    const float eps = 1e-4f;
    double worst = 0.0;
    for (int i = 0; i < 72; i++) {
      const float t = (float)(i * 5.0 * M_PI / 180.0);
      float dq[3], qp[3], qm[3];
      swarmCurveDq(&c, t, r, dq);
      swarmCurveQ(&c, t + eps, r, qp);
      swarmCurveQ(&c, t - eps, r, qm);
      for (int k = 0; k < 3; k++) {
        const double fd = ((double)qp[k] - (double)qm[k]) / (2.0 * eps);
        const double e = fabs(fd - (double)dq[k]);
        if (e > worst) worst = e;
      }
    }
    check("tangent: max |analytic - central diff|", worst, 0.0, 2e-3);
  }

  /* ---------------------------------------------------------------
   * 6. Tangent is orthogonal to the radius: q' . (q - c) = 0, which
   *    follows from the curve lying on a sphere.
   * --------------------------------------------------------------- */
  {
    const swarmCurve_t c = curveA();
    double worst = 0.0;
    for (int i = 0; i < 180; i++) {
      const float t = (float)(i * 2.0 * M_PI / 180.0);
      float q[3], dq[3];
      swarmCurveQ(&c, t, 1.0f, q);
      swarmCurveDq(&c, t, 1.0f, dq);
      const double dot = (double)dq[0] * (q[0] - c.cx)
                       + (double)dq[1] * (q[1] - c.cy)
                       + (double)dq[2] * (q[2] - c.cz);
      if (fabs(dot) > worst) worst = fabs(dot);
    }
    check("tangent: max |q' . (q-c)|", worst, 0.0, 1e-5);
  }

  /* ---------------------------------------------------------------
   * 7. u = dq/dr is a unit vector, and q = c + r*u.
   * --------------------------------------------------------------- */
  {
    const swarmCurve_t c = curveA();
    double worstNorm = 0.0, worstRecon = 0.0;
    for (int i = 0; i < 90; i++) {
      const float t = (float)(i * 4.0 * M_PI / 180.0);
      const float r = 1.4f;
      float u[3], q[3];
      swarmCurveU(&c, t, u);
      swarmCurveQ(&c, t, r, q);
      const double n = sqrt((double)u[0]*u[0] + (double)u[1]*u[1] + (double)u[2]*u[2]);
      if (fabs(n - 1.0) > worstNorm) worstNorm = fabs(n - 1.0);
      const double ex = fabs((c.cx + r * u[0]) - q[0]);
      const double ey = fabs((c.cy + r * u[1]) - q[1]);
      const double ez = fabs((c.cz + r * u[2]) - q[2]);
      const double e = ex > ey ? (ex > ez ? ex : ez) : (ey > ez ? ey : ez);
      if (e > worstRecon) worstRecon = e;
    }
    check("u: max |1 - |u||", worstNorm, 0.0, 1e-6);
    check("u: max |c + r u - q|", worstRecon, 0.0, 1e-6);
  }

  /* ---------------------------------------------------------------
   * 8. Off-centre encirclement centre is honoured.
   * --------------------------------------------------------------- */
  {
    swarmCurve_t c = curveA();
    c.cx = 0.35f; c.cy = -0.20f; c.cz = 1.60f;
    float q[3];
    swarmCurveQ(&c, 1.1f, 1.0f, q);
    check("offset: radius about c", swarmCurveRadius(&c, q), 1.0, 1e-6);
    double d = (double)swarmCurveTheta(&c, q) - 1.1;
    check("offset: theta recovered", d, 0.0, 1e-5);
  }

  printf("\n%d checks, %d failures\n", checks, failures);
  if (failures == 0) {
    printf("PASS\n");
  }
  return failures == 0 ? 0 : 1;
}
