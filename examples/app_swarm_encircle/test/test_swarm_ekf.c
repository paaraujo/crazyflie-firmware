/**
 * Host unit test for swarm_ekf.c.
 *
 *     cd examples/app_swarm_encircle
 *     gcc -DSWARM_CURVE_HOST_TEST -Isrc -o /tmp/te test/test_swarm_ekf.c \
 *         src/swarm_ekf.c src/swarm_curve.c -lm
 *     /tmp/te
 *
 * The important tests are the last two: a closed-loop simulation against a known
 * truth, which is the only thing that catches a filter that is internally
 * consistent but wrong.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "swarm_curve.h"
#include "swarm_ekf.h"

static int failures = 0, checks = 0;

static void check(const char* what, double got, double want, double tol) {
  checks++;
  if (fabs(got - want) > tol || isnan(got)) {
    printf("  FAIL  %-40s got %+.6f  want %+.6f\n", what, got, want);
    failures++;
  }
}

static swarmCurve_t flatCurve(void) {
  swarmCurve_t c = {0};
  c.cx = 0.0f; c.cy = 0.0f; c.cz = 1.0f; c.K = 0; c.b0 = 0.0f;
  return c;
}

static swarmCurve_t saddleCurve(void) {
  swarmCurve_t c = flatCurve();
  c.K = 2;
  c.alpha[1] = -0.34906585f;   /* -20 deg cos(2 theta) */
  return c;
}

int main(void) {
  printf("swarm_ekf host test\n\n");
  const float DNOM = 2.0f * (float)M_PI / 3.0f;   /* three agents */

  /* -----------------------------------------------------------------
   * 1. Covariance propagation must equal the general F P F'.
   *    This is where an in-place shortcut silently double-counts.
   * ----------------------------------------------------------------- */
  {
    swarmEkf_t f;
    swarmEkfDefaults(&f);
    swarmEkfInit(&f, 0.3f, 1.0f, DNOM);
    f.qTheta = f.qDelta = f.qOmega = f.qR = 0.0f;   /* isolate F P F' */

    /* seed a non-trivial symmetric P */
    float P0[SWARM_EKF_DIM][SWARM_EKF_DIM];
    for (int i = 0; i < SWARM_EKF_DIM; i++) {
      for (int j = i; j < SWARM_EKF_DIM; j++) {
        const float v = (i == j) ? (0.4f + 0.1f * i) : (0.05f * (i + 1) - 0.02f * j);
        P0[i][j] = P0[j][i] = v;
      }
    }
    memcpy(f.P, P0, sizeof(P0));

    const float dt = 0.02f;
    swarmEkfPredict(&f, 0.0f, dt);

    /* explicit reference */
    float F[SWARM_EKF_DIM][SWARM_EKF_DIM] = {0}, ref[SWARM_EKF_DIM][SWARM_EKF_DIM] = {0};
    for (int i = 0; i < SWARM_EKF_DIM; i++) F[i][i] = 1.0f;
    F[SWARM_EKF_TH][SWARM_EKF_WE] = dt;
    for (int i = 0; i < SWARM_EKF_DIM; i++)
      for (int j = 0; j < SWARM_EKF_DIM; j++) {
        double s = 0;
        for (int a = 0; a < SWARM_EKF_DIM; a++)
          for (int b = 0; b < SWARM_EKF_DIM; b++)
            s += F[i][a] * P0[a][b] * F[j][b];
        ref[i][j] = (float)s;
      }

    double worst = 0;
    for (int i = 0; i < SWARM_EKF_DIM; i++)
      for (int j = 0; j < SWARM_EKF_DIM; j++) {
        const double e = fabs(f.P[i][j] - ref[i][j]);
        if (e > worst) worst = e;
      }
    check("predict: max |P - F P F'|", worst, 0.0, 1e-6);
  }

  /* -----------------------------------------------------------------
   * 2. omega_e integrates into theta.
   * ----------------------------------------------------------------- */
  {
    swarmEkf_t f;
    swarmEkfDefaults(&f);
    swarmEkfInit(&f, 0.0f, 1.0f, DNOM);
    f.x[SWARM_EKF_WE] = 0.5f;
    for (int i = 0; i < 100; i++) swarmEkfPredict(&f, 1.0f, 0.01f);
    /* 1 s at (wz + we) = 1.5 rad/s */
    check("predict: theta after 1 s", f.x[SWARM_EKF_TH], 1.5, 1e-3);
  }

  /* -----------------------------------------------------------------
   * 3. Structure: chords are blind to the common mode on a flat circle.
   *    Verified through the public API by checking that a chord update
   *    moves delta but leaves theta untouched.
   * ----------------------------------------------------------------- */
  {
    const swarmCurve_t c = flatCurve();
    swarmEkf_t f;
    swarmEkfDefaults(&f);
    swarmEkfInit(&f, 0.4f, 1.0f, DNOM);
    const float th0 = f.x[SWARM_EKF_TH];

    /* a chord longer than predicted: must be explained by delta, not theta */
    float qi[3], qn[3];
    swarmCurveQ(&c, f.x[SWARM_EKF_TH], 1.0f, qi);
    swarmCurveQ(&c, f.x[SWARM_EKF_TH] + DNOM, 1.0f, qn);
    const float dx = qi[0]-qn[0], dy = qi[1]-qn[1], dz = qi[2]-qn[2];
    const float hPred = sqrtf(dx*dx + dy*dy + dz*dz);

    swarmEkfUpdateChord(&f, &c, true, hPred + 0.10f, 0.03f, 0.0f);
    check("flat chord: theta unchanged", f.x[SWARM_EKF_TH] - th0, 0.0, 1e-6);
    checks++;
    if (fabsf(f.x[SWARM_EKF_DK]) < 1e-4f) {
      printf("  FAIL  flat chord did not move delta_k\n"); failures++;
    }
    check("flat chord: delta_j untouched", f.x[SWARM_EKF_DJ], 0.0, 1e-9);
  }

  /* -----------------------------------------------------------------
   * 4. Innovation gate rejects a gross outlier and leaves the state alone.
   * ----------------------------------------------------------------- */
  {
    const swarmCurve_t c = flatCurve();
    const float A[3] = {0.0f, 0.0f, 0.0f};
    swarmEkf_t f;
    swarmEkfDefaults(&f);
    swarmEkfInit(&f, 0.4f, 1.0f, DNOM);
    const float th0 = f.x[SWARM_EKF_TH], r0 = f.x[SWARM_EKF_R];
    const uint32_t rej0 = f.nReject;

    swarmEkfUpdateAnchor(&f, &c, A, 50.0f, 0.03f, 25.0f);   /* absurd range */
    check("gate: theta unchanged", f.x[SWARM_EKF_TH] - th0, 0.0, 1e-9);
    check("gate: r unchanged", f.x[SWARM_EKF_R] - r0, 0.0, 1e-9);
    checks++;
    if (f.nReject != rej0 + 1) { printf("  FAIL  gate did not count a rejection\n"); failures++; }
  }

  /* -----------------------------------------------------------------
   * 5. Convergence against a known truth, flat circle.
   *    Three anchors placed as an L-template plus two neighbour chords,
   *    noise-free, with the filter started deliberately wrong.
   * ----------------------------------------------------------------- */
  {
    const swarmCurve_t c = flatCurve();
    const float A[3][3] = {{0,0,0}, {1.5f,0,0}, {0,1.5f,0}};

    const float thTrue = 0.9f, rTrue = 1.10f;
    const float dkTrue = 0.25f, djTrue = -0.18f;

    swarmEkf_t f;
    swarmEkfDefaults(&f);
    swarmEkfInit(&f, thTrue + 0.45f, rTrue - 0.25f, DNOM);   /* wrong on purpose */
    f.qTheta = 1e-5f; f.qDelta = 1e-6f; f.qOmega = 1e-6f; f.qR = 1e-6f;

    float qT[3], qK[3], qJ[3];
    swarmCurveQ(&c, thTrue, rTrue, qT);
    swarmCurveQ(&c, thTrue + DNOM + dkTrue, rTrue, qK);
    swarmCurveQ(&c, thTrue - DNOM + djTrue, rTrue, qJ);

    for (int it = 0; it < 400; it++) {
      swarmEkfPredict(&f, 0.0f, 0.02f);
      for (int m = 0; m < 3; m++) {
        const float d = sqrtf((qT[0]-A[m][0])*(qT[0]-A[m][0]) +
                              (qT[1]-A[m][1])*(qT[1]-A[m][1]) +
                              (qT[2]-A[m][2])*(qT[2]-A[m][2]));
        swarmEkfUpdateAnchor(&f, &c, A[m], d, 0.02f, 0.0f);
      }
      const float dk = sqrtf((qT[0]-qK[0])*(qT[0]-qK[0]) + (qT[1]-qK[1])*(qT[1]-qK[1]) + (qT[2]-qK[2])*(qT[2]-qK[2]));
      const float dj = sqrtf((qT[0]-qJ[0])*(qT[0]-qJ[0]) + (qT[1]-qJ[1])*(qT[1]-qJ[1]) + (qT[2]-qJ[2])*(qT[2]-qJ[2]));
      swarmEkfUpdateChord(&f, &c, true,  dk, 0.02f, 0.0f);
      swarmEkfUpdateChord(&f, &c, false, dj, 0.02f, 0.0f);
    }

    check("flat sim: theta", f.x[SWARM_EKF_TH], thTrue, 0.02);
    check("flat sim: r", f.x[SWARM_EKF_R], rTrue, 0.02);
    check("flat sim: delta_k", f.x[SWARM_EKF_DK], dkTrue, 0.05);
    check("flat sim: delta_j", f.x[SWARM_EKF_DJ], djTrue, 0.05);

    float phiK, phiJ;
    swarmEkfSeparations(&f, &phiK, &phiJ);
    check("flat sim: phi_ki", phiK, -(DNOM + dkTrue), 0.05);
    check("flat sim: phi_ji", phiJ, +(DNOM - djTrue), 0.05);
  }

  /* -----------------------------------------------------------------
   * 6. Same, on a distorted curve. Exercises the elevation profile in
   *    the measurement model and its tangent.
   * ----------------------------------------------------------------- */
  {
    const swarmCurve_t c = saddleCurve();
    const float A[3][3] = {{0,0,0}, {1.5f,0,0}, {0,1.5f,0}};

    const float thTrue = -1.30f, rTrue = 1.05f;
    const float dkTrue = -0.20f, djTrue = 0.15f;

    swarmEkf_t f;
    swarmEkfDefaults(&f);
    swarmEkfInit(&f, thTrue + 0.40f, rTrue + 0.20f, DNOM);
    f.qTheta = 1e-5f; f.qDelta = 1e-6f; f.qOmega = 1e-6f; f.qR = 1e-6f;

    float qT[3], qK[3], qJ[3];
    swarmCurveQ(&c, thTrue, rTrue, qT);
    swarmCurveQ(&c, thTrue + DNOM + dkTrue, rTrue, qK);
    swarmCurveQ(&c, thTrue - DNOM + djTrue, rTrue, qJ);

    for (int it = 0; it < 600; it++) {
      swarmEkfPredict(&f, 0.0f, 0.02f);
      for (int m = 0; m < 3; m++) {
        const float d = sqrtf((qT[0]-A[m][0])*(qT[0]-A[m][0]) +
                              (qT[1]-A[m][1])*(qT[1]-A[m][1]) +
                              (qT[2]-A[m][2])*(qT[2]-A[m][2]));
        swarmEkfUpdateAnchor(&f, &c, A[m], d, 0.02f, 0.0f);
      }
      const float dk = sqrtf((qT[0]-qK[0])*(qT[0]-qK[0]) + (qT[1]-qK[1])*(qT[1]-qK[1]) + (qT[2]-qK[2])*(qT[2]-qK[2]));
      const float dj = sqrtf((qT[0]-qJ[0])*(qT[0]-qJ[0]) + (qT[1]-qJ[1])*(qT[1]-qJ[1]) + (qT[2]-qJ[2])*(qT[2]-qJ[2]));
      swarmEkfUpdateChord(&f, &c, true,  dk, 0.02f, 0.0f);
      swarmEkfUpdateChord(&f, &c, false, dj, 0.02f, 0.0f);
    }

    check("saddle sim: theta", f.x[SWARM_EKF_TH], thTrue, 0.03);
    check("saddle sim: r", f.x[SWARM_EKF_R], rTrue, 0.03);
    check("saddle sim: delta_k", f.x[SWARM_EKF_DK], dkTrue, 0.06);
    check("saddle sim: delta_j", f.x[SWARM_EKF_DJ], djTrue, 0.06);
  }

  /* -----------------------------------------------------------------
   * 7. Covariance stays symmetric and positive on the diagonal.
   * ----------------------------------------------------------------- */
  {
    const swarmCurve_t c = saddleCurve();
    const float A[3][3] = {{0,0,0}, {1.5f,0,0}, {0,1.5f,0}};
    swarmEkf_t f;
    swarmEkfDefaults(&f);
    swarmEkfInit(&f, 0.2f, 1.0f, DNOM);
    f.qTheta = 1e-4f; f.qDelta = 1e-6f; f.qOmega = 1e-5f; f.qR = 1e-6f;

    double worstAsym = 0.0, minDiag = 1e9;
    for (int it = 0; it < 2000; it++) {
      swarmEkfPredict(&f, 0.7f, 0.02f);
      float q[3];
      swarmCurveQ(&c, f.x[SWARM_EKF_TH], 1.0f, q);
      for (int m = 0; m < 3; m++) {
        const float d = sqrtf((q[0]-A[m][0])*(q[0]-A[m][0]) +
                              (q[1]-A[m][1])*(q[1]-A[m][1]) +
                              (q[2]-A[m][2])*(q[2]-A[m][2]));
        swarmEkfUpdateAnchor(&f, &c, A[m], d, 0.03f, 0.0f);
      }
      for (int i = 0; i < SWARM_EKF_DIM; i++) {
        if (f.P[i][i] < minDiag) minDiag = f.P[i][i];
        for (int j = i + 1; j < SWARM_EKF_DIM; j++) {
          const double e = fabs(f.P[i][j] - f.P[j][i]);
          if (e > worstAsym) worstAsym = e;
        }
      }
    }
    check("P: max asymmetry over 2000 steps", worstAsym, 0.0, 1e-9);
    checks++;
    if (!(minDiag > 0.0)) { printf("  FAIL  P diagonal went non-positive\n"); failures++; }
  }

  printf("\n%d checks, %d failures\n", checks, failures);
  if (failures == 0) printf("PASS\n");
  return failures ? 1 : 0;
}
