#ifndef HORNER
#define HORNER
#include <cfloat>
#include <cmath>
#include <cstdio>

#include "eft.h"
namespace ehrlich_aberth_jax {

#ifdef __CUDACC__
#define EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE __host__ __device__
#else
#define EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE inline
#endif

/* Global Constants */
const double EPS = DBL_EPSILON / 2;
const double ETA = DBL_MIN;
/* Unit in First Place */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE double ufp(const double p) {
  double q = p / DBL_EPSILON + p;
  return fabs(fma(q, EPS - 1, q));
}
/* Gamma Constant */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE double gamma_const(const unsigned int n) {
  double s = 1.41421356237309504880;
  double g = (2 * n * EPS) * s;
  return g / ((1 - DBL_EPSILON) - g);
}
/* Fast Two Sum */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE void fast_two_sum(const double a, const double b,
                                                      struct eft *res) {
  res->fl_res = a + b;
  res->fl_err = (a - res->fl_res) + b;
}
/* Two Sum */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE void two_sum(const double a, const double b, struct eft *res) {
  res->fl_res = a + b;
  double t = res->fl_res - a;
  res->fl_err = (a - (res->fl_res - t)) + (b - t);
}
/* Two Product */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE void two_prod(const double a, const double b,
                                                  struct eft *res) {
  res->fl_res = a * b;
  res->fl_err = fma(a, b, -res->fl_res);
}
/* Two Sum Cmplx */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE void two_sum_cmplx(const thrust::complex<double> a,
                                                       const thrust::complex<double> b,
                                                       struct eft *eft_arr,
                                                       struct eft_cmplx_sum *res) {
  // struct eft real_eft, imag_eft;
  two_sum(a.real(), b.real(), eft_arr);
  two_sum(a.imag(), b.imag(), eft_arr + 1);
  res->fl_res = thrust::complex<double>(eft_arr[0].fl_res, eft_arr[1].fl_res);
  res->fl_err = thrust::complex<double>(eft_arr[0].fl_err, eft_arr[1].fl_err);
}
/* Two Product Cmplx */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE void two_prod_cmplx(const thrust::complex<double> a,
                                                        const thrust::complex<double> b,
                                                        struct eft *eft_arr,
                                                        struct eft_cmplx_prod *res) {
  // struct eft real_prod1, real_prod2, real_prod3, real_prod4, real_sum1, real_sum2;
  two_prod(a.real(), b.real(), eft_arr);
  two_prod(a.imag(), b.imag(), eft_arr + 1);
  two_prod(a.real(), b.imag(), eft_arr + 2);
  two_prod(a.imag(), b.real(), eft_arr + 3);
  two_sum(eft_arr[0].fl_res, -eft_arr[1].fl_res, eft_arr + 4);
  two_sum(eft_arr[2].fl_res, eft_arr[3].fl_res, eft_arr + 5);
  res->fl_res = thrust::complex<double>(eft_arr[4].fl_res, eft_arr[5].fl_res);
  res->fl_err1 = thrust::complex<double>(eft_arr[0].fl_err, eft_arr[2].fl_err);
  res->fl_err2 = thrust::complex<double>(-eft_arr[1].fl_err, eft_arr[3].fl_err);
  res->fl_err3 = thrust::complex<double>(eft_arr[4].fl_err, eft_arr[5].fl_err);
}
/* Error Free Array Extraction */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE double extract(double *p, double sigma) {
  struct eft res;
  for (int i = 0; i < 4; ++i) {
    // two_sum(sigma,p[i],&res);
    fast_two_sum(sigma, p[i], &res);
    sigma = res.fl_res;
    p[i] = res.fl_err;
  }
  return sigma;
}
/* Sum */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE double sum(const double *p) {
  double s = p[0];
  for (int i = 1; i < 4; ++i) {
    s += p[i];
  }
  return s;
}
/* Absolute Sum */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE double abs_sum(const double *p) {
  double s = fabs(p[0]);
  for (int i = 1; i < 4; ++i) {
    s += fabs(p[i]);
  }
  return s;
}
/* Fast Accurate Summation */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE double fast_acc_sum(double *p) {
  // variables
  int n = 0;
  double phi, sigma, sigma_new, T, t, t_new, tau, u;
// goto label
start:
  T = abs_sum(p) / fma(-4, EPS, 1);
  if (T <= ETA / EPS) {
    return sum(p);  // no rounding error
  }
  t = 0;
  do {
    sigma = (2 * T) / fma(-13, EPS, 1);  // 3 flops
    sigma_new = extract(p, sigma);       // 4(3) = 12 flops
    tau = sigma_new - sigma;             // 1 flop
    t_new = t;
    t = t_new + tau;  // 1 flop
    if (t == 0) {
      goto start;  // intermediate sum is zero, recursively apply fast_acc_sum to array of lower
                   // order parts
    }
    u = ufp(sigma);                                                   // 4 flops
    phi = ((48 * EPS) * u) / fma(-5, EPS, 1);                         // 4 flops
    T = fmin((fma(4, EPS, 1.5) * (4 * EPS)) * sigma, (8 * EPS) * u);  // 6 flops
    n += 1;
  } while ((fabs(t) < phi) && (4 * T > ETA / EPS));  // 2 flops
  tau = (t_new - t) + tau;
  // return
  return t + (tau + sum(p));
}
/* Fast Complex Accurate Summation */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE thrust::complex<double> fast_cmplx_acc_sum(
    thrust::complex<double> *p) {
  // variables
  double realp[4] = {p[0].real(), p[1].real(), p[2].real(), p[3].real()};
  double imagp[4] = {p[0].imag(), p[1].imag(), p[2].imag(), p[3].imag()};

  // return
  return thrust::complex<double>(fast_acc_sum(realp), fast_acc_sum(imagp));
}
/* Sort */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE void sort(double *p) {
  double max, temp;
  int ind, i, j;
  for (i = 0; i < 3; ++i) {
    max = fabs(p[i]);
    ind = i;
    for (j = i + 1; j < 4; ++j) {
      temp = fabs(p[j]);
      if (temp > max) {
        max = temp;
        ind = j;
      }
    }
    if (ind != i) {
      temp = p[i];
      p[i] = p[ind];
      p[ind] = temp;
    }
  }
}
/* Priest Summation */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE double priest_sum(double *p) {
  // sort p
  sort(p);
  // initialize
  double s = p[0], c = 0, y, u, t, v, z;
  for (int i = 1; i < 4; ++i) {
    y = c + p[i];
    u = p[i] - (y - c);
    t = y + s;
    v = y - (t - s);
    z = u + v;
    s = t + z;
    c = z - (s - t);
  }
  // return
  return s;
}
/* Priest Complex Summation */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE thrust::complex<double> priest_cmplx_sum(
    thrust::complex<double> *p) {
  // variables
  double realp[4] = {p[0].real(), p[1].real(), p[2].real(), p[3].real()};
  double imagp[4] = {p[0].imag(), p[1].imag(), p[2].imag(), p[3].imag()};
  // return
  return thrust::complex<double>(priest_sum(realp), priest_sum(imagp));
}
/* Horner Method with Double Real Arithmetic */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE void horner_dble(const double *poly, const double x,
                                                     const unsigned int deg, double *h) {
  // Horner's method
  *h = poly[deg];
  for (int i = deg - 1; i >= 0; --i) {
    *h = fma(*h, x, poly[i]);
  }
}
/* Reversal Horner Method with Double Real Arithmetic */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE void rhorner_dble(const double *poly, const double x,
                                                      const unsigned int deg, double *h) {
  // Reversal Horner's method
  *h = poly[0];
  for (int i = 1; i <= deg; ++i) {
    *h = fma(*h, x, poly[i]);
  }
}
/* Horner Method with Complex Arithmetic */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE void horner_cmplx(const thrust::complex<double> *poly,
                                                      const thrust::complex<double> x,
                                                      const unsigned int deg,
                                                      thrust::complex<double> *h,
                                                      thrust::complex<double> *hd) {
  // Horner's method
  *h = poly[deg];
  *hd = 0;
  for (int i = deg - 1; i >= 0; --i) {
    *hd = *hd * x + *h;
    *h = *h * x + poly[i];
  }
}
/* Reversal Horner Method with Complex Arithmetic */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE void rhorner_cmplx(const thrust::complex<double> *poly,
                                                       const thrust::complex<double> x,
                                                       const unsigned int deg,
                                                       thrust::complex<double> *h,
                                                       thrust::complex<double> *hd) {
  // Reversal Horner's method
  *h = poly[0];
  *hd = 0;
  for (int i = 1; i <= deg; ++i) {
    *hd = *hd * x + *h;
    *h = *h * x + poly[i];
  }
}
/* Horner Method */
EHRLICH_ABERTH_JAX_INLINE_OR_DEVICE void horner(const thrust::complex<double> *poly,
                                                const thrust::complex<double> x,
                                                const unsigned int deg,
                                                thrust::complex<double> *h) {
  // Horner's method
  *h = poly[deg];
  for (int i = deg - 1; i >= 0; --i) {
    *h = *h * x + poly[i];
  }
}

}  // namespace ehrlich_aberth_jax
#endif
