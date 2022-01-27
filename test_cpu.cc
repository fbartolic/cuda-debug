// Compile with g++ -o test_cpu test_cpu.cc -Iextern/thrust-1.15.0  -Iextern/cub-1.15.0
// -I/usr/local/cuda-11.2/include/
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
#include "ehrlich_aberth.h"
#include "kernel_helpers.h"
#include "kernels.h"

namespace ehrlich_aberth_jax {

void cpu_ehrlich_aberth(void *out, const void **in) {
  // Parse the inputs
  const std::int64_t size =
      *reinterpret_cast<const std::int64_t *>(in[0]);  // number of polynomials (size of problem)
  const std::int64_t deg =
      *reinterpret_cast<const std::int64_t *>(in[1]);  // degree of polynomials

  const std::int64_t itmax = 50;

  // Flattened polynomial coefficients, shape (deg + 1)*size
  const thrust::complex<double> *poly = reinterpret_cast<const thrust::complex<double> *>(in[2]);

  // Output roots, shape deg*size
  thrust::complex<double> *roots = reinterpret_cast<thrust::complex<double> *>(out);

  // Allocate memory for temporary arrays
  double *alpha = new double[(deg + 1)];
  bool *conv = new bool[deg];
  point *points = new point[deg + 1];
  point *hull = new point[deg + 1];

  // Compute roots
  std::int64_t i;
  for (std::int64_t idx = 0; idx < size; ++idx) {
    i = idx * (deg + 1);
    ehrlich_aberth_jax::ehrlich_aberth(poly + i, roots + i - idx, deg, itmax, alpha, conv, points,
                                       hull);
  }

  // Free memory
  delete[] alpha;
  delete[] conv;
  delete[] points;
  delete[] hull;
}

}  // namespace ehrlich_aberth_jax

/* Main Function */
int main() {
  const std::int64_t size = 20000;
  const std::int64_t deg = 5;

  // initialize polynomial and storage for roots
  thrust::complex<double> *poly = new thrust::complex<double>[size * (deg + 1)];
  thrust::complex<double> *roots = new thrust::complex<double>[size * deg];

  poly[0] = thrust::complex<double>(-0.09681953, -0.00090856);
  poly[1] = thrust::complex<double>(2.01207803, -0.0004709);
  poly[2] = thrust::complex<double>(-1.12402336, 0.00979704);
  poly[3] = thrust::complex<double>(-1.54998502, -0.01104674);
  poly[4] = thrust::complex<double>(1.1198831, 0.00012531);
  poly[5] = thrust::complex<double>(-0.00506219, 0.00697788);

  // Copy the first 6 elements of poly to the rest of the allocated array
  for (std::int64_t i = 6; i < size * (deg + 1); i++) {
    poly[i] = poly[i - 6];
  }

  // Store input data in void**
  const void *in[] = {&size, &deg, poly};

  // Run the root solver
  ehrlich_aberth_jax::cpu_ehrlich_aberth(roots, in);

  // Print the last 5 roots
  for (std::int64_t i = size - 5; i < size; i++) {
    std::cout << "Root " << i << ": " << roots[i] << std::endl;
  }

  // Free memory
  delete[] poly;
  delete[] roots;

  // return
  return 0;
}