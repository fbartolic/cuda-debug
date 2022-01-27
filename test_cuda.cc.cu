// Compile with nvcc -o test_cuda ./lib/test_cuda.cc.cu  --expt-relaxed-constexpr -G
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "ehrlich_aberth.h"
#include "kernel_helpers.h"
#include "kernels.h"

namespace ehrlich_aberth_jax {

namespace {

// CUDA kernel
__global__ void ehrlich_aberth_kernel(std::int64_t size, std::int64_t deg,
                                      const thrust::complex<double> *poly,
                                      thrust::complex<double> *roots, double *alpha, bool *conv,
                                      point *points, point *hull) {
  const std::int64_t itmax = 50;

  // Compute roots
  std::int64_t i;
  // This is a "grid-stride loop" see http://alexminnaar.com/2019/08/02/grid-stride-loops.html
  for (std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += blockDim.x * gridDim.x) {
    i = idx * (deg + 1);
    ehrlich_aberth(poly + i, roots + i - idx, deg, itmax, alpha + i, conv + i - idx, points + i,
                   hull + i);
  }
}

// Function which calls the CUDA kernel
inline void apply_ehrlich_aberth(cudaStream_t stream, void **buffers, const char *opaque,
                                 std::size_t opaque_len) {
  const EhrlichAberthDescriptor &d =
      *UnpackDescriptor<EhrlichAberthDescriptor>(opaque, opaque_len);
  const std::int64_t size = d.size;
  const std::int64_t deg = d.deg;

  const thrust::complex<double> *poly =
      reinterpret_cast<const thrust::complex<double> *>(buffers[0]);
  thrust::complex<double> *roots = reinterpret_cast<thrust::complex<double> *>(buffers[1]);

  const int block_dim = 256;
  const int grid_dim = std::min<int>(1024, (size + block_dim - 1) / block_dim);

  // Preallocate memory for temporary arrays used within the kernel
  // allocating these arrays within the kernel with `new` results in a an illegal memory access
  // error for some reason I don't understand
  double *alpha;
  bool *conv;
  point *points;
  point *hull;

  cudaMalloc(&alpha, size * (deg + 1) * sizeof(double));
  cudaMalloc(&conv, size * deg * sizeof(bool));
  cudaMalloc(&points, size * (deg + 1) * sizeof(point));
  cudaMalloc(&hull, size * (deg + 1) * sizeof(point));

  ehrlich_aberth_kernel<<<grid_dim, block_dim, 0, stream>>>(size, deg, poly, roots, alpha, conv,
                                                            points, hull);

  // free memory
  cudaFree(alpha);
  cudaFree(conv);
  cudaFree(points);
  cudaFree(hull);

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}

}  // namespace

void gpu_ehrlich_aberth(cudaStream_t stream, void **buffers, const char *opaque,
                        std::size_t opaque_len) {
  apply_ehrlich_aberth(stream, buffers, opaque, opaque_len);
}

}  // namespace ehrlich_aberth_jax

/* Main Function */
int main() {
  const std::int64_t size = 1000000;
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

  // Duplicate the first 6 coefficients size times
  for (std::int64_t i = 6; i < size * (deg + 1); i++) {
    poly[i] = poly[i - 6];
  }

  // Allocate device memory for poly and roots
  thrust::complex<double> *d_poly;
  thrust::complex<double> *d_roots;

  cudaMalloc((void **)&d_poly, size * (deg + 1) * sizeof(thrust::complex<double>));
  cudaMalloc((void **)&d_roots, size * deg * sizeof(thrust::complex<double>));

  // Transfer data from host to device memory
  cudaMemcpy(d_poly, poly, size * (deg + 1) * sizeof(thrust::complex<double>),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_roots, roots, size * deg * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);

  // Initialize a EhrlichAberthDescriptor  struct
  ehrlich_aberth_jax::EhrlichAberthDescriptor d = {size, deg};

  // Initialize void** buffers
  void *buffers[] = {d_poly, d_roots};

  // Run the root solver
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  auto t1 = high_resolution_clock::now();
  ehrlich_aberth_jax::gpu_ehrlich_aberth(0, buffers, reinterpret_cast<const char *>(&d),
                                         sizeof(d));
  auto t2 = high_resolution_clock::now();

  // Copy roots from device memory to host memory
  cudaMemcpy(roots, d_roots, size * deg * sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost);
  cudaMemcpy(poly, d_poly, size * (deg + 1) * sizeof(thrust::complex<double>),
             cudaMemcpyDeviceToHost);

  // Print the last 5 roots
  for (std::int64_t i = size - 5; i < size; i++) {
    std::cout << "Root " << i << ": " << roots[i] << std::endl;
  }

  // Free memory
  cudaFree(d_poly);
  cudaFree(d_roots);

  delete[] poly;
  delete[] roots;

  ///* Getting number of milliseconds as a double. */
  duration<double, std::milli> ms_double = t2 - t1;
  std::cout << ms_double.count() << "ms\n";

  // return
  return 0;
}
