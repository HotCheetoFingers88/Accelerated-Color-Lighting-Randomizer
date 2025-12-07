#pragma once
#include <cuda_runtime.h>
#include <iostream>


#define CUDA_CHECK(call) \
do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
std::cerr << "CUDA error: " << cudaGetErrorString(err) << " (" << err << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
exit(1); \
} \
} while (0)


#define CURAND_CHECK(call) \
do { \
curandStatus_t s = call; \
if (s != CURAND_STATUS_SUCCESS) { \
std::cerr << "cuRAND error: " << s << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
exit(1); \
} \
} while (0)
