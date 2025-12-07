// main.cu
// Batched CUDA randomizer with cuRAND, streams and pinned memory.


#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "utils.h"


// Kernel to apply per-pixel brightness/contrast/hue parameters.
extern "C" __global__ void randomize_pixels_kernel(uchar3* d_pixels, float* d_brightness, float* d_contrast, float* d_hue, int numPixels) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= numPixels) return;
uchar3 px = d_pixels[idx];
float r = px.x / 255.0f;
float g = px.y / 255.0f;
float b = px.z / 255.0f;
float c = d_contrast[idx];
float br = d_brightness[idx];
r = (r - 0.5f) * c + 0.5f + br;
g = (g - 0.5f) * c + 0.5f + br;
b = (b - 0.5f) * c + 0.5f + br;
float hshift = d_hue[idx];
if (fabsf(hshift) > 1e-6f) {
float Y = 0.299f*r + 0.587f*g + 0.114f*b;
float I = 0.596f*r - 0.274f*g - 0.322f*b;
float Q = 0.211f*r - 0.523f*g + 0.312f*b;
float cosA = cosf(hshift);
float sinA = sinf(hshift);
float I2 = I * cosA - Q * sinA;
float Q2 = I * sinA + Q * cosA;
r = Y + 0.956f*I2 + 0.621f*Q2;
g = Y - 0.272f*I2 - 0.647f*Q2;
b = Y - 1.106f*I2 + 1.703f*Q2;
}
r = fminf(fmaxf(r, 0.0f), 1.0f);
g = fminf(fmaxf(g, 0.0f), 1.0f);
b = fminf(fmaxf(b, 0.0f), 1.0f);
uchar3 out;
out.x = (unsigned char)(r * 255.0f);
out.y = (unsigned char)(g * 255.0f);
out.z = (unsigned char)(b * 255.0f);
d_pixels[idx] = out;
}


// cuRAND kernel to fill per-pixel params (one RNG per pixel)
extern "C" __global__ void init_rng_and_fill(float* d_brightness, float* d_contrast, float* d_hue, int numPixels, unsigned long long seed) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= numPixels) return;
// simple XORSHIFT-like RNG using curand
// We'll use curand Philox via curandStatePhilox4_32_10_t for good parallel RNG
}


int main(int argc, char** argv) {
if (argc < 4) {
std::cout << "Usage: " << argv[0] << " <input_image> <num_copies> <batch_size>
";
return 0;
}
std::string filename = argv[1];
int numCopies = atoi(argv[2]);
int batchSize = atoi(argv[3]);


cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
if (img.empty()) { std::cerr << "Failed to open image" << std::endl; return 1; }
cudaFree(d_
