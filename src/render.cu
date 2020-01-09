#include "render.hpp"
#include <spdlog/spdlog.h>
#include <cassert>

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)


struct rgba8_t {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};

__device__ uchar4 heat_lut(float x)
{
  assert(0 <= x && x <= 1);
  float x0 = 1.f / 4.f;
  float x1 = 2.f / 4.f;
  float x2 = 3.f / 4.f;

  if (x < x0)
  {
    auto g = static_cast<std::uint8_t>(x / x0 * 255);
    return {0, g, 255, 255};
  }
  else if (x < x1)
  {
    auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);
    return {0, 255, b, 255};
  }
  else if (x < x2)
  {
    auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);
    return {r, 255, 0, 255};
  }
  else if (x == 1)
  {
      return {0, 0, 0, 255};
  }
  else
  {
    auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);
    return {255, b, 0, 255};
  }
}



__device__ uchar4 palette(int x)
{
    float v = (float)x / 100.0f;
    return heat_lut(v);
}


__global__ void mandel_ker(char* buffer, int width, int height, size_t pitch)
{
    //float denum = width * width + height * height;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float x0 = (float)x / width * 3.5f - 2.5f;
    float y0 = (float)y / height * 2.0f - 1.0f;

    float x1 = 0.0f;
    float y1 = 0.0f;
    int it = 0;

    while (x1*x1 + y1*y1 < 2*2 && it < 100) {
        float x_temp = x1 * x1 - y1 * y1 + x0;
        y1 = 2 * x1 * y1 + y0;
        x1 = x_temp;
        it++;
    }

    uchar4*  lineptr = (uchar4*)(buffer + y * pitch);
    lineptr[x] = palette(it);
}



///
/// \param buffer Input buffer of type (uchar4 or uint32_t)
/// \param width Width of the image
/// \param height Height of the image
/// \param pitch Size of a line in bytes
/// \param max_iter Maximum number of iterations
__global__ void compute_iter(char* buffer, int width, int height, size_t pitch, int max_iter)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float x0 = (float)x / width * 3.5f - 2.5f;
    float y0 = (float)y / height * 2.0f - 1.0f;

    float x1 = 0.0f;
    float y1 = 0.0f;
    uint32_t it = 0;

    while (x1*x1 + y1*y1 < 2*2 && it < max_iter) {
        float x_temp = x1 * x1 - y1 * y1 + x0;
        y1 = 2 * x1 * y1 + y0;
        x1 = x_temp;
        it++;
    }
    uint32_t*  lineptr = (uint32_t*)(buffer + y * pitch);
    lineptr[x] = it;
}

/// This function is single thread for now!
///
/// \param buffer Input buffer of type (uchar4 or uint32_t)
/// \param width Width of the image
/// \param height Height of the image
/// \param pitch Size of a line in bytes
/// \param max_iter Maximum number of iterations
/// \param LUT Output look-up table
__global__ void compute_LUT(const char* buffer, int width, int height, size_t pitch, int max_iter, uchar4* LUT)
{
    uint32_t* histo = (uint32_t*)LUT;
    for (int y = 0; y < height; ++y) {
        const uint32_t*  lineptr = (uint32_t*)(buffer + y * pitch);
        for (int x = 0; x < width; ++x) {
            int index = lineptr[x];
            histo[index] += 1;
        }
    }
    int histo_sum_N = 0;
    for (int i = 0; i < max_iter; ++i){
        histo_sum_N += histo[i];
    }
    int histo_sum_K = 0;
    for (int k = 0; k < max_iter; ++k) {
        histo_sum_K += histo[k];
        LUT[k] = heat_lut((float)histo_sum_K / (float)histo_sum_N);
    }
    LUT[max_iter] = {0,0,0,255};
}

///
/// \param buffer Input buffer of type (uchar4 or uint32_t)
/// \param width Width of the image
/// \param height Height of the image
/// \param pitch Size of a line in bytes
/// \param max_iter Maximum number of iterations
__global__ void apply_LUT(char* buffer, int width, int height, size_t pitch, int max_iter, const uchar4* LUT){

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    uchar4*  lineptr = (uchar4*)(buffer + y * pitch);
    uint32_t pixel_iter = ((uint32_t*)lineptr)[x];
    lineptr[x] = LUT[pixel_iter];
}





void render(char* hostBuffer, int width, int height, std::ptrdiff_t stride, int n_iterations)
{
  cudaError_t rc = cudaSuccess;
  cudaError_t rc2 = cudaSuccess;

  // Allocate device memory
  char*  devBuffer;
  uchar4* LUT;
  size_t pitch;

  rc = cudaMallocPitch(&devBuffer, &pitch, width * sizeof(rgba8_t), height);
  rc2 = cudaMalloc(&LUT, (n_iterations + 1) * sizeof(uchar4));
  if (rc || rc2)
    abortError("Fail buffer allocation");
  rc2 = cudaMemset(LUT, 0, n_iterations+1);
  if (rc || rc2)
    abortError("Fail buffer allocation");
  // Run the kernel with blocks of size 64 x 64
  {
    int bsize = 32;
    int w     = std::ceil((float)width / bsize);
    int h     = std::ceil((float)height / bsize);

    spdlog::debug("running kernel of size ({},{})", w, h);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);
    compute_iter<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch, n_iterations);
    compute_LUT<<<1, 1>>>(devBuffer, width, height, pitch, n_iterations, LUT);
    //apply_LUT<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch, n_iterations, LUT);

    if (cudaPeekAtLastError())
      abortError("Computation Error");
  }

  // Copy back to main memory
  rc = cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width * sizeof(rgba8_t), height, cudaMemcpyDeviceToHost);
  if (rc)
    abortError("Unable to copy buffer back to memory");

  // Free
  rc = cudaFree(devBuffer);
  rc2 = cudaFree(LUT);
  if (rc || rc2)
    abortError("Unable to free memory");
}
