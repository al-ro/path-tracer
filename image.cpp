
#include "image.hpp"

#include <cuda_runtime_api.h>

#include "error.hpp"

// Host image
Image::Image(uint width, uint height, const std::vector<vec3>& data) : width{width}, height{height}, data{data} {}
Image::Image(uint width, uint height) : width{width}, height{height}, data{std::vector<vec3>{width * height}} {}

// Device image
GPUImage::GPUImage(uint width, uint height, vec3* data) : width{width}, height{height}, data{data} {}
GPUImage::GPUImage(uint width, uint height) : width{width}, height{height} {
  CHECK_CUDA_ERROR(cudaMalloc((void**)&data, width * height * sizeof(vec3)));
}
GPUImage::GPUImage(const Image& image) : width{image.width}, height{image.height} {
  CHECK_CUDA_ERROR(cudaMalloc((void**)&data, image.data.size() * sizeof(vec3)));
  CHECK_CUDA_ERROR(cudaMemcpy(data, image.data.data(), image.data.size() * sizeof(vec3), cudaMemcpyHostToDevice));
}
GPUImage::~GPUImage() {
  CHECK_CUDA_ERROR(cudaFree(data));
}
