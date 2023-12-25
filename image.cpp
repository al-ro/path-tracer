
#include "image.hpp"

#include <cuda_runtime_api.h>

Image::Image(uint width, uint height, const std::vector<vec3>& data) : width{width}, height{height}, data{data} {}
Image::Image(uint width, uint height) : width{width}, height{height}, data{std::vector<vec3>{width * height}} {}

GPUImage::GPUImage(uint width, uint height, vec3* data) : width{width}, height{height}, data{data} {}
GPUImage::GPUImage(uint width, uint height) : width{width}, height{height} {
  cudaMalloc((void**)data, width * height);
}
GPUImage::GPUImage(const Image& image) : width{image.width}, height{image.height} {
  cudaMalloc((void**)data, width * height);
  cudaMemcpy(data, image.data.data(), image.data.size() * sizeof(vec3), cudaMemcpyHostToDevice);
}
GPUImage::~GPUImage() {
  cudaFree(data);
}