#include "kernels.hpp"

__global__ void getIllumination(Camera camera, GPUImage image) {
  vec2 fragCoord = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if (fragCoord.x >= image.width || fragCoord.y >= image.height) {
    return;
  }

  Ray ray{};
  vec2 resolution{image.width, image.height};

  ray.direction = rayDirection(resolution, camera.fieldOfView, fragCoord);
  ray.direction = normalize(viewMatrix(camera.position, camera.target, camera.up) * ray.direction);
  ray.invDirection = 1.0f / ray.direction;
  ray.t = FLT_MAX;
  image.data[(uint)(fragCoord.y) * image.width + (uint)(fragCoord.x)] = vec3{fragCoord / resolution, 0.0f};
}

__global__ void getIllumination(Camera camera, GPUImage image, vec3* data) {
  vec2 fragCoord = vec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if (fragCoord.x >= image.width || fragCoord.y >= image.height) {
    return;
  }

  Ray ray{};
  vec2 resolution{image.width, image.height};

  ray.direction = rayDirection(resolution, camera.fieldOfView, fragCoord);
  ray.direction = normalize(viewMatrix(camera.position, camera.target, camera.up) * ray.direction);
  ray.invDirection = 1.0f / ray.direction;
  ray.t = FLT_MAX;
  data[(uint)(fragCoord.y) * image.width + (uint)(fragCoord.x)] = vec3{fragCoord / resolution, 0.0f};
}

__global__ void getIllumination() {
  ivec2 fragCoord = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
}