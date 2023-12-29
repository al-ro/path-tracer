#include "gpuRender.hpp"

// The size of the stack for keeping track of BVH nodes to be tested
#define STACK_SIZE 32
#define INV2PI (1.0f / (2.0f * M_PI))
#define INVPI (1.0f / M_PI)

__device__ vec3 getEnvironment(GPUImage* environment, const vec3& direction) {
  uint u = environment->width * (atan2f(direction.z, direction.x) * INV2PI + 0.5f);
  uint v = environment->height * acosf(direction.y) * INVPI;
  uint idx = min(u + v * environment->width, (environment->width * environment->height) - 1);

  return 0.5f * (*environment)[idx];
}

__device__ vec3 getIllumination(int& bounces) {
  bounces--;
  if (bounces <= 0) {
    return vec3{0};
  }
  return getIllumination(bounces);
}

__global__ void render(Camera camera, GPUImage* image, GPUImage* environment) {
  vec2 fragCoord = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if (fragCoord.x >= image->width || fragCoord.y >= image->height) {
    return;
  }

  Ray ray{};
  vec2 resolution{image->width, image->height};

  ray.direction = rayDirection(resolution, camera.fieldOfView, fragCoord);
  ray.direction = normalize(viewMatrix(camera.position, camera.target, camera.up) * ray.direction);
  ray.invDirection = 1.0f / ray.direction;
  ray.t = FLT_MAX;
  (*image)[(uint)(fragCoord.y) * image->width + (uint)(fragCoord.x)] = getEnvironment(environment, ray.direction);
}