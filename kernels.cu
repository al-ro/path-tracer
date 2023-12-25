#include "camera.hpp"
#include "dataTypes.hpp"

__global__ void getIllumination(Camera camera) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  Ray ray{};

  vec2 fragCoord{0};
  vec2 resolution{1};

  ray.direction = rayDirection(resolution, camera.fieldOfView, fragCoord);
  ray.direction = normalize(viewMatrix(camera.position, camera.target, camera.up) * ray.direction);
  ray.invDirection = 1.0f / ray.direction;
  ray.t = FLT_MAX;
}