#pragma once
#include <cuda_runtime_api.h>

#include "dataTypes.hpp"

// Generate default ray for a fragment based on its position, the image and the camera
__device__ __host__ vec3 rayDirection(const vec2& resolution, float fieldOfView, const vec2& fragCoord);

__device__ __host__ mat3 viewMatrix(vec3 camera, vec3 at, vec3 up);
