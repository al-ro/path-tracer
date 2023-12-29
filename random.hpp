#pragma once

#include <cuda_runtime_api.h>

#include <cstdlib>

#include "dataTypes.hpp"

const double inline INV_MAX_UINT32{1.0 / 4294967296.0};

// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
__device__ __host__ inline uint pcg_hash(uint& seed) {
  seed = seed * 747796405u + 2891336453u;
  uint state = seed;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

__device__ __host__ inline float getRandomFloat(uint& state) {
  return pcg_hash(state) * INV_MAX_UINT32;
}

__device__ __host__ inline vec2 getRandomVec2(uint& state) {
  return dvec2(pcg_hash(state), pcg_hash(state)) * INV_MAX_UINT32;
}

__device__ __host__ inline vec3 getRandomVec3(uint& state) {
  return dvec3(pcg_hash(state), pcg_hash(state), pcg_hash(state)) * INV_MAX_UINT32;
}