#include <cstdlib>

#include "structs.h"

const double INV_MAX_UINT32{1.0 / 4294967296.0};

// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
uint pcg_hash(uint& seed) {
  seed = seed * 747796405u + 2891336453u;
  uint state = seed;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

vec2 getRandomXi(uint& state) {
  return dvec2(pcg_hash(state), pcg_hash(state)) * INV_MAX_UINT32;
}

vec3 getRandomPoint(uint& state) {
  return dvec3(pcg_hash(state), pcg_hash(state), pcg_hash(state)) * INV_MAX_UINT32;
}