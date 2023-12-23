#pragma once

#include <cstdlib>

#include "dataTypes.hpp"

const double INV_MAX_UINT32{1.0 / 4294967296.0};

uint pcg_hash(uint& seed);

float getRandomFloat(uint& state);

vec2 getRandomVec2(uint& state);

vec3 getRandomVec3(uint& state);