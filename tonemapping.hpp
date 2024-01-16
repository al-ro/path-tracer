#pragma once

#include "dataTypes.hpp"

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
inline vec3 ACESFilm(vec3 x) {
  return clamp((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f, 1.0f);
}