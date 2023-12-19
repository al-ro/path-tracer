#include "dataTypes.hpp"
#pragma once

class Material {
 public:
  vec3 albedo{};
  float metalness{0};
  float roughness{0.01};
  float IOR{1.55};
};