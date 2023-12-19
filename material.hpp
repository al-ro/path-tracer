#include "dataTypes.hpp"
#pragma once

class Material {
 public:
  vec3 albedo{};
  float metalness{0};
  float roughness{0.01};

  // Index of refraction for common dielectrics. Corresponds to F0 0.04
  const float IOR{1.5f};

  // Reflectance of the surface when looking straight at it along the negative normal
  const vec3 F0 = vec3(pow(IOR - 1.0f, 2.0f) / pow(IOR + 1.0f, 2.0f));
};