#pragma once

#include "dataTypes.hpp"
#include "image.hpp"

class Material {
 public:
  // Albedo value or multiplier for albedo texture if provided
  vec3 albedo{1};
  // Metalness value or multiplier for metalness texture if provided
  float metalness{0};
  // Roughness value or multiplier for roughness texture if provided
  float roughness{0.01};
  // Emission value or multiplier for emission texture if provided
  vec3 emissive{0};

  // Multiplied with albedo
  Image albedoTexture{};
  // Multiplied with emissive
  Image emissiveTexture{};

  // Index of refraction for common dielectrics. Corresponds to F0 0.04
  const float IOR{1.5f};
  // Reflectance of the surface when looking straight at it along the negative normal
  const vec3 F0 = vec3(pow(IOR - 1.0f, 2.0f) / pow(IOR + 1.0f, 2.0f));

  Material() = default;
  Material(vec3 albedo, float metalness, float roughness, vec3 emissive = vec3(0));
  Material(const Material&) = default;
  Material(Material&&) = default;
  const vec3 getAlbedo(vec2 uv) const;
  const vec3 getEmissive(vec2 uv) const;
};