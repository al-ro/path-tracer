#include "material.hpp"

Material::Material(vec3 albedo,
                   float metalness,
                   float roughness,
                   vec3 emissive) : albedo{albedo},
                                    metalness{metalness},
                                    roughness{roughness},
                                    emissive{emissive} {}

const vec3 Material::getAlbedo(vec2 uv) const {
  if (albedoTexture.width > 0.0) {
    return albedoTexture(uv);
  }
  return albedo;
}

const vec3 Material::getEmissive(vec2 uv) const {
  if (emissiveTexture.width > 0.0) {
    return emissiveTexture(uv);
  }
  return emissive;
}