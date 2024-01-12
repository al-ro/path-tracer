#include "material.hpp"

#include "error.hpp"

Material::Material(vec3 albedo,
                   float metalness,
                   float roughness,
                   vec3 emissive) : albedo{albedo},
                                    metalness{metalness},
                                    roughness{roughness},
                                    emissive{emissive} {}

const vec3 Material::getAlbedo(vec2 uv) const {
  if (albedoTexture.width > 0.0) {
    return albedo * albedoTexture(uv);
  }
  return albedo;
}

const vec3 Material::getEmissive(vec2 uv) const {
  if (emissiveTexture.width > 0.0) {
    return emissive * emissiveTexture(uv);
  }
  return emissive;
}

GPUMaterial::GPUMaterial(const Material& material) : albedo{material.albedo},
                                                     metalness{material.metalness},
                                                     roughness{material.roughness},
                                                     emissive{material.emissive},
                                                     albedoTexture{material.albedoTexture},
                                                     emissiveTexture{material.emissiveTexture} {
  if (albedoTexture.width > 0u) {
    CHECK_CUDA_ERROR(cudaMalloc(&albedoTexturePtr, sizeof(GPUImage)));
    CHECK_CUDA_ERROR(cudaMemcpy(albedoTexturePtr, &albedoTexture, sizeof(GPUImage), cudaMemcpyHostToDevice));
  }

  if (emissiveTexture.width > 0u) {
    CHECK_CUDA_ERROR(cudaMalloc(&emissiveTexturePtr, sizeof(GPUImage)));
    CHECK_CUDA_ERROR(cudaMemcpy(emissiveTexturePtr, &emissiveTexture, sizeof(GPUImage), cudaMemcpyHostToDevice));
  }
}

GPUMaterial::~GPUMaterial() {
  if (albedoTexturePtr != nullptr) {
    CHECK_CUDA_ERROR(cudaFree(albedoTexturePtr));
  }
  if (emissiveTexturePtr != nullptr) {
    CHECK_CUDA_ERROR(cudaFree(emissiveTexturePtr));
  }
}