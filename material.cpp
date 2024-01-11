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

GPUMaterial::GPUMaterial(vec3 albedo,
                         float metalness,
                         float roughness,
                         vec3 emissive) : albedo{albedo},
                                          metalness{metalness},
                                          roughness{roughness},
                                          emissive{emissive} {}

GPUMaterial::GPUMaterial(const Material& material) : albedo{material.albedo},
                                                     metalness{material.metalness},
                                                     roughness{material.roughness},
                                                     emissive{material.emissive} {
  if (material.albedoTexture.width > 0u) {
    GPUImage gpuAlbedoTexture{material.albedoTexture};
    CHECK_CUDA_ERROR(cudaMalloc(&albedoTexture, sizeof(GPUImage)));
    CHECK_CUDA_ERROR(cudaMemcpy(albedoTexture, &gpuAlbedoTexture, sizeof(GPUImage), cudaMemcpyHostToDevice));
  }

  if (material.emissiveTexture.width > 0u) {
    GPUImage gpuEmissiveTexture{material.emissiveTexture};
    CHECK_CUDA_ERROR(cudaMalloc(&emissiveTexture, sizeof(GPUImage)));
    CHECK_CUDA_ERROR(cudaMemcpy(emissiveTexture, &gpuEmissiveTexture, sizeof(GPUImage), cudaMemcpyHostToDevice));
  }
}

GPUMaterial::~GPUMaterial() {
  if (albedoTexture != nullptr) {
    CHECK_CUDA_ERROR(cudaFree(albedoTexture));
  }
  if (emissiveTexture != nullptr) {
    CHECK_CUDA_ERROR(cudaFree(emissiveTexture));
  }
}