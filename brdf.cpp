#include "brdf.h"

// Get orthonormal basis from surface normal
// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
void pixarONB(const vec3& n, vec3& b1, vec3& b2) {
  float sign_ = n.z >= 0.0f ? 1.0f : -1.0f;
  float a = -1.0f / (sign_ + n.z);
  float b = n.x * n.y * a;
  b1 = vec3(1.0f + sign_ * n.x * n.x * a, sign_ * b, -sign_ * n.x);
  b2 = vec3(b, sign_ + n.y * n.y * a, -n.y);
}

//--------------------- BRDF sampling ----------------------

// From tangent-space vector to world-space sample vector
vec3 rotateToNormal(const vec3& L, const vec3& N) {
  vec3 tangent;
  vec3 bitangent;

  pixarONB(N, tangent, bitangent);

  tangent = normalize(tangent);
  bitangent = normalize(bitangent);

  return normalize(tangent * L.x + bitangent * L.y + N * L.z);
}

// As the Lambertian reflection lobe is distributed around the surface normal, the resulting
// direction is the final sampling direction
vec3 importanceSampleCosine(const vec2& Xi, const vec3& N) {
  // Cosine sampling
  float cosTheta = sqrt(1.0f - Xi.x);
  float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
  float phi = Xi.y * 2.0f * M_PI;

  vec3 L = normalize(vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta));

  return rotateToNormal(L, N);
}

// Return a world-space halfway vector H around N which corresponds to the GGX normal
// distribution. Reflecting the view ray on H will give a light sample direction
vec3 importanceSampleGGX(const vec2& Xi, const vec3& N, float a) {
  // GGX importance sampling
  float cosTheta = sqrt((1.0f - Xi.x) / (1.0f + (a * a - 1.0f) * Xi.x));
  float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
  float phi = Xi.y * 2.0f * M_PI;

  vec3 L = normalize(vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta));

  return rotateToNormal(L, N);
}

// Trowbridge-Reitz AKA GGX
float distribution(float NdotH, float roughness) {
  float a2 = roughness * roughness;
  return a2 / (M_PI * pow(pow(NdotH, 2.0f) * (a2 - 1.0f) + 1.0f, 2.0f));
}

float geometry(float cosTheta, float k) {
  return (cosTheta) / (cosTheta * (1.0f - k) + k);
}

float smiths(float NdotV, float NdotL, float roughness) {
  float k = roughness / 2.0f;
  return geometry(NdotV, k) * geometry(NdotL, k);
}

// Fresnel-Schlick
vec3 fresnel(float cosTheta, vec3 F0) {
  return F0 + (vec3{1} - F0) * pow(1.0f - cosTheta, 5.0f);
}