#include "dataTypes.h"

vec3 importanceSampleCosine(const vec2& Xi, const vec3& N);
vec3 importanceSampleGGX(const vec2& Xi, const vec3& N, float a);
float smiths(float NdotV, float NdotL, float roughness);
vec3 fresnel(float cosTheta, vec3 F0);