#include "intersection.h"

// Möller–Trumbore
float intersect(const Ray& ray, const Triangle& triangle) {
  const vec3 edge1 = triangle.v1 - triangle.v0;
  const vec3 edge2 = triangle.v2 - triangle.v0;
  const vec3 h = cross(ray.direction, edge2);
  const float a = dot(edge1, h);

  if (a > -1e-4f && a < 1e-4f) {
    // Ray parallel to triangle
    return -1.0f;
  }

  const float f = 1.0f / a;
  const vec3 s = ray.origin - triangle.v0;
  const float u = f * dot(s, h);

  if (u < 0.0f || u > 1.0f) {
    return -1.0f;
  }

  const vec3 q = cross(s, edge1);
  const float v = f * dot(ray.direction, q);

  if (v < 0.0f || u + v > 1.0f) {
    return -1.0f;
  }

  return f * dot(edge2, q);
}