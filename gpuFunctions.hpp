#include "geometry.hpp"
#include "intersection.hpp"
#include "mesh.hpp"

__device__ vec2 GPUGeometry::getTexCoord(uint idx, vec2 barycentric) const {
  if (hasTexCoords) {
    vec2 v0 = texCoords[3u * idx];
    vec2 v1 = texCoords[3u * idx + 1u];
    vec2 v2 = texCoords[3u * idx + 2u];
    return barycentric.x * v1 + barycentric.y * v2 + (1.0f - (barycentric.x + barycentric.y)) * v0;
  }

  return vec2{0};
}

__device__ vec3 GPUGeometry::getNormal(uint idx, vec2 barycentric) const {
  if (hasNormals) {
    vec3 v0 = vertexNormals[3u * idx];
    vec3 v1 = vertexNormals[3u * idx + 1u];
    vec3 v2 = vertexNormals[3u * idx + 2u];
    return barycentric.x * v1 + barycentric.y * v2 + (1.0f - (barycentric.x + barycentric.y)) * v0;
  }

  return faceNormals[idx];
}

// Find the distance to the closest intersection, the index of the primitive and the number of BVH tests.
__device__ void GPUGeometry::intersect(Ray& ray, HitRecord& hitRecord, uint& count) const {
  intersectBVH(ray, bvh, primitives, 0, hitRecord, count);
}

__device__ void GPUMesh::intersect(Ray& ray, HitRecord& hitRecord, uint& count) const {
  Ray transformedRay = ray;
  transformedRay.origin = invModelMatrix * vec4(ray.origin, 1.0f);
  // Not normalized to handle scale transform
  transformedRay.direction = invModelMatrix * vec4(ray.direction, 0.0f);
  transformedRay.invDirection = 1.0f / transformedRay.direction;

  geometry->intersect(transformedRay, hitRecord, count);
  ray.t = transformedRay.t;
}

//-------------------------------- Rotations --------------------------------

__device__ inline vec3 rotate(vec3 p, vec4 q) {
  return 2.0f * cross(vec3(q), p * q.w + cross(vec3(q), p)) + p;
}
__device__ inline vec3 rotateX(vec3 p, float angle) {
  return rotate(p, vec4(sin(angle / 2.0), 0.0, 0.0, cos(angle / 2.0)));
}
__device__ inline vec3 rotateY(vec3 p, float angle) {
  return rotate(p, vec4(0.0, sin(angle / 2.0), 0.0, cos(angle / 2.0)));
}
__device__ inline vec3 rotateZ(vec3 p, float angle) {
  return rotate(p, vec4(0.0, 0.0, sin(angle / 2.0), cos(angle / 2.0)));
}
