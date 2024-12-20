#pragma once

#include "dataTypes.hpp"
#include "mesh.hpp"

// Möller–Trumbore ray-triangle intersection
// Return distance to triangle and the barycentric coordinates if there is a hit
// FLT_MAX if no hit
template <typename Tri>
__host__ __device__ inline float intersect(const Ray& ray, const Tri& triangle, vec2& barycentric) {
  const vec3 edge1 = triangle.v1 - triangle.v0;
  const vec3 edge2 = triangle.v2 - triangle.v0;
  const vec3 h = cross(ray.direction, edge2);
  const float a = dot(edge1, h);

  if (a > -1e-4f && a < 1e-4f) {
    // Ray parallel to triangle
    return FLT_MAX;
  }

  const float f = 1.0f / a;
  const vec3 s = ray.origin - triangle.v0;
  const float u = f * dot(s, h);

  if (u < 0.0f || u > 1.0f) {
    return FLT_MAX;
  }

  const vec3 q = cross(s, edge1);
  const float v = f * dot(ray.direction, q);

  if (v < 0.0f || u + v > 1.0f) {
    return FLT_MAX;
  }

  float t = f * dot(edge2, q);

  if (t > 0.0f) {
    barycentric = vec2{u, v};
    return t;
  }

  return FLT_MAX;
}

// Ray-AABB intersection using the slab method which returns the distance to a hit with the AABB if
// it closer than the existing ray.t value. Returns FLT_MAX
__host__ __device__ inline float intersect(const Ray& ray, const AABB& aabb) {
  vec3 t1 = (aabb.min - ray.origin) * ray.invDirection;
  vec3 t2 = (aabb.max - ray.origin) * ray.invDirection;

  float tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
  float tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
  /*
    float tx1 = (bmin.x - ray.origin.x) * ray.invDirection.x;
    float tx2 = (bmax.x - ray.origin.x) * ray.invDirection.x;
    float tmin = min(tx1, tx2);
    float tmax = max(tx1, tx2);

    float ty1 = (bmin.y - ray.origin.y) * ray.invDirection.y;
    float ty2 = (bmax.y - ray.origin.y) * ray.invDirection.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));

    float tz1 = (bmin.z - ray.origin.z) * ray.invDirection.z;
    float tz2 = (bmax.z - ray.origin.z) * ray.invDirection.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));
  */
  if (tmax >= tmin && tmin < ray.t && tmax > 0) {
    return tmin;
  } else {
    return FLT_MAX;
  }
}

void intersectBVH(Ray& ray,
                  const std::vector<BVHNode>& bvh,
                  const std::vector<Triangle>& primitives,
                  HitRecord& HitRecord,
                  uint& count);

uint intersectTLAS(Ray& ray,
                   const std::vector<BVHNode>& tlas,
                   const std::vector<Mesh>& meshes,
                   HitRecord& hitRecord,
                   uint& count);

template <typename T>
__host__ __device__ inline void swap(T& a, T& b) {
  T temp = a;
  a = b;
  b = temp;
}

// The size of the stack for keeping track of BVH nodes to be tested
#define STACK_SIZE 32

template <typename TriangleClass>
__host__ __device__ inline void intersectBVH(Ray& ray,
                                             const BVHNode* bvh,
                                             const TriangleClass* primitives,
                                             HitRecord& hitRecord,
                                             uint& count) {
  BVHNode node = bvh[0u];

  uint stack[STACK_SIZE];
  uint8_t stackIdx{0u};

  while (1) {
    // If leaf node, intersect with primitives
    if (node.count > 0) {
      for (uint i = 0; i < node.count; i++) {
        uint idx = node.leftFirst + i;
        vec2 barycentric{0};
        float distance = intersect(ray, primitives[idx], barycentric);
        if (distance < ray.t) {
          ray.t = distance;
          hitRecord.barycentric = barycentric;
          hitRecord.hitIndex = idx;
        }
      }

      // If stack is empty, exit loop. Else grab next element on stack
      if (stackIdx == 0u) {
        break;
      }
      node = bvh[stack[--stackIdx]];
      continue;
    }

    // Compare the distances to the two child nodes
    uint idx1 = node.leftFirst;
    uint idx2 = idx1 + 1u;
    float dist1 = intersect(ray, bvh[idx1].aabb);
    float dist2 = intersect(ray, bvh[idx2].aabb);

    // Consider closer one first
    if (dist1 > dist2) {
      swap(dist1, dist2);
      swap(idx1, idx2);
    }

    // If closer node is missed, the farther one is as well
    if (dist1 == FLT_MAX) {
      // Exit if stack empty or grab next element
      if (stackIdx == 0u) {
        break;
      }
      node = bvh[stack[--stackIdx]];
      continue;
    }
    // If closer node is hit, consider it for the next loop
    node = bvh[idx1];
    count++;

    // If the farther node is hit, place it on the stack
    if (dist2 != FLT_MAX) {
      count++;
      stack[stackIdx++] = idx2;
    }
  }
}

template <typename MeshClass>
__host__ __device__ inline uint intersectTLAS(Ray& ray,
                                              const BVHNode* tlas,
                                              const MeshClass* meshes,
                                              HitRecord& closestHit,
                                              uint& count) {
  BVHNode node = tlas[0u];

  uint stack[STACK_SIZE];
  uint8_t stackIdx{0};
  uint meshIndex{UINT_MAX};
  HitRecord hitRecord = closestHit;
  float closestDist{FLT_MAX};

  while (1) {
    // If leaf node, intersect with meshes
    if (node.count > 0) {
      for (uint i = 0; i < node.count; i++) {
        uint idx = node.leftFirst + i;
        meshes[idx].intersect(ray, hitRecord, count);
        if (ray.t < closestDist) {
          closestDist = ray.t;
          closestHit = hitRecord;
          meshIndex = idx;
        }
      }

      // If stack is empty, exit loop. Else grab next element on stack
      if (stackIdx == 0u) {
        break;
      }
      node = tlas[stack[--stackIdx]];
      continue;
    }

    // Compare the distances to the two child nodes
    uint idx1 = node.leftFirst;
    uint idx2 = idx1 + 1u;
    float dist1 = intersect(ray, tlas[idx1].aabb);
    float dist2 = intersect(ray, tlas[idx2].aabb);

    // Consider closer one first
    if (dist1 > dist2) {
      swap(dist1, dist2);
      swap(idx1, idx2);
    }

    // If closer node is missed, the farther one is as well
    if (dist1 == FLT_MAX) {
      // Exit if stack empty or grab next element
      if (stackIdx == 0u) {
        break;
      }
      node = tlas[stack[--stackIdx]];
      continue;
    }

    // If closer node is hit, consider it for the next loop
    node = tlas[idx1];
    count++;

    // If the farther node is hit, place it on the stack
    if (dist2 != FLT_MAX) {
      count++;
      stack[stackIdx++] = idx2;
    }
  }
  return meshIndex;
}