#pragma once

#include "dataTypes.hpp"
#include "mesh.hpp"

// Möller–Trumbore ray-triangle intersection
// Return distance to triangle and the barycentric coordinates if there is a hit
// FLT_MAX if no hit
__host__ __device__ inline float intersect(const Ray& ray, const Triangle& triangle, vec2& barycentric) {
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
__host__ __device__ inline float intersect(const Ray& ray, const vec3& bmin, const vec3& bmax) {
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

  if (tmax >= tmin && tmin < ray.t && tmax > 0) {
    return tmin;
  } else {
    return FLT_MAX;
  }
}

void intersectBVH(Ray& ray,
                  const std::vector<BVHNode>& bvh,
                  const std::vector<Triangle>& primitives,
                  const uint nodeIdx,
                  HitRecord& HitRecord,
                  uint& count);

uint intersectTLAS(Ray& ray,
                   const std::vector<BVHNode>& tlas,
                   const std::vector<Mesh>& meshes,
                   HitRecord& hitRecord,
                   uint& count);

__device__ inline void swap(float& a, float& b) {
  float temp = a;
  a = b;
  b = temp;
}

__device__ inline void swap(const BVHNode*& a, const BVHNode*& b) {
  const BVHNode* temp = a;
  a = b;
  b = temp;
}

// The size of the stack for keeping track of BVH nodes to be tested
#define STACK_SIZE 64

__device__ inline void intersectBVH(
    Ray& ray,
    const BVHNode* bvh,
    const Triangle* primitives,
    const uint* indices,
    const uint nodeIdx,
    HitRecord& hitRecord,
    uint& count) {
  const BVHNode* node = &bvh[nodeIdx];
  const BVHNode* stack[STACK_SIZE];
  // Start with root node on the stack
  int stackIdx{0};

  uint primitiveIndex{};
  vec2 barycentric{0};

  while (1) {
    // If leaf node, intersect with primitives
    if (node->count > 0) {
      for (uint i = 0; i < node->count; i++) {
        primitiveIndex = indices[node->leftFirst + i];
        float distance = intersect(ray, primitives[primitiveIndex], barycentric);
        if (distance < ray.t) {
          ray.t = distance;
          hitRecord.barycentric = barycentric;
          hitRecord.hitIndex = primitiveIndex;
        }
      }

      // If stack is empty, exit loop. Else grab next element on stack
      if (stackIdx <= 0) {
        break;
      } else {
        node = stack[--stackIdx];
      }
      // Skip to the start of the loop
      continue;
    }

    // Compare the distances to the two child nodes
    const BVHNode* child1 = &bvh[node->leftFirst];
    const BVHNode* child2 = &bvh[node->leftFirst + 1];
    float dist1 = intersect(ray, child1->aabb.min, child1->aabb.max);
    float dist2 = intersect(ray, child2->aabb.min, child2->aabb.max);

    // Consider closer one first
    if (dist1 > dist2) {
      swap(dist1, dist2);
      swap(child1, child2);
    }

    // If closer node is missed, the farther one is as well
    if (dist1 == FLT_MAX) {
      // Exit if stack empty or grab next element
      if (stackIdx <= 0) {
        break;
      } else {
        node = stack[--stackIdx];
      }
    } else {
      // If closer node is hit, consider it for the next loop
      node = child1;
      count++;

      // If the farther node is hit, place it on the stack
      if (dist2 != FLT_MAX) {
        count++;
        stack[stackIdx++] = child2;
      }
    }
  }
}

__device__ inline uint intersectTLAS(Ray& ray,
                                     const BVHNode* tlas,
                                     const GPUMesh* meshes,
                                     const uint* indices,
                                     HitRecord& closestHit,
                                     uint& count) {
  const BVHNode* node = &tlas[0];
  const BVHNode* stack[STACK_SIZE];
  // Start with root node on the stack
  int stackIdx{0};
  uint meshIndex{UINT_MAX};
  HitRecord hitRecord = closestHit;
  float closestDist{FLT_MAX};

  while (1) {
    // If leaf node, intersect with meshes
    if (node->count > 0) {
      for (uint i = 0; i < node->count; i++) {
        uint idx = indices[node->leftFirst + i];
        meshes[idx].intersect(ray, hitRecord, count);
        if (ray.t < closestDist) {
          closestDist = ray.t;
          closestHit = hitRecord;
          meshIndex = idx;
        }
      }

      // If stack is empty, exit loop. Else grab next element on stack
      if (stackIdx <= 0) {
        break;
      } else {
        node = stack[--stackIdx];
      }
      // Skip to the start of the loop
      continue;
    }

    // Compare the distances to the two child nodes
    const BVHNode* child1 = &tlas[node->leftFirst];
    const BVHNode* child2 = &tlas[node->leftFirst + 1];
    float dist1 = intersect(ray, child1->aabb.min, child1->aabb.max);
    float dist2 = intersect(ray, child2->aabb.min, child2->aabb.max);

    // Consider closer one first
    if (dist1 > dist2) {
      swap(dist1, dist2);
      swap(child1, child2);
    }

    // If closer node is missed, the farther one is as well
    if (dist1 == FLT_MAX) {
      // Exit if stack empty or grab next element
      if (stackIdx <= 0) {
        break;
      } else {
        node = stack[--stackIdx];
      }
    } else {
      // If closer node is hit, consider it for the next loop
      node = child1;
      count++;

      // If the farther node is hit, place it on the stack
      if (dist2 != FLT_MAX) {
        count++;
        stack[stackIdx++] = child2;
      }
    }
  }
  return meshIndex;
}