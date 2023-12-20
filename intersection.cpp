#include "intersection.hpp"

#include <stack>

// Möller–Trumbore
float intersect(const Ray& ray, const Triangle& triangle) {
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
  return t > 0.0f ? t : FLT_MAX;
}

// Slab method without division
float intersect(const Ray& ray, const vec3& bmin, const vec3& bmax) {
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

void intersectBVH(
    Ray& ray,
    const std::vector<BVHNode>& bvh,
    const std::vector<Triangle>& primitives,
    const std::vector<uint>& indices,
    const uint nodeIdx,
    uint& hitIndex,
    uint& count) {
  const BVHNode* node = &bvh[nodeIdx];
  std::stack<const BVHNode*> stack;
  uint idx{};

  while (1) {
    // If leaf node, intersect with primitives
    if (node->count > 0) {
      for (uint i = 0; i < node->count; i++) {
        idx = indices[node->leftFirst + i];
        float distance = intersect(ray, primitives[idx]);
        if (distance < ray.t) {
          ray.t = distance;
          hitIndex = idx;
        }
      }

      // If stack is empty, exit loop. Else grab next element on stack
      if (stack.empty()) {
        break;
      } else {
        node = stack.top();
        stack.pop();
      }
      // Skip to the start of the loop
      continue;
    }

    // Compare the distances to the two child nodes
    const BVHNode* child1 = &bvh[node->leftFirst];
    const BVHNode* child2 = &bvh[node->leftFirst + 1];
    float dist1 = intersect(ray, child1->aabbMin, child1->aabbMax);
    float dist2 = intersect(ray, child2->aabbMin, child2->aabbMax);

    // Consider closer one first
    if (dist1 > dist2) {
      std::swap(dist1, dist2);
      std::swap(child1, child2);
    }

    // If closer node is missed, the farther one is as well
    if (dist1 == FLT_MAX) {
      // Exit if stack empty or grab next element
      if (stack.empty()) {
        break;
      } else {
        node = stack.top();
        stack.pop();
      }
    } else {
      // If closer node is hit, consider it for the next loop
      node = child1;
      count++;

      // If the farther node is hit, place it on the stack
      if (dist2 != FLT_MAX) {
        count++;
        stack.push(child2);
      }
    }
  }
}

HitRecord intersectTLAS(const Ray& ray,
                        const std::vector<TLASNode>& tlas,
                        const std::vector<Mesh>& scene,
                        const std::vector<uint>& indices,
                        uint& count) {
  const TLASNode* node = &tlas[0];
  std::stack<const TLASNode*> stack{};
  HitRecord closestHit{};
  uint idx{};

  while (1) {
    if (node->count > 0u) {
      for (uint i = 0; i < node->count; i++) {
        idx = indices[node->leftChild + i];
        HitRecord hit = scene[idx].intersect(ray, count);
        if (hit.dist < closestHit.dist) {
          closestHit = hit;
          closestHit.hitIndex = idx;
        }
      }
      // If stack is empty, exit loop. Else grab next element on stack
      if (stack.empty()) {
        break;
      } else {
        node = stack.top();
        stack.pop();
      }
      // Skip to the start of the loop
      continue;
    }

    // Compare the distances to the two child nodes
    const TLASNode* child1 = &tlas[node->leftChild];
    const TLASNode* child2 = &tlas[node->leftChild + 1];
    float dist1 = intersect(ray, child1->aabbMin, child1->aabbMax);
    float dist2 = intersect(ray, child2->aabbMin, child2->aabbMax);

    // Consider closer one first
    if (dist1 > dist2) {
      std::swap(dist1, dist2);
      std::swap(child1, child2);
    }

    // If closer node is missed, the farther one is as well
    if (dist1 == FLT_MAX) {
      // Exit if stack empty or grab next element
      if (stack.empty()) {
        break;
      } else {
        node = stack.top();
        stack.pop();
      }
    } else {
      // If closer node is hit, consider it for the next loop
      node = child1;
      count++;

      // If the farther node is hit, place it on the stack
      if (dist2 != FLT_MAX) {
        count++;
        stack.push(child2);
      }
    }
  }
  return closestHit;
}