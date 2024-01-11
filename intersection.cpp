#include "intersection.hpp"

#include <stack>

void intersectBVH(
    Ray& ray,
    const std::vector<BVHNode>& bvh,
    const std::vector<Triangle>& primitives,
    const uint nodeIdx,
    HitRecord& hitRecord,
    uint& count) {
  const BVHNode* node = &bvh[nodeIdx];
  std::stack<const BVHNode*> stack;
  vec2 barycentric{0};

  while (1) {
    // If leaf node, intersect with primitives
    if (node->count > 0) {
      for (uint i = 0; i < node->count; i++) {
        uint idx = node->leftFirst + i;
        float distance = intersect(ray, primitives[idx], barycentric);
        if (distance < ray.t) {
          ray.t = distance;
          hitRecord.barycentric = barycentric;
          hitRecord.hitIndex = idx;
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
    float dist1 = intersect(ray, child1->aabb);
    float dist2 = intersect(ray, child2->aabb);

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

uint intersectTLAS(Ray& ray,
                   const std::vector<BVHNode>& tlas,
                   const std::vector<Mesh>& meshes,
                   HitRecord& closestHit,
                   uint& count) {
  const BVHNode* node = &tlas[0];
  std::stack<const BVHNode*> stack{};
  uint meshIndex{UINT_MAX};
  HitRecord hitRecord = closestHit;
  float closestDist{FLT_MAX};

  while (1) {
    // If leaf node, intersect with meshes
    if (node->count > 0) {
      for (uint i = 0; i < node->count; i++) {
        uint idx = node->leftFirst + i;
        meshes[idx].intersect(ray, hitRecord, count);
        if (ray.t < closestDist) {
          closestDist = ray.t;
          closestHit = hitRecord;
          meshIndex = idx;
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
    const BVHNode* child1 = &tlas[node->leftFirst];
    const BVHNode* child2 = &tlas[node->leftFirst + 1];
    float dist1 = intersect(ray, child1->aabb);
    float dist2 = intersect(ray, child2->aabb);

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
  return meshIndex;
}
