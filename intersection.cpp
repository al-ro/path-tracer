#include "intersection.hpp"

#include <stack>

void intersectBVH(
    Ray& ray,
    const std::vector<BVHNode>& bvh,
    const std::vector<Triangle>& primitives,
    HitRecord& hitRecord,
    uint& count) {
  intersectBVH(ray, bvh.data(), primitives.data(), hitRecord, count);
}

uint intersectTLAS(Ray& ray,
                   const std::vector<BVHNode>& tlas,
                   const std::vector<Mesh>& meshes,
                   HitRecord& closestHit,
                   uint& count) {
  return intersectTLAS(ray, tlas.data(), meshes.data(), closestHit, count);
}
