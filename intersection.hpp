#include "dataTypes.hpp"
#include "mesh.hpp"

inline void intersect(const Ray& ray, const Triangle& triangle, HitRecord& hitRecord);
inline float intersect(const Ray& ray, const AABB& aabb);

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
