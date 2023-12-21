#include "dataTypes.hpp"
#include "mesh.hpp"

void intersect(const Ray& ray, const Triangle& triangle, HitRecord& hitRecord);
float intersect(const Ray& ray, const AABB& aabb);
float intersect(const Ray& ray, const vec3& aabbMin, const vec3& aabbMax);
void intersect(Ray& ray, const Mesh& mesh, HitRecord& hitRecord, uint& count);

void intersectBVH(Ray& ray,
                  const std::vector<BVHNode>& bvh,
                  const std::vector<Triangle>& primitives,
                  const std::vector<uint>& indices,
                  const uint nodeIdx,
                  HitRecord& HitRecord,
                  uint& count);

uint intersectTLAS(Ray& ray,
                   const std::vector<BVHNode>& tlas,
                   const std::vector<Mesh>& scene,
                   const std::vector<uint>& indices,
                   HitRecord& hitRecord,
                   uint& count);
