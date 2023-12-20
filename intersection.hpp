#include "dataTypes.hpp"
#include "mesh.hpp"

float intersect(const Ray& ray, const Triangle& triangle);
float intersect(const Ray& ray, const AABB& aabb);
float intersect(const Ray& ray, const vec3& aabbMin, const vec3& aabbMax);
HitRecord intersect(Ray& ray, const Mesh& mesh, uint& count);

void intersectBVH(Ray& ray,
                  const std::vector<BVHNode>& bvh,
                  const std::vector<Triangle>& primitives,
                  const std::vector<uint>& indices,
                  const uint nodeIdx,
                  uint& hitIndex,
                  uint& count);

HitRecord intersectTLAS(const Ray& ray,
                        const std::vector<TLASNode>& tlas,
                        const std::vector<Mesh>& scene,
                        const std::vector<uint>& indices,
                        uint& count);
