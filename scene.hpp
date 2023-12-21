#include <iostream>

#include "dataTypes.hpp"
#include "intersection.hpp"
#include "mesh.hpp"

class Scene {
 public:
  std::vector<Mesh> meshes;
  std::vector<uint> indices;
  std::vector<BVHNode> tlas;

  Scene() = default;
  Scene(std::vector<Mesh>&& meshes);
  Scene(const Scene& scene) = delete;
  ~Scene() = default;

  void generateIndices();
  void generateTLAS();

  void add(const Mesh& mesh) = delete;
  void remove(const Mesh& mesh) = delete;

  uint intersect(Ray& ray, HitRecord& hitRecord, uint& count) const;
};