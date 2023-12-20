#include "scene.hpp"

#include <chrono>

#include "tlas.hpp"

Scene::Scene(std::vector<Mesh>&& meshes_) : meshes{std::move(meshes_)},
                                            tlas{std::vector<TLASNode>(2.0 * meshes.size() - 1)},
                                            indices{std::vector<uint>(meshes.size())} {
  generateIndices();
  generateTLAS();
}

void Scene::generateTLAS() {
  uint nodesUsed{1};
  uint rootNodeIdx{0};

  auto start{std::chrono::steady_clock::now()};

  buildTLAS(tlas, meshes, indices, rootNodeIdx, nodesUsed);

  // TLAS vector was created to store the maximum 2N-1 nodes of an N leaf binary tree
  // Resize it to actually used number of nodes to save memory space
  tlas.resize(nodesUsed);

  std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - start;
  std::cout << "TLAS build time: " << std::floor(elapsed_seconds.count() * 1e4f) / 1e4f << " s\n";
  std::cout << "TLAS nodes: " << nodesUsed << std::endl;
}

void Scene::generateIndices() {
  // Populate primitives indices sequentially [0...N)
  for (uint i = 0u; i < indices.size(); i++) {
    indices[i] = i;
  }
}

HitRecord Scene::intersect(Ray& ray, uint& count) const {
  return intersectTLAS(ray, tlas, meshes, indices, count);
}