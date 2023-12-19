
#include "geometry.hpp"

#include <chrono>
#include <iostream>

#include "bvh.hpp"
#include "intersection.hpp"

Geometry::Geometry(std::vector<Triangle> primitives) : primitives{primitives},
                                                       indices{std::vector<uint>(primitives.size())},
                                                       normals{std::vector<vec3>(primitives.size())},
                                                       bvh{std::vector<BVHNode>(2 * primitives.size() - 1)} {
  generateIndices();
  generateNormals();
  generateBVH();

  center = bvh[0].aabbMin + 0.5f * (bvh[0].aabbMax - bvh[0].aabbMin);
}

void Geometry::generateBVH() {
  if (primitives.size() < 1) {
    std::cerr << "Can't generate BVH for empty primitive container" << std::endl;
    return;
  }

  uint nodesUsed{1};
  uint rootNodeIdx{0};

  auto start{std::chrono::steady_clock::now()};

  buildBVH(bvh, primitives, indices, rootNodeIdx, nodesUsed);

  // BVH vector was created to store the maximum 2N-1 nodes of an N leaf binary tree
  // Resize it to actually used number of nodes to save memory space
  bvh.resize(nodesUsed);

  std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - start;
  std::cout << "BVH build time: " << std::floor(elapsed_seconds.count() * 1e4f) / 1e4f << " s\n";
  std::cout << "Nodes: " << nodesUsed << std::endl;
}

vec3 getNormal(const Triangle& triangle) {
  return normalize(cross(triangle.v0 - triangle.v1, triangle.v0 - triangle.v2));
}

void Geometry::generateIndices() {
  // Populate primitives indices sequentially [0...N)
  for (uint i = 0u; i < indices.size(); i++) {
    indices[i] = i;
  }
}

void Geometry::generateNormals() {
  for (uint i = 0; i < normals.size(); i++) {
    normals[i] = getNormal(primitives[i]);
  }
}

void Geometry::intersect(Ray& ray, uint& index, uint& count) {
  return intersectBVH(ray, bvh, primitives, indices, 0, index, count);
}
