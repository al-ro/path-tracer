#include "input.hpp"

#include "lib/stl_reader.h"

std::vector<vec3> loadImage(std::string path, vec2& resolution);

// Read in .stl file and centre it. Generate triangle vertices and centroids
std::vector<Triangle> loadModel(std::string path) {
  stl_reader::StlMesh<float, uint> mesh(path);
  std::vector<Triangle> scene{mesh.num_tris()};
  for (uint i = 0; i < scene.size(); i++) {
    scene[i].v0 = rotateX(vec3{mesh.tri_corner_coords(i, 0)[0], mesh.tri_corner_coords(i, 0)[1], mesh.tri_corner_coords(i, 0)[2]}, -0.5f * 3.1415926f);
    scene[i].v1 = rotateX(vec3{mesh.tri_corner_coords(i, 1)[0], mesh.tri_corner_coords(i, 1)[1], mesh.tri_corner_coords(i, 1)[2]}, -0.5f * 3.1415926f);
    scene[i].v2 = rotateX(vec3{mesh.tri_corner_coords(i, 2)[0], mesh.tri_corner_coords(i, 2)[1], mesh.tri_corner_coords(i, 2)[2]}, -0.5f * 3.1415926f);
  }
  vec3 aabbMin = vec3(FLT_MAX);
  vec3 aabbMax = vec3(-FLT_MAX);

  for (auto& triangle : scene) {
    aabbMin = min(aabbMin, triangle.v0);
    aabbMin = min(aabbMin, triangle.v1);
    aabbMin = min(aabbMin, triangle.v2);
    aabbMax = max(aabbMax, triangle.v0);
    aabbMax = max(aabbMax, triangle.v1);
    aabbMax = max(aabbMax, triangle.v2);
  }

  vec3 centre = aabbMin + 0.5f * (aabbMax - aabbMin);

  for (auto& triangle : scene) {
    triangle.v0 -= centre;
    triangle.v1 -= centre;
    triangle.v2 -= centre;
  }

  for (auto& triangle : scene) {
    triangle.centroid = (triangle.v0 + triangle.v1 + triangle.v2) / 3.0f;
  }

  return scene;
}