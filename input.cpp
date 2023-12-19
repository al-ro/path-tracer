#include "input.hpp"

#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#include "lib/stl_reader.h"

Image loadImage(std::string path);

// Read in .stl file and centre it. Generate triangle vertices and centroids
std::vector<Triangle> loadModel(std::string path) {
  stl_reader::StlMesh<float, uint> mesh(path);

  std::vector<Triangle> triangles{mesh.num_tris()};
  for (uint i = 0; i < triangles.size(); i++) {
    auto v0 = mesh.tri_corner_coords(i, 0);
    auto v1 = mesh.tri_corner_coords(i, 1);
    auto v2 = mesh.tri_corner_coords(i, 2);

    triangles[i].v0 = vec3{v0[0], v0[1], v0[2]};
    triangles[i].v1 = vec3{v1[0], v1[1], v1[2]};
    triangles[i].v2 = vec3{v2[0], v2[1], v2[2]};
  }

  for (auto& triangle : triangles) {
    triangle.centroid = (triangle.v0 + triangle.v1 + triangle.v2) / 3.0f;
  }

  return triangles;
}

// Read in HDR image
Image loadEnvironmentImage(std::string path) {
  int width;
  int height;
  int channels;
  vec3* data = reinterpret_cast<vec3*>(stbi_loadf(path.c_str(), &width, &height, &channels, STBI_rgb));

  return Image{static_cast<uint>(width), static_cast<uint>(height), std::vector<vec3>(data, data + width * height)};
}