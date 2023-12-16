#include "input.hpp"

#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#include "lib/stl_reader.h"

Image loadImage(std::string path);

// Read in .stl file and centre it. Generate triangle vertices and centroids
std::vector<Triangle> loadModel(std::string path) {
  stl_reader::StlMesh<float, uint> mesh(path);

  std::vector<Triangle> model{mesh.num_tris()};
  for (uint i = 0; i < model.size(); i++) {
    auto v0 = mesh.tri_corner_coords(i, 0);
    auto v1 = mesh.tri_corner_coords(i, 1);
    auto v2 = mesh.tri_corner_coords(i, 2);

    float halfPi = M_PI_2;

    model[i].v0 = rotateX(vec3{v0[0], v0[1], v0[2]}, -halfPi);
    model[i].v1 = rotateX(vec3{v1[0], v1[1], v1[2]}, -halfPi);
    model[i].v2 = rotateX(vec3{v2[0], v2[1], v2[2]}, -halfPi);
  }

  vec3 aabbMin = vec3(FLT_MAX);
  vec3 aabbMax = vec3(-FLT_MAX);

  for (auto& triangle : model) {
    aabbMin = min(aabbMin, triangle.v0);
    aabbMin = min(aabbMin, triangle.v1);
    aabbMin = min(aabbMin, triangle.v2);
    aabbMax = max(aabbMax, triangle.v0);
    aabbMax = max(aabbMax, triangle.v1);
    aabbMax = max(aabbMax, triangle.v2);
  }

  vec3 centre = aabbMin + 0.5f * (aabbMax - aabbMin);

  for (auto& triangle : model) {
    triangle.v0 -= centre;
    triangle.v1 -= centre;
    triangle.v2 -= centre;
  }

  for (auto& triangle : model) {
    triangle.centroid = (triangle.v0 + triangle.v1 + triangle.v2) / 3.0f;
  }

  return model;
}

// Read in HDR image
Image loadEnvironmentImage(std::string path) {
  int width;
  int height;
  int channels;
  vec3* data = reinterpret_cast<vec3*>(stbi_loadf(path.c_str(), &width, &height, &channels, STBI_rgb));

  return Image(width, height, std::vector<vec3>(data, data + width * height));
}