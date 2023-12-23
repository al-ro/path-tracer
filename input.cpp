#include "input.hpp"

#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#include "lib/stl_reader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "lib/tiny_obj_loader.h"

std::vector<Triangle> loadSTL(std::string path) {
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

  std::cout << "STL triangle count: " << triangles.size() << std::endl;

  return triangles;
}

std::vector<Triangle> loadObj(std::string path,
                              std::vector<vec3>& normals,
                              std::vector<vec2>& texCoords) {
  tinyobj::ObjReaderConfig reader_config;
  reader_config.triangulate = true;

  tinyobj::ObjReader reader;

  if (!reader.ParseFromFile(path, reader_config)) {
    if (!reader.Error().empty()) {
      std::cerr << "TinyObjReader: " << reader.Error();
    }
    exit(1);
  }

  if (!reader.Warning().empty()) {
    std::cout << "TinyObjReader: " << reader.Warning();
  }

  std::vector<vec3> vertices{};

  auto& attrib = reader.GetAttrib();
  auto& shapes = reader.GetShapes();
  auto& materials = reader.GetMaterials();

  // Loop over shapes
  for (size_t s = 0; s < shapes.size(); s++) {
    // Loop over faces(polygon)
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

      // Loop over vertices in the face.
      for (size_t v = 0; v < fv; v++) {
        // access to vertex
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
        tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
        tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
        tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
        vertices.push_back(vec3{vx, vy, vz});

        // Check if `normal_index` is zero or positive. negative = no normal data
        if (idx.normal_index >= 0) {
          tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
          tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
          tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
          normals.push_back(normalize(vec3{nx, ny, nz}));
        }

        // Check if `texcoord_index` is zero or positive. negative = no texcoord data
        if (idx.texcoord_index >= 0) {
          tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
          tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
          texCoords.push_back(vec2{tx, 1.0 - ty});
        }

        // Optional: vertex colors
        // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
        // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
        // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
      }
      index_offset += 3;

      // per-face material
      shapes[s].mesh.material_ids[f];
    }
  }

  std::vector<Triangle> triangles{};
  for (int i = 0; i < vertices.size(); i += 3) {
    triangles.push_back(Triangle{vertices[i], vertices[i + 1], vertices[i + 2]});
  }
  for (auto& triangle : triangles) {
    triangle.centroid = (triangle.v0 + triangle.v1 + triangle.v2) / 3.0f;
  }

  std::cout << "OBJ triangle count: " << triangles.size() << std::endl;

  return triangles;
}

// Read in HDR image
Image loadEnvironmentImage(std::string path) {
  int width;
  int height;
  int channels;
  vec3* data = reinterpret_cast<vec3*>(stbi_loadf(path.c_str(), &width, &height, &channels, STBI_rgb));
  if (!data) {
    std::cerr << "Failed to load image " << path << std::endl;
  }

  Image image{static_cast<uint>(width), static_cast<uint>(height), std::vector<vec3>(data, data + width * height)};
  free(data);

  return image;
}

Image loadImage(std::string path) {
  int width;
  int height;
  int channels;
  stbi_uc* data = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb);
  if (!data) {
    std::cerr << "Failed to load image " << path << std::endl;
  }

  std::vector<vec3> rgbData(width * height);
  for (uint i = 0u; i < rgbData.size(); i++) {
    rgbData[i] = vec3(data[3 * i] / 255.0f, data[3 * i + 1] / 255.0f, data[3 * i + 2] / 255.0f);
  }
  free(data);
  return Image{static_cast<uint>(width), static_cast<uint>(height), rgbData};
}
