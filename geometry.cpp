
#include "geometry.hpp"

#include <chrono>
#include <iostream>

#include "bvh.hpp"
#include "error.hpp"
#include "intersection.hpp"

Geometry::Geometry(std::vector<Triangle> primitives,
                   VertexAttributes attributes) : primitives{primitives},
                                                  indices{std::vector<uint>(primitives.size())},
                                                  faceNormals{std::vector<vec3>(primitives.size())},
                                                  attributes{attributes},
                                                  bvh{std::vector<BVHNode>(2 * primitives.size() - 1)} {
  generateIndices();
  generateBVH();
  generateNormals();

  aabbMin = bvh[0].aabb.min;
  aabbMax = bvh[0].aabb.max;

  corners[0] = vec3{aabbMin.x, aabbMin.y, aabbMin.z};
  corners[1] = vec3{aabbMin.x, aabbMax.y, aabbMin.z};
  corners[2] = vec3{aabbMin.x, aabbMin.y, aabbMax.z};
  corners[3] = vec3{aabbMin.x, aabbMax.y, aabbMax.z};
  corners[4] = vec3{aabbMax.x, aabbMin.y, aabbMin.z};
  corners[5] = vec3{aabbMax.x, aabbMax.y, aabbMin.z};
  corners[6] = vec3{aabbMax.x, aabbMin.y, aabbMax.z};
  corners[7] = vec3{aabbMax.x, aabbMax.y, aabbMax.z};

  centroid = bvh[0].aabb.min + 0.5f * (bvh[0].aabb.max - bvh[0].aabb.min);
}

void Geometry::generateBVH() {
  if (primitives.size() < 1) {
    std::cerr << "Can't generate BVH for empty primitive container" << std::endl;
    return;
  }

  uint nodesUsed{1};
  uint rootNodeIdx{0};

  /* Timer */ auto start{std::chrono::steady_clock::now()};

  // Build BLAS of triangles of the geometry
  buildBVH(bvh, primitives, indices, rootNodeIdx, nodesUsed);

  // BVH vector was created to store the maximum 2N-1 nodes of an N leaf binary tree
  // Resize it to actually used number of nodes to save memory space
  bvh.resize(nodesUsed);

  reorder(primitives, indices);

  if (attributes.normals.size() > 0) {
    reorder(attributes.normals, indices, 3u);
  }

  if (attributes.texCoords.size() > 0) {
    reorder(attributes.texCoords, indices, 3u);
  }

  /* Timer */ std::chrono::duration<double> duration = std::chrono::steady_clock::now() - start;
  /* Timer */ std::cout << "BVH build time: " << std::floor(duration.count() * 1e4f) / 1e4f << " s\n";
  /* Timer */ std::cout << "Nodes: " << nodesUsed << std::endl;
}

vec3 Geometry::getNormal(uint idx, vec2 barycentric) const {
  if (attributes.normals.size() == 0) {
    return faceNormals[idx];
  }
  vec3 v0 = attributes.normals[3u * idx];
  vec3 v1 = attributes.normals[3u * idx + 1u];
  vec3 v2 = attributes.normals[3u * idx + 2u];
  return barycentric.x * v1 + barycentric.y * v2 + (1.0f - (barycentric.x + barycentric.y)) * v0;
}

vec2 Geometry::getTexCoord(uint idx, vec2 barycentric) const {
  if (attributes.texCoords.size() <= idx) {
    return vec2{0};
  }
  vec2 v0 = attributes.texCoords[3u * idx];
  vec2 v1 = attributes.texCoords[3u * idx + 1u];
  vec2 v2 = attributes.texCoords[3u * idx + 2u];
  return barycentric.x * v1 + barycentric.y * v2 + (1.0f - (barycentric.x + barycentric.y)) * v0;
}

vec3 calculateNormal(const Triangle& triangle) {
  return normalize(cross(triangle.v0 - triangle.v1, triangle.v0 - triangle.v2));
}

void Geometry::generateIndices() {
  // Populate primitives indices sequentially [0...N)
  for (uint i = 0u; i < indices.size(); i++) {
    indices[i] = i;
  }
}

void Geometry::generateNormals() {
  for (uint i = 0; i < faceNormals.size(); i++) {
    faceNormals[i] = calculateNormal(primitives[i]);
  }
}

void Geometry::intersect(Ray& ray, HitRecord& hitRecord, uint& count) const {
  intersectBVH(ray, bvh, primitives, 0, hitRecord, count);
}

GPUGeometry::GPUGeometry(const Geometry& geometry) {
  CHECK_CUDA_ERROR(cudaMalloc((void**)&primitives, geometry.primitives.size() * sizeof(Triangle)));
  CHECK_CUDA_ERROR(cudaMemcpy(primitives, geometry.primitives.data(), geometry.primitives.size() * sizeof(Triangle), cudaMemcpyHostToDevice));

  CHECK_CUDA_ERROR(cudaMalloc((void**)&indices, geometry.indices.size() * sizeof(uint)));
  CHECK_CUDA_ERROR(cudaMemcpy(indices, geometry.indices.data(), geometry.indices.size() * sizeof(uint), cudaMemcpyHostToDevice));

  CHECK_CUDA_ERROR(cudaMalloc((void**)&bvh, geometry.bvh.size() * sizeof(BVHNode)));
  CHECK_CUDA_ERROR(cudaMemcpy(bvh, geometry.bvh.data(), geometry.bvh.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));

  if (geometry.attributes.normals.size() > 0.0) {
    hasNormals = true;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&vertexNormals, geometry.attributes.normals.size() * sizeof(vec3)));
    CHECK_CUDA_ERROR(cudaMemcpy(vertexNormals, geometry.attributes.normals.data(), geometry.attributes.normals.size() * sizeof(vec3), cudaMemcpyHostToDevice));

  } else {
    CHECK_CUDA_ERROR(cudaMalloc((void**)&faceNormals, geometry.faceNormals.size() * sizeof(vec3)));
    CHECK_CUDA_ERROR(cudaMemcpy(faceNormals, geometry.faceNormals.data(), geometry.faceNormals.size() * sizeof(vec3), cudaMemcpyHostToDevice));
  }

  if (geometry.attributes.texCoords.size() > 0.0) {
    hasTexCoords = true;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&texCoords, geometry.attributes.texCoords.size() * sizeof(vec2)));
    CHECK_CUDA_ERROR(cudaMemcpy(texCoords, geometry.attributes.texCoords.data(), geometry.attributes.texCoords.size() * sizeof(vec2), cudaMemcpyHostToDevice));
  }
}

GPUGeometry::~GPUGeometry() {
  CHECK_CUDA_ERROR(cudaFree(primitives));
  CHECK_CUDA_ERROR(cudaFree(indices));
  CHECK_CUDA_ERROR(cudaFree(bvh));

  if (hasNormals) {
    CHECK_CUDA_ERROR(cudaFree(vertexNormals));
  } else {
    CHECK_CUDA_ERROR(cudaFree(faceNormals));
  }

  if (hasTexCoords) {
    CHECK_CUDA_ERROR(cudaFree(texCoords));
  }
}