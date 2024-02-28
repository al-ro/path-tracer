#include "scene.hpp"

#include <chrono>

#include "bvh.hpp"
#include "error.hpp"
#include "intersection.hpp"

Scene::Scene(const std::vector<Mesh>& meshes) : meshes{meshes},
                                                tlas{std::vector<BVHNode>(2.0 * meshes.size() - 1)},
                                                indices{std::vector<uint>(meshes.size())} {
  generateIndices();
  generateTLAS();
}

void Scene::completeScene() {
  generateIndices();
  generateTLAS();
}

void Scene::generateTLAS() {
  tlas = std::vector<BVHNode>(2.0 * meshes.size() - 1);

  uint nodesUsed{1};
  uint rootNodeIdx{0};

  /* Timer */ auto start{std::chrono::steady_clock::now()};

  // Build TLAS of meshes in the scene
  buildBVH(tlas, meshes, indices, rootNodeIdx, nodesUsed);

  // TLAS vector was created to store the maximum 2N-1 nodes of an N leaf binary tree
  // Resize it to actually used number of nodes to save memory space
  tlas.resize(nodesUsed);

  reorder(meshes, indices);

  /* Timer */ std::chrono::duration<double> duration = std::chrono::steady_clock::now() - start;
  /* Timer */ std::cout << "\nTLAS build time: " << std::floor(duration.count() * 1e4f) / 1e4f << " s\n";
  /* Timer */ std::cout << "TLAS nodes: " << nodesUsed << std::endl;
}

void Scene::generateIndices() {
  if (indices.size() == 0) {
    indices = std::vector<uint>(meshes.size());
  }
  // Populate primitive indices sequentially [0...N)
  for (uint i = 0u; i < indices.size(); i++) {
    indices[i] = i;
  }
}

uint Scene::intersect(Ray& ray, HitRecord& hitRecord, uint& count) const {
  return intersectTLAS(ray, tlas, meshes, hitRecord, count);
}

GPUScene::GPUScene(const Scene& scene, const std::vector<GPUMesh>& gpuMeshes) {
  CHECK_CUDA_ERROR(cudaMalloc(&meshes, gpuMeshes.size() * sizeof(GPUMesh)));
  CHECK_CUDA_ERROR(cudaMemcpy(meshes, gpuMeshes.data(), gpuMeshes.size() * sizeof(GPUMesh), cudaMemcpyHostToDevice));

  CHECK_CUDA_ERROR(cudaMalloc(&tlas, scene.tlas.size() * sizeof(BVHNode)));
  CHECK_CUDA_ERROR(cudaMemcpy(tlas, scene.tlas.data(), scene.tlas.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));
}
GPUScene::~GPUScene() {
  CHECK_CUDA_ERROR(cudaFree(meshes));
  CHECK_CUDA_ERROR(cudaFree(tlas));
}
