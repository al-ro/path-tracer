#include "tlas.hpp"

#include <iostream>

//-------------------------- Build TLAS ---------------------------

void updateNodeBounds(TLASNode& node,
                      const std::vector<Mesh>& meshes,
                      std::vector<uint>& indices) {
  node.aabbMin = vec3(FLT_MAX);
  node.aabbMax = vec3(-FLT_MAX);
  for (uint i = 0u; i < node.count; i++) {
    const Mesh& leaf = meshes[indices[node.leftChild + i]];
    node.aabbMin = min(node.aabbMin, leaf.transformedAABBMin);
    node.aabbMin = min(node.aabbMin, leaf.transformedAABBMax);
    node.aabbMax = max(node.aabbMax, leaf.transformedAABBMin);
    node.aabbMax = max(node.aabbMax, leaf.transformedAABBMax);
  }
}

// Determine mesh counts and bounds for given split candidate
float evaluateSAH(TLASNode& node, uint axis, float pos,
                  const std::vector<Mesh>& meshes,
                  std::vector<uint>& indices) {
  AABB leftBox{};
  AABB rightBox{};

  uint leftCount{0};
  uint rightCount{0};

  for (uint i = 0; i < node.count; i++) {
    const Mesh& mesh = meshes[indices[node.leftChild + i]];
    if (mesh.transformedCentroid[axis] < pos) {
      leftCount++;
      leftBox.grow(mesh.transformedAABBMin);
      leftBox.grow(mesh.transformedAABBMax);
    } else {
      rightCount++;
      rightBox.grow(mesh.transformedAABBMin);
      rightBox.grow(mesh.transformedAABBMax);
    }
  }

  // Sum of the products of child box mesh counts and box surface areas
  float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
  return cost > 0 ? cost : FLT_MAX;
}

// Determine best split axis and position using SAH
float findBestSplitPlane(TLASNode& node,
                         const std::vector<Mesh>& meshes,
                         std::vector<uint>& indices,
                         uint& bestAxis, float& splitPos) {
  float bestCost = FLT_MAX;
  const uint COUNT = 64;

  for (uint axis = 0u; axis < 3u; axis++) {
    float boundsMin = FLT_MAX;
    float boundsMax = -FLT_MAX;

    // Split the space bounded by mesh centroids
    for (uint i = 0u; i < node.count; i++) {
      const Mesh& mesh = meshes[indices[node.leftChild + i]];
      boundsMin = min(boundsMin, mesh.transformedCentroid[axis]);
      boundsMax = max(boundsMax, mesh.transformedCentroid[axis]);
    }

    if (boundsMin == boundsMax) {
      // Flat in given dimension
      continue;
    }

    std::vector<Bin> bins{COUNT};
    float binSize = COUNT / (boundsMax - boundsMin);

    for (uint i = 0; i < node.count; i++) {
      const Mesh& mesh = meshes[indices[node.leftChild + i]];
      uint binIdx = min((float)COUNT - 1.0f, floor((mesh.transformedCentroid[axis] - boundsMin) * binSize));
      bins[binIdx].count++;
      bins[binIdx].bounds.grow(mesh.transformedAABBMin);
      bins[binIdx].bounds.grow(mesh.transformedAABBMax);
    }

    std::vector<float> leftArea(COUNT - 1u);
    std::vector<uint> leftCount(COUNT - 1u);

    std::vector<float> rightArea(COUNT - 1u);
    std::vector<uint> rightCount(COUNT - 1u);

    AABB leftBox;
    uint leftSum{0};

    AABB rightBox;
    uint rightSum{0};

    for (int i = 0; i < COUNT - 1; i++) {
      leftSum += bins[i].count;
      leftCount[i] = leftSum;
      leftBox.grow(bins[i].bounds);
      leftArea[i] = leftBox.area();

      rightSum += bins[COUNT - 1 - i].count;
      rightCount[COUNT - 2 - i] = rightSum;
      rightBox.grow(bins[COUNT - 1 - i].bounds);
      rightArea[COUNT - 2 - i] = rightBox.area();
    }

    float slabSize = (boundsMax - boundsMin) / COUNT;
    for (uint i = 0; i < COUNT - 1; i++) {
      float planeCost = 2.0f * leftCount[i] * leftArea[i] + rightCount[i] * rightArea[i];
      //  std::cout << "Split: " << planeCost << " vs " << bestCost << std::endl;

      if (planeCost < bestCost) {
        splitPos = boundsMin + slabSize * (i + 1u);
        bestAxis = axis;
        bestCost = planeCost;
      }
    }
  }
  return bestCost;
}

float getNodeCost(const TLASNode& node) {
  vec3 dim = node.aabbMax - node.aabbMin;
  return node.count * 2.0f * (dim.x * dim.y + dim.y * dim.z + dim.z * dim.x);
}

// Recursively divide TLAS node down to child nodes and include them in the tree
void subdivide(std::vector<TLASNode>& tlas,
               const std::vector<Mesh>& meshes,
               std::vector<uint>& indices,
               uint& nodeIdx,
               uint& nodesUsed) {
  TLASNode& node = tlas[nodeIdx];

  // Determine best split axis and position using SAH
  uint bestAxis{0};
  float splitPos{0};
  float bestSplitCost = findBestSplitPlane(node, meshes, indices, bestAxis, splitPos);

  if (bestSplitCost >= getNodeCost(node)) {
    return;
  }

  // Traverse list of indices from front and back
  uint i = node.leftChild;
  uint j = i + node.count - 1;
  // While the elements are not the same
  while (j < UINT_MAX && i <= j) {
    // If element is to the left of the partition, skip over it
    if (meshes[indices[i]].transformedCentroid[bestAxis] < splitPos) {
      i++;
    } else {
      // Swap the element with the element at the back
      // Decrement rear index counter (suitability of swapped element is evaluated next loop iteration)
      std::swap(indices[i], indices[j--]);
    }
  }

  // Abort split if one of the sides is empty
  uint leftCount = i - node.leftChild;
  if (leftCount == 0 || leftCount == node.count) {
    return;
  }

  // Create child nodes. Left node is followed by right one
  uint leftChildIdx = nodesUsed++;
  uint rightChildIdx = nodesUsed++;

  // Left has meshes [0...leftCount) of the parent node
  tlas[leftChildIdx].leftChild = node.leftChild;
  tlas[leftChildIdx].count = leftCount;

  // Right has meshes [leftCount...count)
  tlas[rightChildIdx].leftChild = i;
  tlas[rightChildIdx].count = node.count - leftCount;

  // Mark parent node as an internal one with reference to left child node
  node.leftChild = leftChildIdx;
  node.count = 0;

  updateNodeBounds(tlas[leftChildIdx], meshes, indices);
  updateNodeBounds(tlas[rightChildIdx], meshes, indices);

  // Recurse
  subdivide(tlas, meshes, indices, leftChildIdx, nodesUsed);
  subdivide(tlas, meshes, indices, rightChildIdx, nodesUsed);
}

void buildTLAS(
    std::vector<TLASNode>& tlas,
    const std::vector<Mesh>& meshes,
    std::vector<uint>& indices,
    uint& rootNodeIdx,
    uint& nodesUsed) {
  TLASNode& root = tlas[rootNodeIdx];

  root.leftChild = 0;
  root.count = meshes.size();
  updateNodeBounds(root, meshes, indices);
  subdivide(tlas, meshes, indices, rootNodeIdx, nodesUsed);
}
