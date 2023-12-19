#include "bvh.hpp"

#include <iostream>
//-------------------------- Build BVH ---------------------------

void updateNodeBounds(BVHNode& node,
                      const std::vector<Triangle>& primitives,
                      std::vector<uint>& indices) {
  node.aabbMin = vec3(FLT_MAX);
  node.aabbMax = vec3(-FLT_MAX);
  for (uint i = 0; i < node.count; i++) {
    const Triangle& leaf = primitives[indices[node.leftFirst + i]];
    node.aabbMin = min(node.aabbMin, leaf.v0);
    node.aabbMin = min(node.aabbMin, leaf.v1);
    node.aabbMin = min(node.aabbMin, leaf.v2);
    node.aabbMax = max(node.aabbMax, leaf.v0);
    node.aabbMax = max(node.aabbMax, leaf.v1);
    node.aabbMax = max(node.aabbMax, leaf.v2);
  }
}

// Determine triangle counts and bounds for given split candidate
float evaluateSAH(BVHNode& node, uint axis, float pos,
                  const std::vector<Triangle>& primitives,
                  std::vector<uint>& indices) {
  AABB leftBox{};
  AABB rightBox{};

  uint leftCount{0};
  uint rightCount{0};

  for (uint i = 0; i < node.count; i++) {
    const Triangle& triangle = primitives[indices[node.leftFirst + i]];
    if (triangle.centroid[axis] < pos) {
      leftCount++;
      leftBox.grow(triangle.v0);
      leftBox.grow(triangle.v1);
      leftBox.grow(triangle.v2);
    } else {
      rightCount++;
      rightBox.grow(triangle.v0);
      rightBox.grow(triangle.v1);
      rightBox.grow(triangle.v2);
    }
  }

  // Sum of the products of child box primitive counts and box surface areas
  float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
  return cost > 0 ? cost : FLT_MAX;
}

// Determine best split axis and position using SAH
float findBestSplitPlane(BVHNode& node,
                         const std::vector<Triangle>& primitives,
                         std::vector<uint>& indices,
                         uint& bestAxis, float& splitPos) {
  float bestCost = FLT_MAX;
  const uint COUNT = 64;

  for (uint axis = 0u; axis < 3u; axis++) {
    float boundsMin = FLT_MAX;
    float boundsMax = -FLT_MAX;

    // Split the space bounded by primitive centroids
    for (uint i = 0u; i < node.count; i++) {
      const Triangle& triangle = primitives[indices[node.leftFirst + i]];
      boundsMin = min(boundsMin, triangle.centroid[axis]);
      boundsMax = max(boundsMax, triangle.centroid[axis]);
    }

    if (boundsMin == boundsMax) {
      // Flat in given dimension
      continue;
    }

    std::vector<Bin> bins{COUNT};
    float binSize = COUNT / (boundsMax - boundsMin);

    for (uint i = 0; i < node.count; i++) {
      const Triangle& triangle = primitives[indices[node.leftFirst + i]];
      uint binIdx = min((float)COUNT - 1.0f, floor((triangle.centroid[axis] - boundsMin) * binSize));
      bins[binIdx].count++;
      bins[binIdx].bounds.grow(triangle.v0);
      bins[binIdx].bounds.grow(triangle.v1);
      bins[binIdx].bounds.grow(triangle.v2);
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
      if (planeCost < bestCost) {
        splitPos = boundsMin + slabSize * (i + 1u);
        bestAxis = axis;
        bestCost = planeCost;
      }
    }
  }
  return bestCost;
}

float getNodeCost(const BVHNode& node) {
  vec3 dim = node.aabbMax - node.aabbMin;
  float area = 2.0f * (dim.x * dim.y + dim.y * dim.z + dim.z * dim.x);
  return node.count * area;
}

// Recursively divide BVH node down to child nodes and include them in the tree
void subdivide(std::vector<BVHNode>& bvh,
               const std::vector<Triangle>& primitives,
               std::vector<uint>& indices,
               uint& nodeIdx,
               uint& nodesUsed) {
  BVHNode& node = bvh[nodeIdx];

  // Determine best split axis and position using SAH
  uint bestAxis{0};
  float splitPos{0};
  float bestSplitCost = findBestSplitPlane(node, primitives, indices, bestAxis, splitPos);

  if (bestSplitCost >= getNodeCost(node)) {
    return;
  }

  // Traverse list of indices from front and back
  uint i = node.leftFirst;
  uint j = i + node.count - 1;
  // While the elements are not the same
  while (j < UINT32_MAX && i <= j) {
    // If element is to the left of the partition, skip over it
    if (primitives[indices[i]].centroid[bestAxis] < splitPos) {
      i++;
    } else {
      // Swap the element with the element at the back
      // Decrement rear index counter (suitability of swapped element is evaluated next loop iteration)
      std::swap(indices[i], indices[j--]);
    }
  }

  // Abort split if one of the sides is empty
  uint leftCount = i - node.leftFirst;
  if (leftCount == 0 || leftCount == node.count) {
    return;
  }

  // Create child nodes. Left node is followed by right one
  uint leftChildIdx = nodesUsed++;
  uint rightChildIdx = nodesUsed++;

  // Left has primitives [0...leftCount) of the parent node
  bvh[leftChildIdx].leftFirst = node.leftFirst;
  bvh[leftChildIdx].count = leftCount;

  // Right has primitives [leftCount...count)
  bvh[rightChildIdx].leftFirst = i;
  bvh[rightChildIdx].count = node.count - leftCount;

  // Mark parent node as an internal one with reference to left child node
  node.leftFirst = leftChildIdx;
  node.count = 0;

  updateNodeBounds(bvh[leftChildIdx], primitives, indices);
  updateNodeBounds(bvh[rightChildIdx], primitives, indices);

  // Recurse
  subdivide(bvh, primitives, indices, leftChildIdx, nodesUsed);
  subdivide(bvh, primitives, indices, rightChildIdx, nodesUsed);
}

void buildBVH(
    std::vector<BVHNode>& bvh,
    const std::vector<Triangle>& primitives,
    std::vector<uint>& indices,
    uint& rootNodeIdx,
    uint& nodesUsed) {
  BVHNode& root = bvh[rootNodeIdx];

  root.leftFirst = 0;
  root.count = primitives.size();
  updateNodeBounds(root, primitives, indices);
  subdivide(bvh, primitives, indices, rootNodeIdx, nodesUsed);
}
