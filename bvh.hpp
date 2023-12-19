#include <vector>

#include "dataTypes.hpp"

void updateNodeBounds(BVHNode& node, const std::vector<Triangle>& primitives, std::vector<uint>& indices);

// Determine triangle counts and bounds for given split candidate
float evaluateSAH(BVHNode& node, uint axis, float pos, const std::vector<Triangle>& primitives, std::vector<uint>& indices);

// Determine best split axis and position using SAH
float findBestSplitPlane(BVHNode& node, const std::vector<Triangle>& primitives, std::vector<uint>& indices,
                         uint& bestAxis, float& splitPos);

float getNodeCost(const BVHNode& node);

// Recursively divide BVH node down to child nodes and include them in the tree
void subdivide(std::vector<BVHNode>& bvh, const std::vector<Triangle>& primitives, std::vector<uint>& indices,
               uint& nodeIdx, uint& nodesUsed);

void buildBVH(std::vector<BVHNode>& bvh, const std::vector<Triangle>& primitives, std::vector<uint>& indices,
              uint& rootNodeIdx, uint& nodesUsed);