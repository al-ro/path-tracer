#include <vector>

#include "dataTypes.hpp"
#include "mesh.hpp"

void updateNodeBounds(TLASNode& node, const std::vector<Mesh>& meshes, std::vector<uint>& indices);

// Determine mesh counts and bounds for given split candidate
float evaluateSAH(TLASNode& node, uint axis, float pos, const std::vector<Mesh>& meshes, std::vector<uint>& indices);

// Determine best split axis and position using SAH
float findBestSplitPlane(TLASNode& node, const std::vector<Mesh>& meshes, std::vector<uint>& indices,
                         uint& bestAxis, float& splitPos);

float getNodeCost(const TLASNode& node);

// Recursively divide TLAS node down to child nodes and include them in the tree
void subdivide(std::vector<TLASNode>& tlas, const std::vector<Mesh>& meshes, std::vector<uint>& indices,
               uint& nodeIdx, uint& nodesUsed);

void buildTLAS(std::vector<TLASNode>& tlas, const std::vector<Mesh>& meshes, std::vector<uint>& indices,
               uint& rootNodeIdx, uint& nodesUsed);