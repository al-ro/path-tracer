#include "bvh.hpp"

float getNodeCost(const BVHNode& node) {
  vec3 dim = node.aabbMax - node.aabbMin;
  float area = 2.0f * (dim.x * dim.y + dim.y * dim.z + dim.z * dim.x);
  return node.count * area;
}