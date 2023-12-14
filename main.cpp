#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "intersection.h"
#include "output.h"
#include "random.h"

using namespace glm;

void traverseBVH(const BVHNode* bvh) {
}

//-------------------------- Rays ---------------------------

// Generate default ray for a fragment based on its position, the image and the camera
vec3 rayDirection(const vec2& resolution, float fieldOfView, const vec2& fragCoord) {
  vec2 xy = fragCoord - 0.5f * resolution;
  float z = (0.5f * resolution.y) / tan(0.5f * radians(fieldOfView));
  return normalize(vec3(xy, -z));
}

mat3 viewMatrix(vec3 camera, vec3 at, vec3 up) {
  vec3 zaxis = normalize(at - camera);
  vec3 xaxis = normalize(cross(zaxis, up));
  vec3 yaxis = cross(xaxis, zaxis);

  return mat3(xaxis, yaxis, -zaxis);
}

//-------------------------- Construct BVH ---------------------------

void updateNodeBounds(BVHNode& node,
                      const std::vector<Triangle>& scene,
                      std::vector<uint>& sceneIndices) {
  node.aabbMin = vec3(FLT_MAX);
  node.aabbMax = vec3(-FLT_MAX);
  for (uint i = 0; i < node.count; i++) {
    const Triangle& leaf = scene[sceneIndices[node.leftFirst + i]];
    node.aabbMin = min(node.aabbMin, leaf.v0);
    node.aabbMin = min(node.aabbMin, leaf.v1);
    node.aabbMin = min(node.aabbMin, leaf.v2);
    node.aabbMax = max(node.aabbMax, leaf.v0);
    node.aabbMax = max(node.aabbMax, leaf.v1);
    node.aabbMax = max(node.aabbMax, leaf.v2);
  }
}

// Determine triangle counts and bounds for given split candidate
float evaluateSAH(BVHNode& node, int axis, float pos,
                  const std::vector<Triangle>& scene,
                  std::vector<uint>& sceneIndices) {
  AABB leftBox{};
  AABB rightBox{};

  uint leftCount{0};
  uint rightCount{0};

  for (uint i = 0; i < node.count; i++) {
    const Triangle& triangle = scene[sceneIndices[node.leftFirst + i]];
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

// Recursively divide BVH node down to child nodes and include them in the tree
void subdivide(std::vector<BVHNode>& bvh,
               const std::vector<Triangle>& scene,
               std::vector<uint>& sceneIndices,
               uint& nodeIdx,
               uint& nodesUsed) {
  BVHNode& node = bvh[nodeIdx];

  // Determine best split axis and position using SAH
  uint bestAxis{0};
  float splitPos{0};
  float bestCost = FLT_MAX;
  for (int axis = 0; axis < 3; axis++) {
    for (uint i = 0; i < node.count; i++) {
      float candidatePos = scene[sceneIndices[node.leftFirst + i]].centroid[axis];
      float cost = evaluateSAH(node, axis, candidatePos, scene, sceneIndices);
      if (cost < bestCost) {
        splitPos = candidatePos;
        bestAxis = axis;
        bestCost = cost;
      }
    }
  }

  vec3 dim = node.aabbMax - node.aabbMin;
  float parentArea = 2.0f * (dim.x * dim.y + dim.y * dim.z + dim.z * dim.x);
  float parentCost = node.count * parentArea;

  if (bestCost >= parentCost) {
    return;
  }

  // Traverse list of indices from front and back
  int i = node.leftFirst;
  int j = i + node.count - 1;
  // While the elements are not the same
  while (i <= j) {
    // If element is to the left of the partition, skip over it
    if (scene[sceneIndices[i]].centroid[bestAxis] < splitPos) {
      i++;
    } else {
      // Swap the element with the element at the back
      // Decrement rear index counter (suitability of swapped element is evaluated next loop iteration)
      std::swap(sceneIndices[i], sceneIndices[j--]);
    }
  }

  // Abort split if one of the sides is empty
  int leftCount = i - node.leftFirst;
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

  updateNodeBounds(bvh[leftChildIdx], scene, sceneIndices);
  updateNodeBounds(bvh[rightChildIdx], scene, sceneIndices);

  // Recurse
  subdivide(bvh, scene, sceneIndices, leftChildIdx, nodesUsed);
  subdivide(bvh, scene, sceneIndices, rightChildIdx, nodesUsed);
}

void buildBVH(
    std::vector<BVHNode>& bvh,
    const std::vector<Triangle>& scene,
    std::vector<uint>& sceneIndices,
    uint& rootNodeIdx,
    uint& nodesUsed) {
  BVHNode& root = bvh[rootNodeIdx];
  root.leftFirst = 0;
  root.count = scene.size();
  updateNodeBounds(root, scene, sceneIndices);
  subdivide(bvh, scene, sceneIndices, rootNodeIdx, nodesUsed);
}

//-------------------------- Traverse BVH ---------------------------

float IntersectBVH(
    Ray& ray,
    const std::vector<BVHNode>& bvh,
    const std::vector<Triangle>& scene,
    const std::vector<uint>& sceneIndices,
    const uint nodeIdx) {
  const BVHNode& node = bvh[nodeIdx];

  float t = FLT_MAX;

  if (intersect(ray, node.aabbMin, node.aabbMax) == FLT_MAX) {
    return t;
  }

  if (node.count > 0) {
    for (uint i = 0; i < node.count; i++) {
      float tt = intersect(ray, scene[sceneIndices[node.leftFirst + i]]);
      if (tt > 0.0f) {
        t = min(t, tt);
      }
    }
  } else {
    float left = IntersectBVH(ray, bvh, scene, sceneIndices, node.leftFirst);
    float right = IntersectBVH(ray, bvh, scene, sceneIndices, node.leftFirst + 1);
    t = min(left, right);
  }
  return t;
}

float intersectBVH(
    Ray& ray,
    const std::vector<BVHNode>& bvh,
    const std::vector<Triangle>& scene,
    const std::vector<uint>& sceneIndices,
    const uint nodeIdx) {
  float t = FLT_MAX;

  const BVHNode* node = &bvh[nodeIdx];
  const BVHNode* stack[64];
  uint stackPtr = 0;

  while (1) {
    if (node->count > 0) {
      for (uint i = 0; i < node->count; i++) {
        float tt = intersect(ray, scene[sceneIndices[node->leftFirst + i]]);
        if (tt > 0.0f) {
          t = min(t, tt);
        }
      }

      if (stackPtr == 0) {
        break;
      } else {
        node = stack[--stackPtr];
      }
      continue;
    }
    const BVHNode* child1 = &bvh[node->leftFirst];
    const BVHNode* child2 = &bvh[node->leftFirst + 1];
    float dist1 = intersect(ray, child1->aabbMin, child1->aabbMax);
    float dist2 = intersect(ray, child2->aabbMin, child2->aabbMax);
    if (dist1 > dist2) {
      std::swap(dist1, dist2);
      std::swap(child1, child2);
    }
    if (dist1 == FLT_MAX) {
      if (stackPtr == 0) {
        break;
      } else {
        node = stack[--stackPtr];
      }
    } else {
      node = child1;
      if (dist2 != FLT_MAX) {
        stack[stackPtr++] = child2;
      }
    }
  }
  return t;
}

//-------------------------- Render ---------------------------

void renderTile(const std::vector<vec3>& image, const Extent& extent) {
}

void render(
    const std::vector<Triangle>& scene,
    const std::vector<BVHNode>& bvh,
    const std::vector<uint>& sceneIndices,
    const Camera& camera,
    const vec2& resolution,
    std::vector<vec3>& image) {
  Ray ray{.origin = camera.position};
  vec2 fragCoord{};

  // for (int i = 0; i < image.size(); i++) {

  for (int y = 0; y < resolution.y; y += 4) {
    for (int x = 0; x < resolution.x; x += 4) {
      for (int v = 0; v < 4; v++) {
        for (int u = 0; u < 4; u++) {
          fragCoord = vec2{x + u, y + v};  //{std::fmod(i, resolution.x), std::floor((float)(i) / resolution.x)};
          ray.direction = rayDirection(resolution, camera.fieldOfView, fragCoord);
          ray.direction = normalize(viewMatrix(camera.position, camera.target, camera.up) * ray.direction);
          ray.invDirection = 1.0f / ray.direction;
          ray.t = FLT_MAX;

          int i = fragCoord.y * resolution.x + fragCoord.x;
          image[i] = 0.5f + 0.5f * ray.direction;

          float t{};
          t = intersectBVH(ray, bvh, scene, sceneIndices, 0);
          if (t > 0.0f && t < ray.t) {
            ray.t = t;
            image[i] = vec3{t / 4.0f};
          }
        }
      }
    }
  }
}

int main() {
  uint rngState{4097};

  const uint width{1800};
  const uint height{800};

  const vec2 resolution{width, height};

  std::vector<vec3> image(width * height);

  // Initialize data to black
  for (auto& v : image) {
    v = vec3{0};
  }

  Camera camera{
      .position = vec3{-1.0f, 0.2f, 1.0f},
      .target = vec3{0},
      .up = normalize(vec3{0, 1, 0}),
      .fieldOfView = 65.0f};

  std::vector<Triangle> scene(12582);

  FILE* file = fopen("obj/unity.tri", "r");
  float a, b, c, d, e, f, g, h, i;
  for (int t = 0; t < 12582; t++) {
    int result = fscanf(file, "%f %f %f %f %f %f %f %f %f\n",
                        &a, &b, &c, &d, &e, &f, &g, &h, &i);
    scene[t].v0 = vec3{a, b, c};
    scene[t].v1 = vec3{d, e, f};
    scene[t].v2 = vec3{g, h, i};
  }
  fclose(file);

  vec3 aabbMin = vec3(FLT_MAX);
  vec3 aabbMax = vec3(-FLT_MAX);

  for (auto& triangle : scene) {
    aabbMin = min(aabbMin, triangle.v0);
    aabbMin = min(aabbMin, triangle.v1);
    aabbMin = min(aabbMin, triangle.v2);
    aabbMax = max(aabbMax, triangle.v0);
    aabbMax = max(aabbMax, triangle.v1);
    aabbMax = max(aabbMax, triangle.v2);
  }

  vec3 centre = aabbMin + 0.5f * (aabbMax - aabbMin);

  for (auto& triangle : scene) {
    triangle.v0 -= centre;
    triangle.v1 -= centre;
    triangle.v2 -= centre;
  }

  for (auto& triangle : scene) {
    triangle.centroid = (triangle.v0 + triangle.v1 + triangle.v2) / 3.0f;
  }

  std::vector<uint> sceneIndices(scene.size());

  // Populate scene indices sequentially [0...N)
  for (uint i = 0u; i < sceneIndices.size(); i++) {
    sceneIndices[i] = i;
  }

  // BVH tree with reserved space for 2N-1 nodes which is the maximum number of nodes in a binary tree with N leaves
  std::vector<BVHNode> bvh{2 * scene.size() - 1};

  uint rootNodeIdx = 0;
  uint nodesUsed = 1;

  // ----- Build BVH ----- //

  auto start{std::chrono::steady_clock::now()};
  buildBVH(bvh, scene, sceneIndices, rootNodeIdx, nodesUsed);
  std::chrono::duration<double> elapsed_seconds{std::chrono::steady_clock::now() - start};
  std::cout << "BVH build time: " << std::floor(elapsed_seconds.count() * 1e4f) / 1e4f << " s\n";
  std::cout << "Nodes: " << nodesUsed << std::endl;

  // ----- Render scene ----- //

  start = std::chrono::steady_clock::now();
  render(scene, bvh, sceneIndices, camera, resolution, image);
  elapsed_seconds = std::chrono::steady_clock::now() - start;
  std::cout << "Render time: " << std::floor(elapsed_seconds.count() * 1e4f) / 1e4f << " s\n";

  // ----- Output ----- //

  outputToFile(resolution, image);

  return EXIT_SUCCESS;
}