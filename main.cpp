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

vec3 min(const vec3& a, const vec3& b) {
  return vec3{fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)};
}

vec3 max(const vec3& a, const vec3& b) {
  return vec3{fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)};
}

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

void subdivide(std::vector<BVHNode>& bvh,
               const std::vector<Triangle>& scene,
               std::vector<uint>& sceneIndices,
               uint& nodeIdx,
               uint& nodesUsed) {
  // Terminate recursion
  BVHNode& node = bvh[nodeIdx];
  if (node.count <= 2) {
    return;
  }

  // Determine longest axis
  vec3 extent = node.aabbMax - node.aabbMin;
  int axis = 0;
  if (extent.y > extent.x) {
    axis = 1;
  }
  if (extent.z > extent[axis]) {
    axis = 2;
  }

  // Split halfway along the exis
  float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;

  // Traverse list of indices from front and back
  int i = node.leftFirst;
  int j = i + node.count - 1;
  // While the elements are not the same
  while (i <= j) {
    // If element is to the left of the partition, skip over it
    if (scene[sceneIndices[i]].centroid[axis] < splitPos) {
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
  updateNodeBounds(bvh[rootNodeIdx], scene, sceneIndices);
  subdivide(bvh, scene, sceneIndices, rootNodeIdx, nodesUsed);
}

//-------------------------- Traverse BVH ---------------------------

float intersectBVH(
    Ray& ray,
    const std::vector<BVHNode>& bvh,
    const std::vector<Triangle>& scene,
    const std::vector<uint>& sceneIndices,
    const uint nodeIdx) {
  const BVHNode& node = bvh[nodeIdx];

  float t = FLT_MAX;

  if (!intersect(ray, node.aabbMin, node.aabbMax)) {
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
    float left = intersectBVH(ray, bvh, scene, sceneIndices, node.leftFirst);
    float right = intersectBVH(ray, bvh, scene, sceneIndices, node.leftFirst + 1);
    t = min(left, right);
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
  Ray ray{camera.position, vec3{}, 0.0f};
  vec2 fragCoord{};

  for (int i = 0; i < image.size(); i++) {
    fragCoord = {std::fmod(i, resolution.x), std::floor((float)(i) / resolution.x)};
    ray.direction = rayDirection(resolution, camera.fieldOfView, fragCoord);
    ray.direction = normalize(viewMatrix(camera.position, camera.target, camera.up) * ray.direction);
    ray.t = FLT_MAX;

    image[i] = 0.5f + 0.5f * ray.direction;
    float t{};
    t = intersectBVH(ray, bvh, scene, sceneIndices, 0);
    if (t > 0.0f && t < ray.t) {
      ray.t = t;
      image[i] = vec3(0.95);
    }
  }
}

int main() {
  uint rngState{4097};

  const uint width{512};
  const uint height{256};

  const vec2 resolution{width, height};

  std::vector<vec3> image(width * height);

  // Initialize data to black
  for (auto& v : image) {
    v = vec3{0};
  }

  vec3 cameraPosition = vec3{0, 0, 10};
  Camera camera{
      .position = cameraPosition,
      .target = vec3{0},
      .up = normalize(vec3{0, 1, 0}),
      .fieldOfView = 65.0f};

  std::vector<Triangle> scene(128);
  std::vector<uint> sceneIndices(scene.size());

  // Populate scene indices sequentially [0...N)
  for (uint i = 0u; i < sceneIndices.size(); i++) {
    sceneIndices[i] = i;
  }

  vec3 p{};
  for (auto& triangle : scene) {
    p = 6.0f * (2.0f * getRandomVec3(rngState) - 1.0f);

    triangle.v0 = p + normalize(2.0f * getRandomVec3(rngState) - 1.0f);
    triangle.v1 = p + normalize(2.0f * getRandomVec3(rngState) - 1.0f);
    triangle.v2 = p + normalize(2.0f * getRandomVec3(rngState) - 1.0f);

    triangle.centroid = (triangle.v0 + triangle.v1 + triangle.v2) / 3.0f;
  }

  // BVH tree with reserved space for 2N-1 nodes which is the maximum number of nodes in a binary tree with N leaves
  std::vector<BVHNode> bvh{2 * scene.size() - 1};

  uint rootNodeIdx = 0;
  uint nodesUsed = 1;

  buildBVH(bvh, scene, sceneIndices, rootNodeIdx, nodesUsed);

  const auto start{std::chrono::steady_clock::now()};

  render(scene, bvh, sceneIndices, camera, resolution, image);

  const auto end{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> elapsed_seconds{end - start};

  std::cout << "Time: " << std::floor(elapsed_seconds.count() * 1e4f) / 1e4f << " s\n";

  outputToFile(resolution, image);

  return EXIT_SUCCESS;
}