#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stack>
#include <thread>
#include <vector>

#include "brdf.hpp"
#include "input.hpp"
#include "intersection.hpp"
#include "output.hpp"
#include "random.hpp"

using namespace glm;

#define INV2PI (1.0f / (2.0f * M_PI))
#define INVPI (1.0f / M_PI)

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
float evaluateSAH(BVHNode& node, uint axis, float pos,
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

// Determine best split axis and position using SAH
float findBestSplitPlane(BVHNode& node,
                         const std::vector<Triangle>& scene,
                         std::vector<uint>& sceneIndices,
                         uint& bestAxis, float& splitPos) {
  float bestCost = FLT_MAX;
  const uint COUNT = 64;

  for (uint axis = 0u; axis < 3u; axis++) {
    float boundsMin = FLT_MAX;
    float boundsMax = -FLT_MAX;

    // Split the space bounded by primitive centroids
    for (uint i = 0u; i < node.count; i++) {
      const Triangle& triangle = scene[sceneIndices[node.leftFirst + i]];
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
      const Triangle& triangle = scene[sceneIndices[node.leftFirst + i]];
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
               const std::vector<Triangle>& scene,
               std::vector<uint>& sceneIndices,
               uint& nodeIdx,
               uint& nodesUsed) {
  BVHNode& node = bvh[nodeIdx];

  // Determine best split axis and position using SAH
  uint bestAxis{0};
  float splitPos{0};
  float bestSplitCost = findBestSplitPlane(node, scene, sceneIndices, bestAxis, splitPos);

  if (bestSplitCost >= getNodeCost(node)) {
    return;
  }

  // Traverse list of indices from front and back
  uint i = node.leftFirst;
  uint j = i + node.count - 1;
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

float intersectBVH(
    Ray& ray,
    const std::vector<BVHNode>& bvh,
    const std::vector<Triangle>& scene,
    const std::vector<uint>& sceneIndices,
    const uint nodeIdx,
    uint& hitIndex) {
  float t = FLT_MAX;

  const BVHNode* node = &bvh[nodeIdx];
  std::stack<const BVHNode*> stack;

  while (1) {
    // If leaf node, intersect with primitives
    if (node->count > 0) {
      for (uint i = 0; i < node->count; i++) {
        float distance = intersect(ray, scene[sceneIndices[node->leftFirst + i]]);
        if (distance > 0.0f && distance < t) {
          t = distance;
          hitIndex = sceneIndices[node->leftFirst + i];
        }
      }

      // If stack is empty, exit loop. Else grab next element on stack
      if (stack.empty()) {
        break;
      } else {
        node = stack.top();
        stack.pop();
      }
      // Skip to the start of the loop
      continue;
    }

    // Compare the distances to the two child nodes
    const BVHNode* child1 = &bvh[node->leftFirst];
    const BVHNode* child2 = &bvh[node->leftFirst + 1];
    float dist1 = intersect(ray, child1->aabbMin, child1->aabbMax);
    float dist2 = intersect(ray, child2->aabbMin, child2->aabbMax);

    // Consider closer one first
    if (dist1 > dist2) {
      std::swap(dist1, dist2);
      std::swap(child1, child2);
    }

    // If closer node is missed, the farther one is as well
    if (dist1 == FLT_MAX) {
      // Exit if stack empty or grab next element
      if (stack.empty()) {
        break;
      } else {
        node = stack.top();
        stack.pop();
      }
    } else {
      // If closer node is hit, consider it for the next loop
      node = child1;

      // If the farther node is hit, place it on the stack
      if (dist2 != FLT_MAX) {
        stack.push(child2);
      }
    }
  }
  return t;
}

//-------------------------- Render ---------------------------

float dot_c(const vec3& a, const vec3& b) {
  return max(dot(a, b), 1e-5f);
}

Image environment = loadEnvironmentImage("environment/evening_road_01_puresky_2k.hdr");

vec3 getEnvironment(const vec3& direction) {
  uint u = environment.width * atan2f(direction.z, direction.x) * INV2PI - 0.5f;
  uint v = environment.height * acosf(direction.y) * INVPI - 0.5f;
  uint idx = min(u + v * environment.width, (uint)(environment.data.size()) - 1);

  return 0.5f * environment[idx];
}

vec3 getNormal(const Triangle& triangle) {
  return normalize(cross(triangle.v0 - triangle.v1, triangle.v0 - triangle.v2));
}

float metalness = 0.0;
float roughness = 0.01;
vec3 albedo = vec3(1);

// Index of refraction for common dielectrics. Corresponds to F0 0.04
float IOR = 1.5;

// Reflectance of the surface when looking straight at it along the negative normal
vec3 F0 = mix(vec3(pow(IOR - 1.0, 2.0) / pow(IOR + 1.0, 2.0)), albedo, metalness);

vec3 getIllumination(Ray& ray,
                     const std::vector<BVHNode>& bvh,
                     const std::vector<Triangle>& scene,
                     const std::vector<vec3>& normals,
                     const std::vector<uint>& sceneIndices,
                     const uint nodeIdx,
                     uint& rngState,
                     int bounceCount) {
  if (--bounceCount < 0) {
    return vec3{0};
  }

  bool hit = true;
  vec3 col{0};
  uint hitIndex{};

  float t = intersectBVH(ray, bvh, scene, sceneIndices, nodeIdx, hitIndex);
  hit = t > 0.0f && t < ray.t;

  if (hit) {
    ray.t = t;
    vec3 p = ray.origin + ray.direction * ray.t;
    vec3 N = normals[hitIndex];
    p += 1e-4f * N;
    vec3 V = -ray.direction;

    //--------------------- Diffuse ------------------------

    vec2 Xi = getRandomVec2(rngState);

    vec3 sampleDir = importanceSampleCosine(Xi, N);
    Ray sampleRay{p, sampleDir, 1.0f / sampleDir, FLT_MAX};

    /*
        The discrete Riemann sum for the lighting equation is
        1/N * Σ(brdf(l, v) * L(l) * dot(l, n)) / pdf(l))
        Lambertian BRDF is c/PI and the pdf for cosine sampling is dot(l, n)/PI
        PI term and dot products cancel out leaving just c * L(l)
    */
    vec3 diffuse = albedo * getIllumination(sampleRay, bvh, scene, normals, sceneIndices, 0, rngState, bounceCount);

    //--------------------- Specular ------------------------

    Xi = getRandomVec2(rngState);
    // Get a random halfway vector around the surface normal (in world space)
    vec3 H = importanceSampleGGX(Xi, N, roughness);

    // Generate sample direction as view ray reflected around h (note sign)
    sampleDir = normalize(reflect(-V, H));

    float NdotL = dot_c(N, sampleDir);
    float NdotV = dot_c(N, V);
    float NdotH = dot_c(N, H);
    float VdotH = dot_c(V, H);

    vec3 F = fresnel(VdotH, F0);
    float G = smiths(NdotV, NdotL, roughness);

    /*

        The following can be simplified as the D term and many dot products cancel out

        float D = distribution(NdotH, roughness);

        // Cook-Torrance BRDF
        vec3 brdfS = D * F * G / max(0.0001, (4.0 * NdotV * NdotL));

        float pdfSpecular = (D * NdotH) / (4.0 * VdotH);
        vec3 specular = (getEnvironment(sampleDir) * brdfS * NdotL) / pdfSpecular;

    */

    // Simplified from the above

    sampleRay = Ray(p, sampleDir, 1.0f / sampleDir, FLT_MAX);
    vec3 specular = (getIllumination(sampleRay, bvh, scene, normals, sceneIndices, 0, rngState, bounceCount) * F * G * VdotH) / (NdotV * NdotH);

    // Combine diffuse and specular
    vec3 kD = (1.0f - F) * (1.0f - metalness);
    col = specular + kD * diffuse;  // (1.0-metalness) * (1.0-fresnel(NdotL, F0))*(1.0-fresnel(NdotV, F0));

  } else {
    col = getEnvironment(ray.direction);
  }
  return col;
}

std::atomic<uint> atomicIdx{0u};

void render(
    const std::vector<Triangle>& scene,
    const std::vector<vec3>& normals,
    const std::vector<BVHNode>& bvh,
    const std::vector<uint>& sceneIndices,
    const Camera& camera,
    Image& image,
    const vec2 threadInfo) {
  Ray ray{.origin = camera.position};
  vec2 fragCoord{};

  const uint samples = 32;
  uint rngState{1031};

  vec2 resolution{image.width, image.height};

  for (auto idx = atomicIdx.fetch_add(1, std::memory_order_relaxed);
       idx < image.data.size();
       idx = atomicIdx.fetch_add(1, std::memory_order_relaxed)) {
    if (threadInfo.x < 1) {
      std::cout << "\r" << int(101.f * (float)idx / image.data.size()) << "%";
    }
    fragCoord = vec2{(float)(idx % image.width), std::floor((float)idx / image.width)};

    for (uint s = 0u; s < samples; s++) {
      vec2 fC = fragCoord + 0.5f * (2.0f * getRandomVec2(rngState) - 1.0f);
      ray.direction = rayDirection(resolution, camera.fieldOfView, fC);
      ray.direction = normalize(viewMatrix(camera.position, camera.target, camera.up) * ray.direction);
      ray.invDirection = 1.0f / ray.direction;
      ray.t = FLT_MAX;

      image[idx] += getIllumination(ray, bvh, scene, normals, sceneIndices, 0, rngState, 10);
    }
    image[idx] /= samples;
    image[idx] *= 1.0f - vec3{expf(-image[idx].r), expf(-image[idx].g), expf(-image[idx].b)};
    image[idx] = pow(image[idx], vec3{1.0f / 2.2f});
  }
}

int main() {
  const uint width{750};
  const uint height{400};

  const vec2 resolution{width, height};

  Image image{width, height};

  // Initialize data to black
  for (auto& v : image.data) {
    v = vec3{0};
  }

  Camera camera{
      .position = 140.0f * vec3{-1.0f, 0.3f, 0.8f},
      .target = vec3{0},
      .up = normalize(vec3{0, 1, 0}),
      .fieldOfView = 45.0f};

  // ----- Load model, generate normals and indices ----- //

  std::vector<Triangle> scene = loadModel("models/bust-of-menelaus.stl");

  std::vector<vec3> normals{scene.size()};
  for (uint i = 0; i < scene.size(); i++) {
    normals[i] = getNormal(scene[i]);
  }

  std::vector<uint> sceneIndices(scene.size());

  // Populate scene indices sequentially [0...N)
  for (uint i = 0u; i < sceneIndices.size(); i++) {
    sceneIndices[i] = i;
  }

  // ----- Build BVH ----- //

  // BVH tree with reserved space for 2N-1 nodes which is the maximum number of nodes in a binary tree with N leaves
  std::vector<BVHNode> bvh{2 * scene.size() - 1};

  uint rootNodeIdx = 0;
  uint nodesUsed = 1;

  auto start{std::chrono::steady_clock::now()};
  buildBVH(bvh, scene, sceneIndices, rootNodeIdx, nodesUsed);
  std::chrono::duration<double> elapsed_seconds{std::chrono::steady_clock::now() - start};
  std::cout << "BVH build time: " << std::floor(elapsed_seconds.count() * 1e4f) / 1e4f << " s\n";

  std::cout << "Triangles: " << scene.size() << std::endl;
  std::cout << "Nodes: " << nodesUsed << std::endl;

  // ----- Render scene ----- //

  start = std::chrono::steady_clock::now();

  uint numThreads{8};
  std::vector<std::thread> threads;

  // Launch threads
  for (uint i = 0u; i < numThreads; i++) {
    threads.push_back(std::thread(render,
                                  std::ref(scene),
                                  std::ref(normals),
                                  std::ref(bvh),
                                  std::ref(sceneIndices),
                                  std::ref(camera),
                                  std::ref(image),
                                  vec2{i, numThreads}));
  }

  // Wait for all threads to finish
  for (auto& t : threads) {
    t.join();
  }

  elapsed_seconds = std::chrono::steady_clock::now() - start;
  std::cout << "\nRender time: " << std::floor(elapsed_seconds.count() * 1e4f) / 1e4f << " s\n";

  // ----- Output ----- //

  outputToFile(image);

  return EXIT_SUCCESS;
}