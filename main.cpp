

#include <chrono>
#include <cstdlib>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>

#include "output.hpp"

using namespace glm;

void traverseBVH(const BVH* bvh) {
}

//-------------------------- Rays ---------------------------

// Generate default ray for a fragment based on its position, the image and the camera
vec3 rayDirection(const vec2& resolution, float fieldOfView, const vec2& fragCoord) {
  vec2 xy = fragCoord - 0.5f * resolution;
  float z = (0.5f * resolution.y) / (0.5f * tan(radians(fieldOfView)));
  return normalize(vec3(xy, -z));
}

//-------------------------- Render ---------------------------

void renderTile(const std::vector<vec3>& target, const Extent& extent) {
}

void render(const Camera& camera, const vec2& resolution, std::vector<vec3>& target) {
  Ray ray{camera.position, vec3{0, 0, -1}, 0.0f};
  vec2 fragCoord{};

  for (int i = 0; i < target.size(); i++) {
    fragCoord = {std::fmod(i, resolution.x), std::floor(i / resolution.y)};
    ray.direction = rayDirection(resolution, camera.fieldOfView, fragCoord);
    ray.direction = lookAt(camera.position, vec3{0}, camera.up) * vec4{ray.direction, 0.0};

    target[i] = 0.5f + 0.5f * ray.direction;
  }
}

int main() {
  const uint width{512};
  const uint height{256};

  const vec2 resolution{width, height};

  std::vector<vec3> target(width * height);

  // Initialize data to black
  for (auto& v : target) {
    v = vec3{0};
  }

  vec3 cameraPosition = vec3{1, 1, -1};
  Camera camera{
      .position = cameraPosition,
      .target = vec3{0},
      .up = normalize(vec3{0, 1, 0}),
      .fieldOfView = 65.0f};

  const auto start{std::chrono::steady_clock::now()};

  render(camera, resolution, target);

  const auto end{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> elapsed_seconds{end - start};

  std::cout << "Time: " << std::floor(elapsed_seconds.count() * 1e4) / 1e4 << " s\n";

  outputToFile(resolution, target);

  return EXIT_SUCCESS;
}