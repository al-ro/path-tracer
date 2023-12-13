

#include <chrono>
#include <cstdlib>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>

#include "output.hpp"

using namespace glm;

void traverseBVH(const BVH* bvh) {
}

void renderTile(const std::vector<vec3>& target, Extent extent) {
}

void render(const vec2 resolution, std::vector<vec3>& target) {
  for (int i = 0; i < target.size(); i++) {
    target[i].r = (std::fmod(i, resolution.x) / resolution.x);
    target[i].g = (std::fmod(std::floor(i / resolution.x), resolution.y) / resolution.y);
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

  const auto start{std::chrono::steady_clock::now()};

  render(resolution, target);

  const auto end{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> elapsed_seconds{end - start};

  std::cout << "Time: " << std::floor(elapsed_seconds.count() * 1e4) / 1e4 << " s\n";

  outputToFile(resolution, target);

  return EXIT_SUCCESS;
}