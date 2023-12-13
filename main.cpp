

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "intersection.h"
#include "output.h"
#include "random.h"

using namespace glm;

void traverseBVH(const BVH* bvh) {
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

//-------------------------- Render ---------------------------

void renderTile(const std::vector<vec3>& image, const Extent& extent) {
}

void render(const std::vector<Triangle>& scene, const Camera& camera, const vec2& resolution, std::vector<vec3>& image) {
  Ray ray{camera.position, vec3{}, 0.0f};
  vec2 fragCoord{};

  for (int i = 0; i < image.size(); i++) {
    fragCoord = {std::fmod(i, resolution.x), std::floor((float)(i) / resolution.x)};
    ray.direction = rayDirection(resolution, camera.fieldOfView, fragCoord);
    ray.direction = normalize(viewMatrix(camera.position, camera.target, camera.up) * ray.direction);
    ray.t = INT_MAX;

    image[i] = 0.5f + 0.5f * ray.direction;
    float t{};
    for (auto& triangle : scene) {
      t = intersect(ray, triangle);
      if (t > 0.0f && t < ray.t) {
        ray.t = t;
        image[i] = vec3(0.95);
      }
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

  std::vector<Triangle> scene{128};

  vec3 p{};
  vec3 angle{};

  for (auto& triangle : scene) {
    p = 6.0f * (2.0f * getRandomPoint(rngState) - 1.0f);
    angle = 10.0f * getRandomPoint(rngState);

    triangle.v0 = vec3(0.0f, 0.0f, 0.0f);
    triangle.v1 = vec3(-0.5f, 1.0f, 0.0f);
    triangle.v2 = vec3(-1.0f, -1.0f, 0.0f);

    triangle.v0 = p + rotateX(triangle.v0, angle.x);
    triangle.v1 = p + rotateY(triangle.v1, angle.y);
    triangle.v2 = p + rotateZ(triangle.v2, angle.z);
  }

  const auto start{std::chrono::steady_clock::now()};

  render(scene, camera, resolution, image);

  const auto end{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> elapsed_seconds{end - start};

  std::cout << "Time: " << std::floor(elapsed_seconds.count() * 1e4f) / 1e4f << " s\n";

  outputToFile(resolution, image);

  return EXIT_SUCCESS;
}