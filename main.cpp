#include <getopt.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stack>
#include <thread>
#include <vector>

#include "brdf.hpp"
#include "camera.hpp"
#include "colors.hpp"
#include "error.hpp"
#include "geometry.hpp"
#include "gpuRender.hpp"
#include "image.hpp"
#include "input.hpp"
#include "intersection.hpp"
#include "mesh.hpp"
#include "output.hpp"
#include "random.hpp"
#include "sampleScenes.hpp"
#include "scene.hpp"
#include "tonemapping.hpp"

using namespace glm;

#define INV2PI (1.0f / (2.0f * M_PI))
#define INVPI (1.0f / M_PI)

Image environment = loadEnvironmentImage("environment/evening_road_01_puresky_2k.hdr");

// Atomically incremented index to control which threads works on which pixel
std::atomic<uint> atomicIdx{0u};

/*
  TODO:
    Stackless traversal

    Specular dielectric
    Transmission
    Alpha texture
    Lights

    Robust self-intersection fix
    Normal mapping
*/

//----------------------- Rotations --------------------------

inline vec3 rotate(vec3 p, vec4 q) {
  return 2.0f * cross(vec3(q), p * q.w + cross(vec3(q), p)) + p;
}
inline vec3 rotateX(vec3 p, float angle) {
  return rotate(p, vec4(sin(angle / 2.0), 0.0, 0.0, cos(angle / 2.0)));
}
inline vec3 rotateY(vec3 p, float angle) {
  return rotate(p, vec4(0.0, sin(angle / 2.0), 0.0, cos(angle / 2.0)));
}
inline vec3 rotateZ(vec3 p, float angle) {
  return rotate(p, vec4(0.0, 0.0, sin(angle / 2.0), cos(angle / 2.0)));
}

vec3 getEnvironment(const vec3& direction) {
  vec3 sampleDir = normalize(rotateY(direction, -M_PI));
  uint u = environment.width * (atan2f(sampleDir.z, sampleDir.x) * INV2PI + 0.5f);
  uint v = environment.height * acosf(sampleDir.y) * INVPI;
  uint idx = min(u + v * environment.width, (environment.width * environment.height) - 1);

  return environment[idx];
}

vec3 getIllumination(Ray ray,
                     const Scene& scene,
                     uint& rngState,
                     int& bounceCount,
                     uint& testCount) {
  // Initialize light to white and track attenuation
  vec3 col{1};

  for (uint i = 0; i < bounceCount; i++) {
    HitRecord closestHit{};
    uint meshIdx = scene.intersect(ray, closestHit, testCount);

    if (ray.t < FLT_MAX) {
      const Mesh& mesh{scene.meshes[meshIdx]};

      vec3 p = ray.origin + ray.direction * ray.t;
      vec3 N = normalize(vec3(transpose(mesh.invModelMatrix) * vec4(mesh.geometry->getNormal(closestHit.hitIndex, closestHit.barycentric), 0.0f)));
      if (dot(ray.direction, N) > 0.0f) {
        N = -N;
      }

      vec3 V = -ray.direction;

      const float metalness = {mesh.material->metalness};
      const float roughness = {mesh.material->roughness};
      vec2 uv = mesh.geometry->getTexCoord(closestHit.hitIndex, closestHit.barycentric);
      const vec3 albedo = mesh.material->getAlbedo(uv);
      const vec3 emissive = mesh.material->getEmissive(uv);

      vec3 F0 = mix(mesh.material->F0, albedo, metalness);

      vec3 localCol{};
      vec3 sampleDir{};

      if (metalness == 0.0f) {
        //--------------------- Diffuse ------------------------

        vec2 Xi = getRandomVec2(rngState);

        sampleDir = importanceSampleCosine(Xi, N);

        /*
            The discrete Riemann sum for the lighting equation is
            1/N * Σ(brdf(l, v) * L(l) * dot(l, n)) / pdf(l))
            Lambertian BRDF is c/PI and the pdf for cosine sampling is dot(l, n)/PI
            PI term and dot products cancel out leaving just c * L(l)
        */
        localCol = albedo;

      } else {
        //--------------------- Specular ------------------------

        vec2 Xi = getRandomVec2(rngState);
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
            vec3 specular = (L(sampleDir) * brdfS * NdotL) / pdfSpecular;

        */

        // Simplified from the above

        localCol = (F * G * VdotH) / (NdotV * NdotH);
      }
      col *= localCol + emissive;
      ray = Ray{p + 1e-4f * N, sampleDir, 1.0f / sampleDir, FLT_MAX};
    } else {
      col *= getEnvironment(ray.direction);
      break;
    }
  }

  return col;
}

//-------------------------- Render ---------------------------

void render(
    const Scene& scene,
    const Camera& camera,
    Image& image,
    const uint samples,
    const int maxBounces,
    const bool renderBVH,
    const uint threadId) {
  Ray ray{.origin = camera.position};
  vec2 fragCoord{};

  uint rngState{1031};

  const vec2 resolution{image.width, image.height};
  const float inverseSize = 1.0f / image.data.size();

  for (auto idx = atomicIdx.fetch_add(1, std::memory_order_relaxed);
       idx < image.data.size();
       idx = atomicIdx.fetch_add(1, std::memory_order_relaxed)) {
    if (threadId < 1) {
      // First thread outputs progress
      std::cout << "\r" << int(101.0f * (float)idx * inverseSize) << "%";
    }

    fragCoord = vec2{(float)(idx % image.width), std::floor((float)idx / image.width)};

    vec3 col{0};
    for (uint s = 0u; s < (renderBVH ? 1u : samples); s++) {
      vec2 fC = fragCoord;
      if (!renderBVH && samples > 1u) {
        // Jitter position for antialiasing
        fC += 0.5f * (2.0f * getRandomVec2(rngState) - 1.0f);
      }
      ray.direction = rayDirection(resolution, camera.fieldOfView, fC);
      ray.direction = normalize(viewMatrix(camera.position, camera.target, camera.up) * ray.direction);
      ray.invDirection = 1.0f / ray.direction;
      ray.t = FLT_MAX;

      uint bvhTests = 0u;
      if (renderBVH) {
        HitRecord closestHit{};
        // Get number of BVH tests for primary ray
        scene.intersect(ray, closestHit, bvhTests);
        image[idx] = vec3(bvhTests);
      } else {
        // Path trace scene
        int bounces = maxBounces;
        col += getIllumination(ray, scene, rngState, bounces, bvhTests);
      }
    }
    if (!renderBVH) {
      // Average result
      col /= samples;
      // An attempt at colour grading
      col *= smoothstep(vec3{-0.75f}, vec3{1.45f}, col);
      // Tonemapping
      col = ACESFilm(0.275f * col);
      // Gamma correction
      col = pow(col, vec3{1.0f / 2.2f});
      // Output data
      image[idx] = col;
    }
  }
}

int main(int argc, char** argv) {
  // Default values
  uint width{750};
  uint height{400};
  uint samples{32};
  uint bounces{6};
  uint numThreads{10};
  bool renderBVH{false};
  SampleScene sampleScene{THREE_STL};

  // Parse command line arguments
  int opt;
  while ((opt = getopt(argc, argv, " w:h:s:b:t:p:ad:")) != -1) {
    switch (opt) {
      case 'w':
        width = atoi(optarg);
        continue;

      case 'h':
        height = atoi(optarg);
        continue;

      case 's':
        samples = atoi(optarg);
        continue;

      case 'b':
        bounces = atoi(optarg);
        continue;

      case 't':
        numThreads = atoi(optarg);
        continue;

      case 'p':
        sampleScene = static_cast<SampleScene>(atoi(optarg));
        continue;

      case 'a':
        renderBVH = true;
        continue;

      default:
        std::cout << "Options\n-w\t<width>\n-h\t<height>\n-s\t<samples>\n-t\t<threads>\n-b\t<bounces>\n-p\t<0|1|2>\tpreset scene\n-a\trender BVH heat map (only main ray, single sample, no jitter)\n";
        return EXIT_SUCCESS;
        break;
    }

    break;
  }

  if (renderBVH) {
    samples = 1u;
    bounces = 1u;
  }

  std::cout << "\nDimensions: [" << width << ", " << height << "]\tSamples: " << samples
            << "\tBounces: " << bounces << "\tThreads: " << numThreads << std::endl
            << std::endl;

  Image image{width, height};

  // Initialize data to black
  for (auto& v : image.data) {
    v = vec3{0};
  }

  Camera camera{
      .position = 1.0f * vec3{0.5f, 0.25f, -0.8f},
      .target = vec3{0},
      .up = normalize(vec3{0, 1, 0}),
      .fieldOfView = 45.0f};

  /* Timer */ auto start{std::chrono::steady_clock::now()};

  // Scene scene(meshes);
  std::vector<std::shared_ptr<Geometry>> geometryPool{};
  std::vector<std::shared_ptr<Material>> materialPool{};

  Scene scene{};
  getScene(sampleScene, scene, geometryPool, materialPool, camera);

  // ----- Render geometry_ ----- //

  /* Timer */ start = std::chrono::steady_clock::now();

  renderGPU(scene, geometryPool, materialPool, camera, image, environment, samples, bounces, renderBVH);

  /*
    std::vector<std::thread> threads(numThreads);

    // Launch threads
    for (uint i = 0u; i < numThreads; i++) {
      threads[i] = std::thread(render,
                               std::ref(scene),
                               std::ref(camera),
                               std::ref(image),
                               samples, bounces, renderBVH, i);
    }

    // Wait for all threads to finish
    for (auto& t : threads) {
      t.join();
    }
  */
  /* Timer */ std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - start;
  /* Timer */ std::cout << "\nRender time: " << std::floor(elapsed_seconds.count() * 1e4f) / 1e4f << " s\n";

  // ----- Output ----- //

  if (renderBVH) {
    vec3 maxElement = *std::max_element(image.data.begin(), image.data.end(), [](vec3& a, vec3& b) { return a.x < b.x; });
    std::cout << "Maximum BVH tests: " << maxElement.x << std::endl;

    float inverseMaxElement = 1.0f / maxElement.x;

    for (vec3& p : image.data) {
      if (p.x > 0.0f) {
        p = afmhot(p.x * inverseMaxElement);
      }
    }
  }

  outputToFile(image);

  return EXIT_SUCCESS;
}