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
#include "colors.hpp"
#include "geometry.hpp"
#include "image.hpp"
#include "input.hpp"
#include "intersection.hpp"
#include "mesh.hpp"
#include "output.hpp"
#include "random.hpp"
#include "sampleScenes.hpp"
#include "scene.hpp"

using namespace glm;

#define INV2PI (1.0f / (2.0f * M_PI))
#define INVPI (1.0f / M_PI)

Image environment = loadEnvironmentImage("environment/evening_road_01_puresky_2k.hdr");

// Atomically incremented index to control which threads works on which pixel
std::atomic<uint> atomicIdx{0u};

/*
  TODO:
    Robust self-intersection fix
    Normal mapping

    CUDA

    Transmission
    Alpha texture
    Lights
*/

//-------------------------- Render ---------------------------

float dot_c(const vec3& a, const vec3& b) {
  return max(dot(a, b), 1e-5f);
}

vec3 getEnvironment(const vec3& direction) {
  uint u = environment.width * (atan2f(direction.z, direction.x) * INV2PI + 0.5f);
  uint v = environment.height * acosf(direction.y) * INVPI;
  uint idx = min(environment.width * v + u, (uint)(environment.data.size()) - 1);

  return 0.5f * environment[idx];
}

vec3 getIllumination(Ray ray,
                     const Scene& scene,
                     uint& rngState,
                     int& bounceCount,
                     uint& testCount) {
  if (--bounceCount < 0) {
    return vec3{0};
  }

  vec3 col{0};

  HitRecord closestHit{};

  uint meshIdx = scene.intersect(ray, closestHit, testCount);

  if (ray.t < FLT_MAX) {
    const Mesh& mesh{scene.meshes[meshIdx]};

    vec3 p = ray.origin + ray.direction * ray.t;
    vec3 N = normalize(vec3(mesh.normalMatrix * vec4(mesh.geometry->getNormal(closestHit.hitIndex, closestHit.barycentric), 0.0f)));
    if (dot(ray.direction, N) > 0.0f) {
      N = -N;
    }
    vec3 V = -ray.direction;

    const float metalness{mesh.material->metalness};
    const float roughness{mesh.material->roughness};
    vec2 uv = mesh.geometry->getTexCoord(closestHit.hitIndex, closestHit.barycentric);
    const vec3 albedo = mesh.material->getAlbedo(uv);
    const vec3 emissive = mesh.material->getEmissive(uv);

    vec3 F0 = mix(mesh.material->F0, albedo, metalness);

    //--------------------- Diffuse ------------------------
    vec3 diffuse{0};
    if (metalness < 1.0f) {
      vec2 Xi = getRandomVec2(rngState);

      vec3 sampleDir = importanceSampleCosine(Xi, N);
      Ray sampleRay{p, sampleDir, 1.0f / sampleDir, FLT_MAX};
      sampleRay.origin += 1e-4f * N;

      /*
          The discrete Riemann sum for the lighting equation is
          1/N * Σ(brdf(l, v) * L(l) * dot(l, n)) / pdf(l))
          Lambertian BRDF is c/PI and the pdf for cosine sampling is dot(l, n)/PI
          PI term and dot products cancel out leaving just c * L(l)
      */
      diffuse = albedo * getIllumination(sampleRay, scene, rngState, bounceCount, testCount);
    }

    //--------------------- Specular ------------------------

    vec2 Xi = getRandomVec2(rngState);
    // Get a random halfway vector around the surface normal (in world space)
    vec3 H = importanceSampleGGX(Xi, N, roughness);

    // Generate sample direction as view ray reflected around h (note sign)
    vec3 sampleDir = normalize(reflect(-V, H));

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

    Ray sampleRay{p, sampleDir, 1.0f / sampleDir, FLT_MAX};
    sampleRay.origin += 1e-4f * N;

    vec3 specular = (getIllumination(sampleRay, scene, rngState, bounceCount, testCount) * F * G * VdotH) / (NdotV * NdotH);

    // Combine diffuse and specular
    vec3 kD = (1.0f - F) * (1.0f - metalness);
    col = emissive + specular + kD * diffuse;  // (1.0-metalness) * (1.0-fresnel(NdotL, F0))*(1.0-fresnel(NdotV, F0));

  } else {
    col = getEnvironment(ray.direction);
  }

  return col;
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
      // Tonemapping
      col *= 1.0f - vec3{expf(-col.r), expf(-col.g), expf(-col.b)};
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
  while ((opt = getopt(argc, argv, " w:h:s:b:t:p:a")) != -1) {
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
        std::cout << "Specify -w width, -h height, -s samples, -t threads or -b bounces -p preset[0, 2]\n";
        std::cout << "Use -a to render BVH heat map (only main ray, single sample, no jitter)\n";
        break;
    }

    break;
  }

  std::cout << "Dimensions: [" << width << ", " << height << "]\tSamples: " << samples
            << "\tBounces: " << bounces << "\tThreads: " << numThreads << std::endl;

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