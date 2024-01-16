#include <algorithm>
#include <chrono>

#include "brdf.hpp"
#include "error.hpp"
#include "gpuFunctions.hpp"
#include "gpuRender.hpp"
#include "random.hpp"
#include "tonemapping.hpp"

#define INV2PI (1.0f / (2.0f * M_PI))
#define INVPI (1.0f / M_PI)

__device__ vec3 getEnvironment(const GPUImage* environment, const vec3& direction) {
  // Rotate environment map
  vec3 sampleDir = normalize(rotateY(direction, -M_PI));
  uint u = environment->width * (atan2f(sampleDir.z, sampleDir.x) * INV2PI + 0.5f);
  uint v = environment->height * acosf(sampleDir.y) * INVPI;
  uint idx = min(u + v * environment->width, (environment->width * environment->height) - 1);

  return (*environment)[idx];
}

__device__ vec3 getIllumination(Ray& ray, const GPUScene* scene, const GPUImage* environment,
                                uint& rngState, int& bounceCount, uint& testCount) {
  // Initialize light to white and track attenuation
  vec3 col{1};

  for (uint i = 0; i < bounceCount; i++) {
    __syncwarp();
    HitRecord closestHit{};
    uint meshIdx = scene->intersect(ray, closestHit, testCount);

    if (ray.t < FLT_MAX) {
      const GPUMesh& mesh{scene->meshes[meshIdx]};

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
      col *= getEnvironment(environment, ray.direction);
      break;
    }
  }

  return col;
}

__global__ void render(GPUScene* scene, Camera camera, GPUImage* image, GPUImage* environment,
                       uint samples, int maxBounces, bool renderBVH) {
  vec2 fragCoord = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if (fragCoord.x >= image->width || fragCoord.y >= image->height) {
    return;
  }

  uint idx = (uint)(fragCoord.y) * image->width + (uint)(fragCoord.x);

  vec2 resolution{image->width, image->height};
  vec3 col{0};
  uint rngState{1023u + idx};

  for (uint s = 0u; s < (renderBVH ? 1u : samples); s++) {
    vec2 fC = fragCoord;
    if (!renderBVH && samples > 1u) {
      // Jitter position for antialiasing
      fC += 0.5f * (2.0f * getRandomVec2(rngState) - 1.0f);
    }

    Ray ray{.origin = camera.position};
    ray.direction = rayDirection(resolution, camera.fieldOfView, fC);
    ray.direction = normalize(viewMatrix(camera.position, camera.target, camera.up) * ray.direction);
    ray.invDirection = 1.0f / ray.direction;
    ray.t = FLT_MAX;

    uint bvhTests = 0u;
    if (renderBVH) {
      HitRecord closestHit{};
      // Get number of BVH tests for primary ray
      scene->intersect(ray, closestHit, bvhTests);
      (*image)[idx] = vec3(bvhTests);
    } else {
      // Path trace scene
      int bounces = maxBounces;
      col += getIllumination(ray, scene, environment, rngState, bounces, bvhTests);
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
    (*image)[idx] = col;
  }
}

void renderGPU(
    const Scene& scene,
    const std::vector<std::shared_ptr<Geometry>>& geometryPool,
    const std::vector<std::shared_ptr<Material>>& materialPool,
    const Camera& camera,
    Image& image,
    const Image& environment,
    const uint samples,
    const int maxBounces,
    const bool renderBVH) {
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds{};

  /* Timer */ cudaEventRecord(start);

  // Make copy of render target image
  GPUImage gpuImage{image};
  GPUImage* imageDevicePtr;
  CHECK_CUDA_ERROR(cudaMalloc(&imageDevicePtr, sizeof(GPUImage)));
  CHECK_CUDA_ERROR(cudaMemcpy(imageDevicePtr, &gpuImage, sizeof(GPUImage), cudaMemcpyHostToDevice));

  // Copy environment
  GPUImage gpuEnvironment{environment};
  GPUImage* environmentDevicePtr;
  CHECK_CUDA_ERROR(cudaMalloc(&environmentDevicePtr, sizeof(GPUImage)));
  CHECK_CUDA_ERROR(cudaMemcpy(environmentDevicePtr, &gpuEnvironment, sizeof(GPUImage), cudaMemcpyHostToDevice));

  // While the CPU code has objects managing their meshes and geometries using STL containers (and automatic RAII), the GPU code
  // should have separate pointer handles for all data to allocate and clean up memory correctly.

  // Geometry and material data is cleaned up when the destructors of their objects are called
  // Cannot use STL smart pointers here but Thrust probably has a better solution to avoid raw pointers
  std::vector<GPUGeometry*> gpuGeometryPool;
  gpuGeometryPool.reserve(geometryPool.size());

  std::vector<GPUMaterial*> gpuMaterialPool;
  gpuMaterialPool.reserve(gpuMaterialPool.size());

  for (const auto& geometryPtr : geometryPool) {
    gpuGeometryPool.emplace_back(new GPUGeometry(*geometryPtr));
  }

  for (const auto& materialPtr : materialPool) {
    gpuMaterialPool.emplace_back(new GPUMaterial(*materialPtr));
  }

  // Containers for device pointers handed to GPUMesh objects. These do not manage data
  std::vector<GPUGeometry*> geometryDevicePtr;
  geometryDevicePtr.reserve(geometryPool.size());

  std::vector<GPUMaterial*> materialDevicePtr;
  materialDevicePtr.reserve(gpuMaterialPool.size());

  for (const auto& geometry : gpuGeometryPool) {
    GPUGeometry* gpuGeometryPtr;
    CHECK_CUDA_ERROR(cudaMalloc(&gpuGeometryPtr, sizeof(GPUGeometry)));
    CHECK_CUDA_ERROR(cudaMemcpy(gpuGeometryPtr, geometry, sizeof(GPUGeometry), cudaMemcpyHostToDevice));
    geometryDevicePtr.emplace_back(gpuGeometryPtr);
  }

  for (const auto& material : gpuMaterialPool) {
    GPUMaterial* gpuMaterialPtr;
    CHECK_CUDA_ERROR(cudaMalloc(&gpuMaterialPtr, sizeof(GPUMaterial)));
    CHECK_CUDA_ERROR(cudaMemcpy(gpuMaterialPtr, material, sizeof(GPUMaterial), cudaMemcpyHostToDevice));
    materialDevicePtr.emplace_back(gpuMaterialPtr);
  }

  // The order of geometries and materials in the CPU and GPU pools is the same. Find the matching pointers and connect GPU-side data
  /*
    Create pointers for each GPUGeometry and GPUMaterial (the data already exists on the GPU)
    Find the index match in the two CPU containers
    Assign correct pointers to GPU meshes on creation
  */
  std::vector<GPUMesh> gpuMeshes;
  gpuMeshes.reserve(scene.meshes.size());

  for (const auto& mesh : scene.meshes) {
    uint geometryIdx = std::distance(geometryPool.begin(), std::find(geometryPool.begin(), geometryPool.end(), mesh.geometry));
    uint materialIdx = std::distance(materialPool.begin(), std::find(materialPool.begin(), materialPool.end(), mesh.material));

    gpuMeshes.push_back(GPUMesh(mesh, geometryDevicePtr[geometryIdx], materialDevicePtr[materialIdx]));
  }

  // Copy scene
  GPUScene gpuScene{scene, gpuMeshes};
  GPUScene* sceneDevicePtr;
  CHECK_CUDA_ERROR(cudaMalloc(&sceneDevicePtr, sizeof(GPUScene)));
  CHECK_CUDA_ERROR(cudaMemcpy(sceneDevicePtr, &gpuScene, sizeof(GPUScene), cudaMemcpyHostToDevice));

  /* Timer */ cudaEventRecord(stop);
  /* Timer */ cudaEventSynchronize(stop);
  /* Timer */ cudaEventElapsedTime(&milliseconds, start, stop);
  /* Timer */ std::cout << "\nGPU data transfer time: " << std::floor((milliseconds / 1e3f) * 1e4f) / 1e4f << " s\n";

  // Determine number of threads and blocks covering all pixels
  dim3 threadsPerBlock(8, 8);
  dim3 numBlocks(ceil((float)(gpuImage.width) / (float)(threadsPerBlock.x)),
                 ceil((float)(gpuImage.height) / (float)(threadsPerBlock.y)));

  /* Timer */ cudaEventRecord(start);

  std::cout << "Rendering..." << std::endl;
  /* Call Kernel */ render<<<numBlocks, threadsPerBlock>>>(sceneDevicePtr, camera, imageDevicePtr, environmentDevicePtr, samples, maxBounces, renderBVH);
  CHECK_LAST_CUDA_ERROR();

  /* Timer */ cudaEventRecord(stop);
  /* Timer */ cudaEventSynchronize(stop);
  /* Timer */ cudaEventElapsedTime(&milliseconds, start, stop);
  /* Timer */ std::cout << "\nRender time: " << std::floor((milliseconds / 1e3f) * 1e4f) / 1e4f << " s\n";

  // Copy data back to host. cudaMemcpy does not start until all GPU operations have finished.
  CHECK_CUDA_ERROR(cudaMemcpy(image.data.data(), gpuImage.data, image.data.size() * sizeof(vec3), cudaMemcpyDeviceToHost));

  // Free device pointers
  CHECK_CUDA_ERROR(cudaFree(imageDevicePtr));
  CHECK_CUDA_ERROR(cudaFree(environmentDevicePtr));
  CHECK_CUDA_ERROR(cudaFree(sceneDevicePtr));

  for (const auto& geometryPtr : geometryDevicePtr) {
    CHECK_CUDA_ERROR(cudaFree(geometryPtr));
  }

  for (const auto& materialPtr : materialDevicePtr) {
    CHECK_CUDA_ERROR(cudaFree(materialPtr));
  }

  for (const auto& geometry : gpuGeometryPool) {
    delete geometry;
  }

  for (const auto& material : gpuMaterialPool) {
    delete material;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}