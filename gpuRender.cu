#include <algorithm>
#include <chrono>

#include "brdf.hpp"
#include "error.hpp"
#include "gpuRender.hpp"
#include "random.hpp"

#define INV2PI (1.0f / (2.0f * M_PI))
#define INVPI (1.0f / M_PI)

__device__ vec2 GPUGeometry::getTexCoord(uint idx, vec2 barycentric) const {
  if (hasTexCoords) {
    vec2 v0 = texCoords[3u * idx];
    vec2 v1 = texCoords[3u * idx + 1u];
    vec2 v2 = texCoords[3u * idx + 2u];
    return barycentric.x * v1 + barycentric.y * v2 + (1.0f - (barycentric.x + barycentric.y)) * v0;
  }

  return vec2{0};
}

__device__ vec3 GPUGeometry::getNormal(uint idx, vec2 barycentric) const {
  if (hasNormals) {
    vec3 v0 = vertexNormals[3u * idx];
    vec3 v1 = vertexNormals[3u * idx + 1u];
    vec3 v2 = vertexNormals[3u * idx + 2u];
    return barycentric.x * v1 + barycentric.y * v2 + (1.0f - (barycentric.x + barycentric.y)) * v0;
  }

  return faceNormals[idx];
}

// Find the distance to the closest intersection, the index of the primitive and the number of BVH tests.
__device__ void GPUGeometry::intersect(Ray& ray, HitRecord& hitRecord, uint& count) const {
  intersectBVH(ray, bvh, primitives, indices, 0, hitRecord, count);
}

__device__ void GPUMesh::intersect(Ray& ray, HitRecord& hitRecord, uint& count) const {
  Ray transformedRay = ray;
  transformedRay.origin = invModelMatrix * vec4(ray.origin, 1.0f);
  // Not normalized to handle scale transform
  transformedRay.direction = invModelMatrix * vec4(ray.direction, 0.0f);
  transformedRay.invDirection = 1.0f / transformedRay.direction;

  geometry->intersect(transformedRay, hitRecord, count);
  ray.t = transformedRay.t;
}

//-------------------------------- Rotations --------------------------------

__device__ inline vec3 rotate(vec3 p, vec4 q) {
  return 2.0f * cross(vec3(q), p * q.w + cross(vec3(q), p)) + p;
}
__device__ inline vec3 rotateX(vec3 p, float angle) {
  return rotate(p, vec4(sin(angle / 2.0), 0.0, 0.0, cos(angle / 2.0)));
}
__device__ inline vec3 rotateY(vec3 p, float angle) {
  return rotate(p, vec4(0.0, sin(angle / 2.0), 0.0, cos(angle / 2.0)));
}
__device__ inline vec3 rotateZ(vec3 p, float angle) {
  return rotate(p, vec4(0.0, 0.0, sin(angle / 2.0), cos(angle / 2.0)));
}

__device__ vec3 getEnvironment(const GPUImage* environment, const vec3& direction) {
  // Rotate environment map
  vec3 sampleDir = normalize(rotateY(direction, -M_PI));
  uint u = environment->width * (atan2f(sampleDir.z, sampleDir.x) * INV2PI + 0.5f);
  uint v = environment->height * acosf(sampleDir.y) * INVPI;
  uint idx = min(u + v * environment->width, (environment->width * environment->height) - 1);

  return 0.5f * (*environment)[idx];
}

__device__ vec3 getIllumination(Ray& ray, const GPUScene* scene, const GPUImage* environment,
                                uint& rngState, int& bounceCount, uint& testCount) {
  // Initialize light to white and track attenuation
  vec3 col{1};

  for (uint i = 0; i < bounceCount; i++) {
    HitRecord closestHit{};
    uint meshIdx = scene->intersect(ray, closestHit, testCount);

    if (ray.t < FLT_MAX) {
      const GPUMesh& mesh{scene->meshes[meshIdx]};

      vec3 p = ray.origin + ray.direction * ray.t;
      vec3 N = normalize(vec3(mesh.normalMatrix * vec4(mesh.geometry->getNormal(closestHit.hitIndex, closestHit.barycentric), 0.0f)));
      if (dot(ray.direction, N) > 0.0f) {
        N = -N;
      }

      const float metalness{mesh.material->metalness};
      const float roughness{mesh.material->roughness};
      vec2 uv = mesh.geometry->getTexCoord(closestHit.hitIndex, closestHit.barycentric);
      const vec3 albedo = mesh.material->getAlbedo(uv);
      const vec3 emissive = mesh.material->getEmissive(uv);

      vec3 F0 = mix(mesh.material->F0, albedo, metalness);

      vec3 sampleDir{};
      vec3 localCol{};

      //--------------------- Specular ------------------------

      vec2 Xi = getRandomVec2(rngState);
      // Get a random halfway vector around the surface normal (in world space)
      vec3 H = importanceSampleGGX(Xi, N, roughness);

      vec3 V = -ray.direction;

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

      if (metalness < 1.0f && getRandomFloat(rngState) > F.x) {
        //--------------------- Diffuse ------------------------
        vec2 Xi = getRandomVec2(rngState);
        vec3 kD = (1.0f - F) * (1.0f - metalness);

        sampleDir = mix(sampleDir, importanceSampleCosine(Xi, N), kD.x);

        /*
            The discrete Riemann sum for the lighting equation is
            1/N * Σ(brdf(l, v) * L(l) * dot(l, n)) / pdf(l))
            Lambertian BRDF is c/PI and the pdf for cosine sampling is dot(l, n)/PI
            PI term and dot products cancel out leaving just c * L(l)
        */
        localCol += kD * albedo;
      }

      col *= localCol;
      col += emissive;
      ray = Ray{p + 1e-4f * N, sampleDir, 1.0f / sampleDir, FLT_MAX};
    } else {
      col *= getEnvironment(environment, ray.direction);
      return col;
    }
  }

  return vec3(0);
}

__global__ void printTLAS(GPUScene* scene, uint size) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 1) {
    return;
  }
  Ray ray{};
  HitRecord closestHit{};
  uint bvhTests{0u};

  scene->intersect(ray, closestHit, bvhTests);
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
    // Tonemapping
    col *= 1.0f - vec3{expf(-col.r), expf(-col.g), expf(-col.b)};
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

  // Determine number of threads and blocks covering all pixels
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(ceil((float)(gpuImage.width) / (float)(threadsPerBlock.x)),
                 ceil((float)(gpuImage.height) / (float)(threadsPerBlock.y)));

  // printTLAS<<<dim3(10), dim3(1)>>>(sceneDevicePtr, 10);
  // cudaDeviceSetLimit(cudaLimitStackSize, 1e4);

  /* Timer */ auto start = std::chrono::steady_clock::now();

  /* Call Kernel */ render<<<numBlocks, threadsPerBlock>>>(sceneDevicePtr, camera, imageDevicePtr, environmentDevicePtr, samples, maxBounces, renderBVH);

  CHECK_LAST_CUDA_ERROR();

  cudaDeviceSynchronize();

  /* Timer */ std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - start;
  /* Timer */ std::cout << "\nRender time: " << std::floor(elapsed_seconds.count() * 1e4f) / 1e4f << " s\n";

  // Copy data back to host
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
}