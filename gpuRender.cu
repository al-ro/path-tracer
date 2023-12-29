#include "error.hpp"
#include "gpuRender.hpp"

// The size of the stack for keeping track of BVH nodes to be tested
#define STACK_SIZE 32
#define INV2PI (1.0f / (2.0f * M_PI))
#define INVPI (1.0f / M_PI)

__device__ vec3 getEnvironment(GPUImage* environment, const vec3& direction) {
  uint u = environment->width * (atan2f(direction.z, direction.x) * INV2PI + 0.5f);
  uint v = environment->height * acosf(direction.y) * INVPI;
  uint idx = min(u + v * environment->width, (environment->width * environment->height) - 1);

  return 0.5f * (*environment)[idx];
}

__device__ vec3 getIllumination(int& bounces) {
  bounces--;
  if (bounces <= 0) {
    return vec3{0};
  }
  return getIllumination(bounces);
}

__global__ void render(Camera camera, GPUImage* image, GPUImage* environment) {
  vec2 fragCoord = ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  if (fragCoord.x >= image->width || fragCoord.y >= image->height) {
    return;
  }

  Ray ray{};
  vec2 resolution{image->width, image->height};

  ray.direction = rayDirection(resolution, camera.fieldOfView, fragCoord);
  ray.direction = normalize(viewMatrix(camera.position, camera.target, camera.up) * ray.direction);
  ray.invDirection = 1.0f / ray.direction;
  ray.t = FLT_MAX;
  (*image)[(uint)(fragCoord.y) * image->width + (uint)(fragCoord.x)] = getEnvironment(environment, ray.direction);
}

void renderGPU(
    const Scene& scene,
    const Camera& camera,
    Image& image,
    const Image& environment,
    const uint samples,
    const int maxBounces,
    const bool renderBVH) {
  // Make copy of image
  GPUImage gpuImage{image};
  GPUImage* imageDevicePtr;
  CHECK_CUDA_ERROR(cudaMalloc(&imageDevicePtr, sizeof(GPUImage)));
  CHECK_CUDA_ERROR(cudaMemcpy(imageDevicePtr, &gpuImage, sizeof(GPUImage), cudaMemcpyHostToDevice));

  // Copy environment
  GPUImage gpuEnvironment{environment};
  GPUImage* environmentDevicePtr;
  CHECK_CUDA_ERROR(cudaMalloc(&environmentDevicePtr, sizeof(GPUImage)));
  CHECK_CUDA_ERROR(cudaMemcpy(environmentDevicePtr, &gpuEnvironment, sizeof(GPUImage), cudaMemcpyHostToDevice));

  // Copy scene
  GPUScene gpuScene{scene};
  GPUScene* sceneDevicePtr;
  CHECK_CUDA_ERROR(cudaMalloc(&sceneDevicePtr, sizeof(GPUImage)));
  CHECK_CUDA_ERROR(cudaMemcpy(sceneDevicePtr, &gpuScene, sizeof(GPUImage), cudaMemcpyHostToDevice));

  // Determine number of threads and blocks covering all pixels
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(ceil((float)(gpuImage.width) / (float)(threadsPerBlock.x)),
                 ceil((float)(gpuImage.height) / (float)(threadsPerBlock.y)));

  /* Call Kernel */ render<<<numBlocks, threadsPerBlock>>>(camera, imageDevicePtr, environmentDevicePtr);

  CHECK_LAST_CUDA_ERROR();

  // Free device side pointers
  CHECK_CUDA_ERROR(cudaFree(imageDevicePtr));
  CHECK_CUDA_ERROR(cudaFree(environmentDevicePtr));

  // Copy data back to host
  CHECK_CUDA_ERROR(cudaMemcpy(image.data.data(), gpuImage.data, image.data.size() * sizeof(vec3), cudaMemcpyDeviceToHost));
  /*
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
    */
}