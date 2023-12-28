#include "camera.hpp"
#include "dataTypes.hpp"
#include "image.hpp"

__global__ void getIllumination(Camera camera, const GPUImage& image);
__global__ void getIllumination(Camera camera, GPUImage image, vec3* data);
__global__ void getIllumination();