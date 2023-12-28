#include "camera.hpp"
#include "dataTypes.hpp"
#include "image.hpp"

__global__ void getIllumination(Camera camera, GPUImage* image);