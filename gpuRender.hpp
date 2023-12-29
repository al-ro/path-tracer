#include "camera.hpp"
#include "dataTypes.hpp"
#include "image.hpp"

__global__ void render(Camera camera, GPUImage* image, GPUImage* environment);