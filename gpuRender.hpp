#include "camera.hpp"
#include "dataTypes.hpp"
#include "image.hpp"
#include "scene.hpp"

void renderGPU(const Scene& scene, const Camera& camera, Image& image, const Image& environment,
               const uint samples, const int maxBounces, const bool renderBVH);
