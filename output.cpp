#include "output.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

using namespace glm;

vec3 saturate(const vec3& v) {
  return vec3{
      std::clamp(v.r, 0.0f, 1.0f),
      std::clamp(v.g, 0.0f, 1.0f),
      std::clamp(v.b, 0.0f, 1.0f)};
}

void outputToFile(const Image& image) {
  std::vector<uint8_t> bmpData(image.width * image.height * 3);

  uint outIdx = 0;

  for (int i = 0; i < image.height; i++) {
    for (int j = 0; j < image.width; j++) {
      uint idx = (image.height - 1u - i) * image.width + j;

      vec3 data = saturate(image[idx]);

      bmpData[outIdx++] = (uint8_t)(data.r * 255.0);
      bmpData[outIdx++] = (uint8_t)(data.g * 255.0);
      bmpData[outIdx++] = (uint8_t)(data.b * 255.0);
    }
  }

  stbi_write_bmp("output.bmp", image.width, image.height, 3, bmpData.data());
}
