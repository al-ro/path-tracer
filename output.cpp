#include "output.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

using namespace glm;

vec3 saturate(const vec3& v) {
  return vec3{
      std::clamp(v.r, 0.0f, 1.0f),
      std::clamp(v.g, 0.0f, 1.0f),
      std::clamp(v.b, 0.0f, 1.0f)};
}

void outputToFile(const vec2& resolution, const std::vector<vec3>& image) {
  std::vector<uint8_t> bmpData(resolution.x * resolution.y * 3);

  uint outIdx = 0;

  for (int i = 0; i < resolution.y; i++) {
    for (int j = 0; j < resolution.x; j++) {
      uint idx = (resolution.y - 1u - i) * resolution.x + j;

      vec3 data = saturate(image[idx]);

      bmpData[outIdx++] = (uint8_t)(data.r * 255.0);
      bmpData[outIdx++] = (uint8_t)(data.g * 255.0);
      bmpData[outIdx++] = (uint8_t)(data.b * 255.0);
    }
  }

  stbi_write_bmp("output.bmp", resolution.x, resolution.y, 3, bmpData.data());
}
