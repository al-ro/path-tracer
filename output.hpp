#include <vector>

#include "structs.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

glm::vec3 saturate(glm::vec3 v) {
  return glm::vec3{
      std::clamp(v.r, 0.0f, 1.0f),
      std::clamp(v.g, 0.0f, 1.0f),
      std::clamp(v.b, 0.0f, 1.0f)};
}

void outputToFile(const glm::vec2 resolution, const std::vector<glm::vec3>& target) {
  std::vector<uint8_t> bmpData(resolution.x * resolution.y * 3);

  uint outIdx = 0;

  for (int i = 0; i < resolution.y; i++) {
    for (int j = 0; j < resolution.x; j++) {
      uint idx = (resolution.y - 1u - i) * resolution.x + j;

      glm::vec3 data = saturate(target[idx]);

      bmpData[outIdx++] = (uint8_t)(data.r * 255.0);
      bmpData[outIdx++] = (uint8_t)(data.g * 255.0);
      bmpData[outIdx++] = (uint8_t)(data.b * 255.0);
    }
  }

  stbi_write_bmp("output.bmp", resolution.x, resolution.y, 3, bmpData.data());
}
