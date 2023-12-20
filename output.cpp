#include "output.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

using namespace glm;

void outputToFile(const Image& image) {
  std::vector<uint8_t> bmpData(image.width * image.height * 3);

  uint outIdx = 0;

  for (int i = 0; i < image.height; i++) {
    for (int j = 0; j < image.width; j++) {
      uint idx = (image.height - 1u - i) * image.width + j;

      vec3 data = clamp(image[idx], 0.0f, 1.0f);

      bmpData[outIdx++] = (uint8_t)(data.r * 255.0);
      bmpData[outIdx++] = (uint8_t)(data.g * 255.0);
      bmpData[outIdx++] = (uint8_t)(data.b * 255.0);
    }
  }

  std::string outputName = "output.bmp";
  stbi_write_bmp(outputName.c_str(), image.width, image.height, 3, bmpData.data());
  std::cout << "Output result to " << outputName << std::endl;
}
