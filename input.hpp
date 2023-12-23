#include <string>
#include <vector>

#include "dataTypes.hpp"
#include "image.hpp"

using namespace glm;

Image loadImage(std::string path);
Image loadEnvironmentImage(std::string path);

std::vector<Triangle> loadSTL(std::string path);

std::vector<Triangle> loadObj(std::string path,
                              std::vector<vec3>& normals,
                              std::vector<vec2>& uvs);