#include <string>
#include <vector>

#include "dataTypes.hpp"

using namespace glm;

std::vector<vec3> loadImage(std::string path, vec2& resolution);
std::vector<Triangle> loadModel(std::string path);