#include <string>
#include <vector>

#include "dataTypes.hpp"

using namespace glm;

Image loadImage(std::string path);
Image loadEnvironmentImage(std::string path);
std::vector<Triangle> loadModel(std::string path);