
#include "image.hpp"

Image::Image(uint width, uint height, const std::vector<vec3>& data) : width{width}, height{height}, data{data} {}
Image::Image(uint width, uint height) : width{width}, height{height}, data{std::vector<vec3>{width * height}} {}
