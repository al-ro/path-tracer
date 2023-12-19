#include "dataTypes.hpp"
#include "mesh.hpp"

struct HitRecord {
  uint hitIndex{UINT32_MAX};
  Ray ray{};
  const Mesh* mesh;
};