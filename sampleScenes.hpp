#include "dataTypes.hpp"
#include "geometry.hpp"
#include "material.hpp"
#include "scene.hpp"

enum SampleScene { THREE_STL,
                   SCATTER_STL,
                   OBJ };

void getScene(SampleScene sampleScene,
              Scene& scene,
              std::vector<std::shared_ptr<Geometry>>& geometryPool,
              std::vector<std::shared_ptr<Material>>& materialPool,
              Camera& camera);