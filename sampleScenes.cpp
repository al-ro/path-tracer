#include "sampleScenes.hpp"

#include <chrono>

#include "colors.hpp"
#include "input.hpp"
#include "random.hpp"

void getScene(SampleScene sampleScene,
              Scene& scene,
              std::vector<std::shared_ptr<Geometry>>& geometryPool,
              std::vector<std::shared_ptr<Material>>& materialPool,
              Camera& camera) {
  /* Timer */ auto start = std::chrono::steady_clock::now();

  switch (sampleScene) {
    case THREE_STL: {
      // Camera
      camera.position = 200.0f * vec3{-1.0f, 0.2f, 0.05f};
      camera.target = vec3{0};
      camera.up = normalize(vec3{0, 1, 0});
      camera.fieldOfView = 45.0f;

      // Model
      std::vector<Triangle> triangles = loadSTL("models/bust-of-menelaus.stl");

      // Geometry
      geometryPool.emplace_back(std::make_shared<Geometry>(Geometry{triangles}));

      // Material
      materialPool.emplace_back(std::make_shared<Material>(Material{}));
      materialPool[0]->albedo = vec3{1.0, 0.8, 0.6};
      materialPool[0]->metalness = 1.0f;
      materialPool[0]->roughness = 0.05f;

      materialPool.emplace_back(std::make_shared<Material>(Material{}));
      materialPool[1]->albedo = vec3{1.0};

      materialPool.emplace_back(std::make_shared<Material>(Material{}));
      materialPool[2]->albedo = vec3{0.1, 0.7, 0.4};
      materialPool[2]->metalness = 1.0f;
      materialPool[2]->roughness = 0.2f;

      // Mesh
      scene.meshes.emplace_back(Mesh{geometryPool[0], materialPool[0]});
      scene.meshes[0].setRotationX(-0.5f * M_PI);
      scene.meshes[0].setRotationZ(0.0f * M_PI);
      scene.meshes[0].setPosition(vec3{0.0, 0.0, -90.0});
      scene.meshes[0].setScale(0.85f);

      scene.meshes.emplace_back(Mesh{geometryPool[0], materialPool[1]});
      scene.meshes[1].setRotationX(-0.5f * M_PI);
      scene.meshes[1].setRotationZ(0.0f * M_PI);

      scene.meshes.emplace_back(Mesh{geometryPool[0], materialPool[2]});
      scene.meshes[2].setRotationX(-0.5f * M_PI);
      scene.meshes[2].setRotationZ(-0.3f * M_PI);
      scene.meshes[2].setPosition(vec3{0.0, 0.0, 90.0});
      scene.meshes[2].setScale(0.8);

    } break;

    case SCATTER_STL: {
      // Camera
      camera.position = 500.0f * vec3{1.0f, 0.2f, -0.15f};
      camera.target = vec3{0};
      camera.up = normalize(vec3{0, 1, 0});
      camera.fieldOfView = 45.0f;

      // Model
      std::vector<Triangle> triangles = loadSTL("models/bust-of-menelaus.stl");

      // Geometry
      geometryPool.emplace_back(std::make_shared<Geometry>(Geometry{triangles}));

      // Material
      materialPool.emplace_back(std::make_shared<Material>(Material{}));
      materialPool[0]->albedo = vec3{1.0, 0.8, 0.6};
      materialPool[0]->metalness = 1.0f;
      materialPool[0]->roughness = 0.05f;

      materialPool.emplace_back(std::make_shared<Material>(Material{}));
      materialPool[1]->albedo = vec3{1.0};

      uint rngState{7142u};
      uint materialCount = 10u;
      for (uint i = 0; i < materialCount; i++) {
        materialPool.emplace_back(std::make_shared<Material>(Material{}));
        materialPool[i]->albedo = hsv(static_cast<float>(i) / static_cast<float>(materialCount));
        materialPool[i]->metalness = getRandomFloat(rngState) > 0.7f;
      }

      // Mesh
      for (uint i = 0u; i < 10000u; i++) {
        uint materialIdx = floor(getRandomFloat(rngState) * materialPool.size());
        scene.meshes.emplace_back(Mesh{geometryPool[0], materialPool[materialIdx]});
        scene.meshes[i].setRotationX(-0.5f * M_PI);
        scene.meshes[i].setRotationY(2.0f * M_PI * getRandomFloat(rngState));
        scene.meshes[i].setRotationZ(M_PI * getRandomFloat(rngState));
        scene.meshes[i].setScale(0.2f);
        scene.meshes[i].setPosition(700.0f * (2.0f * getRandomVec3(rngState) - 1.0f));
      }

    } break;

    case OBJ: {
      // Camera
      camera.position = 1.0f * vec3{0.5f, 0.35f, -1.0f};
      camera.target = vec3{0};
      camera.up = normalize(vec3{0, 1, 0});
      camera.fieldOfView = 45.0f;

      // Model
      std::vector<vec3> normals{};
      std::vector<vec2> texCoords{};
      std::vector<Triangle> triangles = loadObj("models/viking-room/viking_room.obj", normals, texCoords);
      VertexAttributes attributes{.normals = normals, .texCoords = texCoords};

      // Geometry
      geometryPool.emplace_back(std::make_shared<Geometry>(Geometry{triangles, attributes}));

      // Material
      materialPool.emplace_back(std::make_shared<Material>(Material{}));
      materialPool[0]->albedoTexture = loadImage("models/viking-room/albedo.png");
      materialPool[0]->emissiveTexture = loadImage("models/viking-room/emissive.png");
      materialPool[0]->emissive = vec3{5};

      // Mesh
      scene.meshes.emplace_back(Mesh{geometryPool[0], materialPool[0]});
      scene.meshes[0].setRotationX(-0.5f * M_PI);
      scene.meshes[0].setRotationZ(0.15f * M_PI);

    } break;

    default:
      std::cout << "Scene " << sampleScene << " is not defined\n";
      exit(1);
      return;
  }
  /* Timer */ std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - start;
  /* Timer */ std::cout << "\nObject processing time: " << std::floor(elapsed_seconds.count() * 1e4f) / 1e4f << " s\n";

  scene.completeScene();
}