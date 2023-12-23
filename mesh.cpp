#include "mesh.hpp"

#include <iostream>

Mesh::Mesh(std::shared_ptr<Geometry> geometry, std::shared_ptr<Material> material) : geometry{geometry}, material{material} {
  update();
}

void Mesh::intersect(Ray& ray, HitRecord& hitRecord, uint& count) const {
  Ray transformedRay = ray;
  transformedRay.origin = invModelMatrix * vec4(ray.origin, 1.0f);
  // Not normalized to handle scale transform
  transformedRay.direction = invModelMatrix * vec4(ray.direction, 0.0f);
  transformedRay.invDirection = 1.0f / transformedRay.direction;

  geometry->intersect(transformedRay, hitRecord, count);
  ray.t = transformedRay.t;
}

void Mesh::update() {
  modelMatrix = identity<mat4>();

  modelMatrix = glm::scale(modelMatrix, vec3(scale));
  modelMatrix = glm::rotate(modelMatrix, rotation.x, vec3(1, 0, 0));
  modelMatrix = glm::rotate(modelMatrix, rotation.y, vec3(0, 1, 0));
  modelMatrix = glm::rotate(modelMatrix, rotation.z, vec3(0, 0, 1));

  modelMatrix = glm::translate(modelMatrix, vec3(inverse(modelMatrix) * vec4(translation, 1.0f)));
  modelMatrix = glm::translate(modelMatrix, -geometry->centroid);

  invModelMatrix = inverse(modelMatrix);
  normalMatrix = transpose(invModelMatrix);

  aabbMin = vec3{FLT_MAX};
  aabbMax = vec3{-FLT_MAX};

  for (const auto& corner : geometry->corners) {
    aabbMin = min(aabbMin, vec3(modelMatrix * vec4(corner, 1.0f)));
    aabbMax = max(aabbMax, vec3(modelMatrix * vec4(corner, 1.0f)));
  }

  centroid = aabbMin + 0.5f * (aabbMax - aabbMin);
}

void Mesh::setPosition(vec3 t) {
  translation = t;
  update();
}

void Mesh::setRotationX(float angle) {
  rotation.x = angle;
  update();
}

void Mesh::setRotationY(float angle) {
  rotation.y = angle;
  update();
}

void Mesh::setRotationZ(float angle) {
  rotation.z = angle;
  update();
}

void Mesh::setScale(float scale) {
  this->scale = scale;
  update();
}

void Mesh::center() {
  translation = -geometry->centroid;
  update();
}

vec3 Mesh::getMin() const {
  return aabbMin;
}

vec3 Mesh::getMax() const {
  return aabbMax;
}