#include "mesh.hpp"

#include <iostream>

Mesh::Mesh(const Geometry& geometry, Material material) : geometry{geometry}, material{material} {
  update();
}

void Mesh::update() {
  invModelMatrix = inverse(modelMatrix);
  normalMatrix = transpose(invModelMatrix);
  transformedAABB.min = modelMatrix * vec4(geometry.bvh[0].aabbMin, 1.0f);
  transformedAABB.max = modelMatrix * vec4(geometry.bvh[0].aabbMax, 1.0f);
}

void Mesh::translate(vec3 t) {
  modelMatrix = glm::translate(modelMatrix, t);
  update();
}

void Mesh::rotateX(float angle) {
  modelMatrix = rotate(modelMatrix, radians(angle), vec3(1, 0, 0));
  update();
}

void Mesh::rotateY(float angle) {
  modelMatrix = rotate(modelMatrix, radians(angle), vec3(0, 1, 0));
  update();
}

void Mesh::rotateZ(float angle) {
  modelMatrix = rotate(modelMatrix, radians(angle), vec3(0, 0, 1));
  update();
}

void Mesh::scale(float scale) {
  modelMatrix = glm::scale(modelMatrix, vec3{scale});
  update();
}

void Mesh::center() {
  modelMatrix = glm::translate(modelMatrix, -geometry.center);
  update();
}