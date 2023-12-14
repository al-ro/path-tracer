#include "dataTypes.h"

float intersect(const Ray& ray, const Triangle& triangle);
float intersect(const Ray& ray, const AABB& aabb);
float intersect(const Ray& ray, const vec3& aabbMin, const vec3& aabbMax);