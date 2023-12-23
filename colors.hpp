#pragma once

#include "dataTypes.hpp"

// Colour gradients taken from https://www.shadertoy.com/view/Nd3fR2

// makes afmhot colormap with polynimal 6
vec3 afmhot(float t);

// makes hsv colormap with polynimal 6
vec3 hsv(float t);

// makes viridis colormap with polynimal 6
vec3 viridis(float t);

// makes CMRmap colormap with polynimal 6
vec3 CMRmap(float t);

// makes coolwarm colormap with polynimal 6
vec3 coolwarm(float t);
