#include "colors.hpp"

// Colour gradients taken from https://www.shadertoy.com/view/Nd3fR2

// makes afmhot colormap with polynimal 6
vec3 afmhot(float t) {
  const vec3 c0 = vec3(-0.020390, 0.009557, 0.018508);
  const vec3 c1 = vec3(3.108226, -0.106297, -1.105891);
  const vec3 c2 = vec3(-14.539061, -2.943057, 14.548595);
  const vec3 c3 = vec3(71.394557, 22.644423, -71.418400);
  const vec3 c4 = vec3(-152.022488, -31.024563, 152.048692);
  const vec3 c5 = vec3(139.593599, 12.411251, -139.604042);
  const vec3 c6 = vec3(-46.532952, -0.000874, 46.532928);
  return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), 0.0f, 1.0f);
}

// makes hsv colormap with polynimal 6
vec3 hsv(float t) {
  const vec3 c0 = vec3(0.834511, -0.153764, -0.139860);
  const vec3 c1 = vec3(8.297883, 13.629371, 7.673034);
  const vec3 c2 = vec3(-80.602944, -80.577977, -90.865764);
  const vec3 c3 = vec3(245.028545, 291.294154, 390.181844);
  const vec3 c4 = vec3(-376.406597, -575.667879, -714.180803);
  const vec3 c5 = vec3(306.639709, 538.472148, 596.580595);
  const vec3 c6 = vec3(-102.934273, -187.108098, -189.286489);
  return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), 0.0f, 1.0f);
}

// makes viridis colormap with polynimal 6
vec3 viridis(float t) {
  const vec3 c0 = vec3(0.274344, 0.004462, 0.331359);
  const vec3 c1 = vec3(0.108915, 1.397291, 1.388110);
  const vec3 c2 = vec3(-0.319631, 0.243490, 0.156419);
  const vec3 c3 = vec3(-4.629188, -5.882803, -19.646115);
  const vec3 c4 = vec3(6.181719, 14.388598, 57.442181);
  const vec3 c5 = vec3(4.876952, -13.955112, -66.125783);
  const vec3 c6 = vec3(-5.513165, 4.709245, 26.582180);
  return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), 0.0f, 1.0f);
}

// makes CMRmap colormap with polynimal 6
vec3 CMRmap(float t) {
  const vec3 c0 = vec3(-0.046981, 0.001239, 0.005501);
  const vec3 c1 = vec3(4.080583, 1.192717, 3.049337);
  const vec3 c2 = vec3(-38.877409, 1.524425, 20.200215);
  const vec3 c3 = vec3(189.038452, -32.746447, -140.774611);
  const vec3 c4 = vec3(-382.197327, 95.587531, 270.024592);
  const vec3 c5 = vec3(339.891791, -100.379096, -212.471161);
  const vec3 c6 = vec3(-110.928480, 35.828481, 60.985694);
  return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), 0.0f, 1.0f);
}

// makes coolwarm colormap with polynimal 6
vec3 coolwarm(float t) {
  const vec3 c0 = vec3(0.227376, 0.286898, 0.752999);
  const vec3 c1 = vec3(1.204846, 2.314886, 1.563499);
  const vec3 c2 = vec3(0.102341, -7.369214, -1.860252);
  const vec3 c3 = vec3(2.218624, 32.578457, -1.643751);
  const vec3 c4 = vec3(-5.076863, -75.374676, -3.704589);
  const vec3 c5 = vec3(1.336276, 73.453060, 9.595678);
  const vec3 c6 = vec3(0.694723, -25.863102, -4.558659);
  return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), 0.0f, 1.0f);
}
