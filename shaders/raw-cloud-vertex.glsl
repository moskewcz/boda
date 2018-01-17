#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 pt;

// Output data ; will be interpolated for each fragment.
out vec4 fragmentColor;
// Values that stay constant for the whole mesh.
uniform mat4 MVP;

uniform float pt_sz;
uniform uint pt_color_mode;
uniform vec3 pt_const_color;

vec3 hsv2rgb(vec3 c) {
   vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
   vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
   return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main(){	
  // Output position of the vertex, in clip space : MVP * position
  gl_Position =  MVP * vec4(pt,1);
  float hue = (-1. + exp(-max(pt[2] - 0.5, 0.) / 1.5)) * 0.7 - 0.33;
  if( pt_color_mode == 0U ) { fragmentColor = vec4( pt_const_color, 1.0 ); }
  else if( pt_color_mode == 1U ) { fragmentColor = vec4( hsv2rgb(vec3(hue, 0.8, 1.0)), 1.0); }
  gl_PointSize = pt_sz;

}

