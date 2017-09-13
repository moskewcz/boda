#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in float pt_dist;

// Output data ; will be interpolated for each fragment.
out vec3 fragmentColor;
// Values that stay constant for the whole mesh.
uniform mat4 MVP;

uniform uint hbins;
uniform uint lasers;

uniform samplerBuffer lut_tex;       

vec3 hsv2rgb(vec3 c) {
   vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
   vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
   return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

struct laser_corr_t {
    float vert_corr;
    float rot_corr;
    float dist_corr;    
    float dist_corr_x;
    float dist_corr_y;
    float off_corr_y;
    float off_corr_x;
    float focal_dist;
    float focal_slope;
 };
#define SIZEOF_LC 9
#define OFF_LC_VERT 0
#define OFF_LC_ROT 1

void main(){	
  // Output position of the vertex, in clip space : MVP * position
  vec3 pos;
  uint laser_id = uint(gl_VertexID) / hbins;
  uint hbin = uint(gl_VertexID) % hbins;
  //float elev_ang = radians(-24.8) + float(laser_id) * radians(0.5);     
  float elev_ang = radians(texelFetch(lut_tex, int(laser_id)*SIZEOF_LC + OFF_LC_VERT ).r);
  float azi_ang = (float(hbin) - float(hbins)/2.0)*radians(0.172) + radians(texelFetch(lut_tex, int(laser_id)*SIZEOF_LC + OFF_LC_ROT ).r);
  //float dist = 50.;
  float dist = pt_dist / 500.;
  float sin_azi = sin(azi_ang);
  float cos_azi = cos(azi_ang);
  float sin_elev = sin(elev_ang);
  float cos_elev = cos(elev_ang);
  
  float dist_xy = dist * cos_elev; // elev 0 --> dist_xy = dist
  pos[0] = dist_xy * sin_azi; // azi 0 --> x = 0; y = dist_xy
  pos[1] = dist_xy * cos_azi;
  pos[2] = dist * sin_elev + 2.;
  //pos[2] = pt_dist / 500. / 10.;
  //pos[2] = laser_id;
  //pos[2] = texelFetch(lut_tex, gl_VertexID % 100 ).r;

  gl_Position =  MVP * vec4(pos,1);
  float hue = (-1. + exp(-max(pos[2] - 0.5, 0.) / 1.5)) * 0.7 - 0.33;
  fragmentColor = hsv2rgb(vec3(hue, 0.8, 1.0));


  gl_PointSize = 2.;

}

