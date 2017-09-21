#version 330 core

// Input vertex data, different for all executions of this shader.
//layout(location = 0) in float pt_dist;

// Output data ; will be interpolated for each fragment.
out vec3 fragmentColor;
// Values that stay constant for the whole mesh.
uniform mat4 MVP;

uniform samplerBuffer lut_tex;       

#if 0 // for refernece only
struct obj_t {
       float range;
       float velocity;
       float azi;
       float elev,
       float snr;
       float power;
 };
#endif

#define SIZEOF_OBJ 6
#define RANGE 0
#define AZI 2
#define ELEV 3
#define SNR 4

void main(){	
// Output position of the vertex, in clip space : MVP * position
  vec3 pos;

  float azi_ang = texelFetch(lut_tex, int(gl_VertexID)*SIZEOF_OBJ + AZI ).r;
  float elev_ang = texelFetch(lut_tex, int(gl_VertexID)*SIZEOF_OBJ + ELEV ).r;
  float dist = texelFetch(lut_tex, int(gl_VertexID)*SIZEOF_OBJ + RANGE ).r;
  float snr = texelFetch(lut_tex, int(gl_VertexID)*SIZEOF_OBJ + SNR ).r;
  float sin_azi = sin(azi_ang);
  float cos_azi = cos(azi_ang);
  float sin_elev = sin(elev_ang);
  float cos_elev = cos(elev_ang);
  
  float dist_xy = dist * cos_elev; // elev 0 --> dist_xy = dist
  pos[0] = - dist_xy * sin_azi; // azi 0 --> x = 0; y = dist_xy
  pos[1] = dist_xy * cos_azi;
  pos[2] = dist * sin_elev;

  gl_Position =  MVP * vec4(pos,1);
  fragmentColor = vec3(snr/30.0, snr/30.0, snr/30.0);

  gl_PointSize = 10.;
}

