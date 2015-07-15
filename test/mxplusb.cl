kernel void mxplusb(float m, global float *x, float b, global float *out) {
     int i = get_global_id(0);
     out[i] = m*x[i]+b;
}