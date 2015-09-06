extern "C"  __global__ void %(rtc_func_name)( float * const out ) {
  int32_t const ix = blockDim.x * blockIdx.x + threadIdx.x;
  if( ix < %(out_sz) ) { out[ix] = (out[ix] <= 0) ? 0.0f : out[ix]; }
}
