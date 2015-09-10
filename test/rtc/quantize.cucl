extern "C"  __global__ void %(rtc_func_name)( float * const out, uint32_t const out_sz, uint32_t const max_val, uint32_t const drop_mask ) {
  uint32_t const ix = blockDim.x * blockIdx.x + threadIdx.x;
  if( ix < out_sz ) { 
    int32_t v = out[ix];
    v = max(0,v);
    v = min(max_val,v);
    v &= ~drop_mask;
    out[ix] = v;
  }
}
