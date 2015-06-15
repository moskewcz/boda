// each thread: computes outputs across chan dim, using inputs across chan dim
extern "C"  __global__ void %(cu_func_name)( float const * const in, float * const out ) {
  int32_t const in_ix = blockDim.x * blockIdx.x + threadIdx.x;
  if( in_ix >= %(in_ix_sz) ) { return; }
  int32_t const out_ix = %(in_ix_img)*%(out_ix_img_sz) + (%(in_ix_chan)+%(ocix))*%(out_ix_chan_sz) +
    %(in_ix_y)*%(out_ix_y_sz) + %(in_ix_x)*%(out_ix_x_sz);  
  out[out_ix] = in[in_ix];
}

