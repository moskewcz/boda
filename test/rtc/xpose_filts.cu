extern "C"  __global__ void %(cu_func_name)( float const * const in, float * const out ) {
  uint32_t const tix = blockDim.x * blockIdx.x + threadIdx.x;
  if( tix >= %(filts_ix_sz) ) { return; }
  out[tix] = in[tix];
}

