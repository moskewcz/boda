extern "C"  __global__ void %(cu_func_name)( float const * const filts, float const * const biases, float const * const in, float * const out ) {
  uint32_t const ix = blockDim.x * blockIdx.x + threadIdx.x;
  if( ix < %(out_sz) ) { out[ix] = 37.0f; }
}
