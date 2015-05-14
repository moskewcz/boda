typedef unsigned uint32_t;
extern "C" {
  __global__ void dot( float const * const a, float const * const b, float * const c, uint32_t const n ) {
    uint32_t const ix = blockDim.x * blockIdx.x + threadIdx.x;
    if( ix < n ) { c[ix] = a[ix] + b[ix]; }
  }
}
