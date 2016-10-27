typedef unsigned uint32_t;
extern "C" {
  __global__ void my_dot( float const * const a, float const * const b, float * const c, uint32_t const n ) {
    uint32_t const ix = blockDim.x * blockIdx.x + threadIdx.x;
    if( ix < n ) { c[ix] = a[ix] + b[ix]; }
  }

  struct n_t {
    uint32_t n;
  };

  __global__ void my_dot_struct( float const * const a, float const * const b, float * const c, struct n_t const n ) {
    uint32_t const ix = blockDim.x * blockIdx.x + threadIdx.x;
    if( ix < n.n ) { c[ix] = a[ix] + b[ix]; }
  }
}
