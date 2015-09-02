typedef unsigned uint32_t;

kernel void my_dot( global float const * const a, global float const * const b, global float * const c, uint32_t const n ) {
  uint32_t const ix = get_global_id(0);
  if( ix < n ) { c[ix] = a[ix] + b[ix]; }
}
