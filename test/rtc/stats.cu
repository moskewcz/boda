extern "C"  __global__ void %(cu_func_name)( %(params), uint32_t const in_sz ) {
  uint32_t const tbp = 256;
  uint32_t const tid = threadIdx.x;
  uint32_t const ix = blockDim.x * blockIdx.x + tid;
  %(body)
}
