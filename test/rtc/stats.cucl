extern "C"  __global__ void %(rtc_func_name)( %(params), int32_t const in_sz, int32_t const primary_in ) {
  int32_t const tbp = 256;
  int32_t const tid = threadIdx.x;
  int32_t const ix = blockDim.x * blockIdx.x + tid;
  %(body)
}
