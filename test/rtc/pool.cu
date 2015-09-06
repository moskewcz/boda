extern "C"  __global__ void %(rtc_func_name)( float const * const in, float * const out ) {
  int32_t const out_ix = blockDim.x * blockIdx.x + threadIdx.x;
  if( out_ix >= %(out_ix_sz) ) { return; }
  float out_v = 0.0f;
  for( int32_t kx = 0; kx != %(kern_sz); ++kx ) {
    for( int32_t ky = 0; ky != %(kern_sz); ++ky ) {
      float v = 0;
      int const in_ix_y = %(out_ix_y)*%(stride) + ky - %(in_pad);
      int const in_ix_x = %(out_ix_x)*%(stride) + kx - %(in_pad);
      if(in_ix_y >= 0 && in_ix_x >= 0 && in_ix_x < %(in_ix_x_dim) && in_ix_y < %(in_ix_y_dim) ) {
	int32_t const in_ix = %(out_ix_img)*%(in_ix_img_sz) + %(out_ix_chan)*%(in_ix_chan_sz) + 
	  in_ix_y*%(in_ix_y_sz) + in_ix_x*%(in_ix_x_sz);
	v = in[in_ix];
      }
      %(op);
    }
  }
  %(op_post);
  out[out_ix] = out_v;
}
