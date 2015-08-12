extern "C"  __global__ void %(cu_func_name)( float const * const in, float * const out ) {
  int32_t const out_ix = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t const iy = %(out_ix_blk_by)*%(tix_pels_tile_sz)*%(stride) + %(out_ix_blk_y) - %(in_pad);
  int32_t const ix = %(out_ix_blk_bx)*%(t_tile_sz)*%(stride) + %(out_ix_blk_x) - %(in_pad);
  float v = 0.0f;
  if(  1 
       && ( ix >= 0 )
       && ( iy >= 0 )
       && ( ix < %(in_ix_x_dim) )
       && ( iy < %(in_ix_y_dim) )
       )
  {
    v = in[ %(out_ix_blk_img)*%(in_ix_img_sz) +
	    %(out_ix_blk_in_chan)*%(in_ix_chan_sz) +
	    iy*%(in_ix_y_sz) +
	    ix*%(in_ix_x_sz) ];
  }
  out[out_ix] = v;
}
