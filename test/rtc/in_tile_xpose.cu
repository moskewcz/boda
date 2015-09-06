extern "C"  __global__ void %(rtc_func_name)( float const * const in, float * const out ) {
  int32_t const out_ix = blockDim.x * blockIdx.x + threadIdx.x;
  if( out_ix >= %(out_ix_sz) ) { return; }

  int32_t const out_line = %(out_ix_blk_bline)*%(tix_pels_tile_sz);

  int32_t const fi_skip_in_lines = %(out_line_y)*%(stride); 
  int32_t const in_line = (%(out_ix_blk_y)+fi_skip_in_lines);

  int32_t const img_in_lines = (%(out_line_y_dim) - 1)*%(stride) + %(kern_sz);

  int32_t const img_off = in_line/img_in_lines;
  int32_t const img = %(out_line_img) + img_off;

  int32_t const iy = (in_line %% img_in_lines) - %(in_pad); //%(out_line_y)*%(stride) + %(out_ix_blk_y) - %(in_pad);
  int32_t const ix = %(out_ix_blk_bx)*%(t_tile_sz)*%(stride) + %(out_ix_blk_x) - %(in_pad);
  
  float v = 0.0f;
  if(  1 
       && ( ix >= 0 )
       && ( iy >= 0 )
       && ( ix < %(in_ix_x_dim) )
       && ( iy < %(in_ix_y_dim) )
       && ( img < %(in_ix_img_dim) )
       )
  {
    v = in[ img*%(in_ix_img_sz) +
	    %(out_ix_blk_in_chan)*%(in_ix_chan_sz) +
	    iy*%(in_ix_y_sz) +
	    ix*%(in_ix_x_sz) ];
  }
  out[out_ix] = v;
}
