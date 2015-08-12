extern "C"  __global__ void %(cu_func_name)( float const * const in, float * const out ) {
  int32_t const filts_ix = blockDim.x * blockIdx.x + threadIdx.x;
  if( filts_ix >= %(filts_ix_sz) ) { return; }
  int32_t const fioc = %(filts_ix_out_chan);
  float val = 0.0f;  
  int32_t const filts_xp_ix  = 
    %(fioc_out_chan_blk)*%(filts_xp_ix_out_chan_blk_sz) +
    %(fioc_out_chan_reg)*%(filts_xp_ix_out_chan_reg_sz) +
    %(fioc_out_chan_tile)*%(filts_xp_ix_out_chan_tile_sz) +
    %(filts_ix_in_chan)*%(filts_xp_ix_in_chan_sz) +
    %(filts_ix_y)*%(filts_xp_ix_y_sz) +
    %(filts_ix_x)*%(filts_xp_ix_x_sz);
#if 1
  val = in[filts_ix];
#else
  if( %(filts_ix_in_chan) == 0 ) {
//    if( (%(filts_ix_x) == 5) && (%(filts_ix_y) == 5) ) 
    {
      val = %(filts_ix_x)*100 + %(filts_ix_y);
    }
  }
#endif
  out[filts_xp_ix] = val;
}

