// each thread: computes outputs across chan dim, using inputs across chan dim
extern "C"  __global__ void %(rtc_func_name)( float const * const in, float * const out ) {
  int32_t const tix = blockDim.x * blockIdx.x + threadIdx.x;
  if( tix >= %(tix_sz) ) { return; }
  // iteratate over chans
  float ls_buf[%(local_size)] = {0.0f};
  int32_t const hls = %(local_size) >> 1;
  int32_t const out_base_ix = %(tix_img)*%(out_ix_img_sz) + %(tix_y)*%(out_ix_y_sz) + %(tix_x)*%(out_ix_x_sz);  
  for( int32_t in_chan_ix = 0; in_chan_ix < %(out_ix_chan_dim) + hls; ++in_chan_ix ) {
    int32_t const in_off = in_chan_ix*%(out_ix_chan_sz);
    int32_t const lsb_ix = in_chan_ix %% %(local_size);
    ls_buf[lsb_ix] = (in_chan_ix < %(out_ix_chan_dim)) ? in[out_base_ix + in_off] : 0.0f;
    
    if( in_chan_ix >= hls ) {
      int32_t const out_chan_ix = in_chan_ix - hls;
      float ls_sum = 0.0f;
      for( int32_t i = 0; i != %(local_size); ++i ) { ls_sum += ls_buf[i]*ls_buf[i]; }

      float const scale = powf( (%(k) + %(alpha)*ls_sum/%(local_size)), -%(beta) );
      
      out[out_base_ix + out_chan_ix*%(out_ix_chan_sz)] = ls_buf[(lsb_ix+%(local_size)-hls) %% %(local_size)] * scale;
    }
  }
}

