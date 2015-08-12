// each thread: computes 8x8 block of out
// loop over k dim
extern "C"  __global__ void %(cu_func_name)( float const * const filts, float const * const biases, float const * const in, float * const out, int32_t const flags ) {
  __shared__ float all_smem[%(all_smem_sz)]; // note: max(filts+in,out) == max(%(filts_smem_sz)+%(in_smem_sz),%(out_smem_sz))
  float * const filts_smem = all_smem;
  float * const in_smem = filts_smem + %(filts_smem_sz);
  float out_tile[%(t_tile_sz)*%(t_tile_sz)] = {0.0f}; // tile of output for this thread to compute, stored in registers
  // reg. buffers for one strip each from in and filts of %(t_tile_sz) elements, for the same filts_ix_out_chan_elem
  float filts_strip[%(t_tile_sz)]; // across output chans (stride is blk_filt_ix_sz )
  float in_strip[%(in_ix_blk_x_dim)]; // segment of input line sufficient for one unrolling of inner loop

  int32_t blk_in_ix_base = %(blockIdx.x_blk_bx_nomod)*%(in_ix_blk_bx_sz) + threadIdx.x;// index of first input pel to load for this thread

  int32_t const blk_filt_ix_base = %(blockIdx.x_out_chan_blk)*%(filts_xp_ix_out_chan_blk_sz); // index of first out chan
  int32_t filts_off = blk_filt_ix_base + %(filts_off_adj); // adj is either 0 or threadIdx.x;

  float * const filts_smem_off = filts_smem + %(threadIdx.x_out_chan_tile);

  for( int32_t in_chan = 0; in_chan != %(in_ix_blk_in_chan_dim); ++in_chan ) {
    __syncthreads();
    %(in_smem_loads);
    for( int32_t ky = 0; ky != %(kern_sz); ++ky ) {
      if( ky != 0 ) { __syncthreads(); }
      %(filt_smem_loads);
      __syncthreads();
      float * const in_smem_off = in_smem + (%(threadIdx.x_blk_y)*%(stride)+ky)*%(in_ix_blk_y_sz);

      %(inner_loop_body);
    }
  }

  __syncthreads();
  for( int32_t i = 0; i != %(out_chan_bias_smem_load_iter); ++i ) {
    int32_t const t_smem_bias_ix = threadIdx.x+%(tpb)*i;
    if( t_smem_bias_ix < %(blk_filt_ix_sz) ) { 
      int32_t const ocix_base = %(blockIdx.x_out_chan_blk)*%(blk_filt_ix_sz);
      int32_t const load_reg = t_smem_bias_ix / %(threadIdx.x_out_chan_tile_dim);
      int32_t const load_tile = t_smem_bias_ix %% %(threadIdx.x_out_chan_tile_dim);
      int32_t const ocix = ocix_base + load_tile*%(t_tile_sz) + load_reg;
      if( ocix < %(out_ix_chan_dim) ) { filts_smem[t_smem_bias_ix] = biases[ ocix ]; }
    }
  }
  __syncthreads();
  %(t_tile_bias_loads);
  %(t_tile_stores);
}

