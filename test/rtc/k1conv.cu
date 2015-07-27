// each thread: computes 8x8 block of out
// loop over k dim
extern "C"  __global__ void %(cu_func_name)( float const * const filts, float const * const biases, float const * const in, float * const out, int32_t const flags ) {
  //int32_t const blk_in_ix_sz = %(threadIdx.x_pels_tile_dim)*%(t_tile_sz);
  __shared__ float filts_smem[%(blk_filt_ix_sz)*%(in_chan_tile)];
  __shared__ float in_smem[%(threadIdx.x_pels_tile_dim)*%(t_tile_sz)*%(in_chan_tile)];
  float out_tile[%(t_tile_sz)*%(t_tile_sz)] = {0.0f}; // tile of output for this thread to compute, stored in registers
  // reg. buffers for one strip each from in and filts of %(t_tile_sz) elements, for the same filts_ix_out_chan_elem
  float filts_strip[%(t_tile_sz)]; // across output chans (stride is blk_filt_ix_sz )
  float in_strip[%(t_tile_sz)]; // segment of input line sufficient for one unrolling of inner loop
  int32_t const blk_filt_ix_base = %(blockIdx.x_out_chan_blk)*%(blk_filt_ix_sz); // index of first out chan
  int32_t blk_in_ix_base = %(blockIdx.x_pels_blk)*%(in_ix_blk_sz) + threadIdx.x;// index of first input pel to load for this thread

  float * const filts_smem_off = filts_smem + %(threadIdx.x_out_chan_tile);
  float * const in_smem_off = in_smem + %(t_tile_sz)*%(threadIdx.x_pels_tile);
  int32_t filts_off = blk_filt_ix_base + %(filts_off_adj); // adj is either 0 or threadIdx.x;
  // iteratate over filter elements
  for( int32_t blk_iter = 0; blk_iter != %(in_ix_blk_iter_dim); ++blk_iter ) {
    __syncthreads();
    %(smem_loads);
    __syncthreads();
    filts_off += %(filts_xp_ix_in_chan_sz)*%(in_chan_tile);
    blk_in_ix_base += %(in_ix_blk_iter_sz);
    %(inner_loop_body);
  }
  if( flags ) { return; }
  // load per-block biases into smem
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
  // load biases into filts_strip
  %(t_tile_bias_loads);
  // add bias to each elem of out_tile[] and store the results to out[]
  %(t_tile_stores);
}

