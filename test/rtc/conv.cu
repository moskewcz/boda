// 256 tbp
// each thread: computes 8x8 block of out
// loop over k dim
extern "C"  __global__ void %(cu_func_name)( float const * const filts, float const * const biases, float const * const in, float * const out ) {
  __shared__ float in_smem[%(threadIdx.x_patch_tile_dim)*%(t_tile_sz)];
  int32_t const blk_filt_ix_sz = %(threadIdx.x_out_chan_tile_dim)*%(t_tile_sz);
  __shared__ float filts_smem[blk_filt_ix_sz];
  float out_tile[%(t_tile_sz)*%(t_tile_sz)] = {0}; // tile of output for this thread to compute, stored in registers
  // reg. buffers for one strip each from in and filts of %(t_tile_sz) elements, for the same filts_ix_out_chan_elem
  float filts_strip[%(t_tile_sz)]; // across output chans (stride is blk_filt_ix_sz )
  float in_strip[%(t_tile_sz)]; // across patches (approx square block in x/y space, favoring x if sqrt() not integer)
  int32_t const blk_filt_ix_base = %(blockIdx.x_out_chan_blk)*blk_filt_ix_sz;

  int32_t const blk_patch_ix_sz = %(threadIdx.x_patch_tile_dim)*%(t_tile_sz);
  int32_t const blk_patch_ix_base = %(blockIdx.x_patch_blk)*blk_patch_ix_sz;

  // iteratate over filter elements
  int32_t filts_off = blk_filt_ix_base + threadIdx.x;
  for( int32_t filts_ix_out_chan_elem = 0; filts_ix_out_chan_elem != (%(filts_xp_ix_sz) / %(filts_xp_ix_x_sz));
       ++filts_ix_out_chan_elem ) {
    __syncthreads();
    if( threadIdx.x < blk_filt_ix_sz ) { 
#if 0
      filts_smem[threadIdx.x] = threadIdx.x;
      //filts_smem[threadIdx.x] = filts[threadIdx.x];
#else
      filts_smem[threadIdx.x] = filts[filts_off];
#endif
    }
    filts_off += %(filts_xp_ix_x_sz);
    for( int32_t i = 0; i != %(patch_smem_load_iter); ++i ) {
      if( (threadIdx.x+blockDim.x*i) < blk_patch_ix_sz ) { 
	int32_t const t_smem_patch_ix = (blk_patch_ix_base+threadIdx.x+blockDim.x*i);

#if 0
	float v = threadIdx.x;
#else
	%(get_in);
#endif
	in_smem[threadIdx.x+blockDim.x*i] = v;
      }
    }
    __syncthreads();
#if 0
    %(t_tile_dummy_loads);
#else
    %(t_tile_loads);
#endif
    // (2) do %(t_tile_sz)^2 fmas into out_tile
    %(t_tile_fmas);
  }

  // load per-block biases into smem
  __syncthreads();
  if( threadIdx.x < blk_filt_ix_sz ) { 
    int32_t const ocix_base = %(blockIdx.x_out_chan_blk)*blk_filt_ix_sz;
    int32_t const load_reg = threadIdx.x / %(threadIdx.x_out_chan_tile_dim);
    int32_t const load_tile = threadIdx.x %% %(threadIdx.x_out_chan_tile_dim);
    int32_t const ocix = ocix_base + load_tile*%(t_tile_sz) + load_reg;
    if( ocix < %(out_ix_chan_dim) ) { filts_smem[threadIdx.x] = biases[ ocix ]; }
    //int32_t const ocix_tile = (ocix / %(t_tile_sz)) %% %(threadIdx.x_out_chan_tile_dim);
    //int32_t const ocix_reg = ocix %% %(t_tile_sz);
    //filts_smem[ocix_tile * %(filts_xp_ix_out_chan_tile_sz) + ocix_reg * %(filts_xp_ix_out_chan_reg_sz)] = biases[ocix];
  }
  __syncthreads();
  // load biases into filts_strip
  %(t_tile_loads);

  // add bias to each elem of out_tile[] and store the results to out[]
#if 0
  %(t_tile_dummy_stores);
#else
  %(t_tile_stores);
#endif
}

