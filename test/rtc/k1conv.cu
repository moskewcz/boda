// each thread: computes 8x8 block of out
// loop over k dim
extern "C"  __global__ void %(cu_func_name)( float const * const filts, float const * const biases, float const * const in, float * const out, int32_t const flags ) {
  __shared__ float in_smem[%(line_buf_sz)*%(threadIdx.x_line_dim)*%(in_chan_tile)];
#if (%(in_pad) > 0)
  // zero init padding part of in_smem
  if( threadIdx.x < ( 2*%(in_pad)*%(threadIdx.x_line_dim)*%(in_chan_tile) ) ) {
    int32_t const pad_ix = threadIdx.x %% (2*%(in_pad));
    int32_t const line_off = (threadIdx.x / (2*%(in_pad))) * %(line_buf_sz); 
    in_smem[ line_off + pad_ix + ((pad_ix < %(in_pad)) ? 0 : %(in_ix_x_dim))] = 0.0f; 
  }  
#endif
  int32_t const blk_filt_ix_sz = %(threadIdx.x_out_chan_tile_dim)*%(t_tile_sz);
  __shared__ float filts_smem[blk_filt_ix_sz*%(in_chan_tile)];
  float out_tile[%(t_tile_sz)*%(t_tile_sz)] = {0}; // tile of output for this thread to compute, stored in registers
  // reg. buffers for one strip each from in and filts of %(t_tile_sz) elements, for the same filts_ix_out_chan_elem
  float filts_strip[%(t_tile_sz)]; // across output chans (stride is blk_filt_ix_sz )
  float in_strip[%(t_tile_sz)]; // segment of input line sufficient for one unrolling of inner loop
  int32_t const blk_filt_ix_base = %(blockIdx.x_out_chan_blk)*blk_filt_ix_sz; // index of first out chan

  // iteratate over filter elements
  int32_t filts_smem_off = 0;
  int32_t in_smem_off = 0;
  int32_t filts_off = blk_filt_ix_base + %(filts_off_adj); // adj is either 0 or threadIdx.x;

  int32_t do_load_bits = 0;
  int32_t in_off[%(in_smem_load_iter)];
  int32_t t_smem_ix[%(in_smem_load_iter)];
#pragma unroll
  for( int32_t i = 0; i < %(in_smem_load_iter); ++i ) {   
    int32_t const t_smem_ld_pel = threadIdx.x + i * blockDim.x; // may need loop
    //int32_t const t_smem_line = threadIdx.x / %(in_ix_x_dim);
    //int32_t const t_smem_line_x = threadIdx.x %% %(in_ix_x_dim);
    t_smem_ix[i] = %(t_smem_ld_pel_line_nomod)*%(line_buf_sz)+%(in_pad)+%(t_smem_ld_pel_x);
    // note: this out_line is for this thread's smem reading, not this thread's calc
    int32_t out_line = %(blockIdx.x_lines_blk)*%(threadIdx.x_line_dim) + %(t_smem_ld_pel_line);
    int32_t in_line = %(out_line_y) - %(in_pad);
    // since ky_sz == 1, in_line and thus do_load are constant per-thread. also we can thus include in_line in in_off
    do_load_bits |= int32_t(bool( ( in_line >= 0 && in_line < %(in_ix_y_dim) ) && ( %(out_line_img) < %(in_ix_img_dim) ) ) ) << i;
    in_off[i] = %(out_line_img)*%(in_ix_img_sz) + %(t_smem_ld_pel_chan)*%(in_ix_chan_sz) +
      %(t_smem_ld_pel_x)*%(in_ix_x_sz) + in_line*%(in_ix_y_sz);
  }
  int32_t in_chan_off = 0;
  for( int32_t filts_ix_out_chan_elem = 0; filts_ix_out_chan_elem != %(filts_ix_out_chan_elem_sz); ++filts_ix_out_chan_elem ) {
    __syncthreads();
    %(filts_smem_loads);
    filts_off += %(filts_xp_ix_in_chan_sz)*%(in_chan_tile);

#pragma unroll
    for( int32_t i = 0; i < %(in_smem_load_iter); ++i ) {   
      int32_t const t_smem_ld_pel = threadIdx.x + i * blockDim.x; // may need loop
      if( t_smem_ld_pel < %(t_smem_ld_pel_sz) ) { 
	float v;
	if( do_load_bits&(1<<i) ) { v = in[ in_off[i] + in_chan_off ]; }
	else { v = 0.0f; }
	in_smem[t_smem_ix[i]] = v;
      }
    }
    __syncthreads();
    %(inner_loop_body);
    in_chan_off += %(in_ix_chan_sz)*%(in_chan_tile); 
  }
  // load per-block biases into smem
  __syncthreads();
  filts_smem_off = 0;
  for( int32_t i = 0; i != %(out_chan_bias_smem_load_iter); ++i ) {
    int32_t const t_smem_bias_ix = threadIdx.x+blockDim.x*i;
    if( t_smem_bias_ix < blk_filt_ix_sz ) { 
      int32_t const ocix_base = %(blockIdx.x_out_chan_blk)*blk_filt_ix_sz;
      int32_t const load_reg = t_smem_bias_ix / %(threadIdx.x_out_chan_tile_dim);
      int32_t const load_tile = t_smem_bias_ix %% %(threadIdx.x_out_chan_tile_dim);
      int32_t const ocix = ocix_base + load_tile*%(t_tile_sz) + load_reg;
      if( ocix < %(out_ix_chan_dim) ) { filts_smem[filts_smem_off+t_smem_bias_ix] = biases[ ocix ]; }
    }
  }
  __syncthreads();
  // load biases into filts_strip
  %(t_tile_filt_loads);
  // note: this out_line is for this thread's calculation/output region, used to guard writes
  int32_t out_line = %(blockIdx.x_lines_blk)*%(threadIdx.x_line_dim) + %(threadIdx.x_line);
  // add bias to each elem of out_tile[] and store the results to out[]
  %(t_tile_stores);
}

