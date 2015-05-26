// 256 tbp
// each thread: computes 8x8 block of out
// loop over k dim
extern "C"  __global__ void %(cu_func_name)( float const * const filts, float const * const biases, float const * const in, float * const out ) {
  if( %(patch_ix_0) >= %(patch_ix_0_sz) ) { return; } // if no valid patches, done
  //if( (%(patch_ix_0) + %(t_tile_sz) - 1) >= %(patch_ix_0_sz) ) { return; } // HACK: if any in-valid patches, done

  __shared__ float in_smem[%(threadIdx.x_patch_tile_dim)*%(t_tile_sz)];
  __shared__ float filts_smem[%(threadIdx.x_out_chan_tile_dim)*%(t_tile_sz)];
  float out_tile[%(t_tile_sz)*%(t_tile_sz)] = {0}; // tile of output for this thread to compute, stored in registers
  // reg. buffers for one strip each from in and filts of %(t_tile_sz) elements, for the same filts_ix_out_chan_elem
  float filts_strip[%(t_tile_sz)]; // across output chans (stride is %(filts_ix_out_chan_sz) )
  float in_strip[%(t_tile_sz)]; // across patches (approx square block in x/y space, favoring x if sqrt() not integer)
  // iteratate over filter elements
  for( uint32_t filts_ix_out_chan_elem = 0; filts_ix_out_chan_elem != %(filts_ix_out_chan_sz); ++filts_ix_out_chan_elem ) {
    // (1) load %(t_tile_sz) elements from in and filts    
    __syncthreads();
    %(t_tile_loads);
    __syncthreads();
    // (2) do %(t_tile_sz)^2 fmas into out_tile
    %(t_tile_fmas);
  }
  // add bias to each elem of out_tile[] and store the results to out[]
  %(t_tile_stores);
}

#if 0
extern "C"  __global__ void %(cu_func_name)_direct( float const * const filts, float const * const biases, float const * const in, float * const out ) {
  uint32_t const out_ix = blockDim.x * blockIdx.x + threadIdx.x;
  if( out_ix >= %(out_ix_sz) ) { return; }
  float out_v = 0.0f;
  uint32_t const in_ix = %(out_ix_img) * %(in_ix_img_sz) + %(out_ix_y)*%(in_ix_y_sz)*%(stride) + %(out_ix_x)*%(in_ix_x_sz)*%(stride);
  uint32_t const filts_ix = %(out_ix_chan) * %(filts_ix_out_chan_sz);
  %(ops);
  out_v += biases[%(out_ix_chan)];
  out[out_ix] = out_v;
}

#endif

