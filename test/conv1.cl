
//typedef unsigned uint32_t;
typedef int int32_t;
typedef long long int64_t;
//float const FLT_MAX = /*0x1.fffffep127f*/ 340282346638528859811704183484516925440.0f;

// 256 tbp
// each thread: computes 8x8 block of out
// loop over k dim

kernel void conv__num_imgs_20__in_pad_3__in_dim_0_227__in_dim_1_227__conv_has_relu_1__kern_sz_7__stride_2__out_chans_64__in_chans_3( global float const * const filts, global float const * const biases, global float const * const in, global float * const out ) {
  int32_t const threadIdx_x = get_local_id(0);
  int32_t const blockIdx_x = get_group_id(0);
  int32_t const blockDim_x = get_local_size(0);

  local float in_smem[15*8];
  int32_t const blk_filt_ix_sz = 8*8;
  local float filts_smem[8*8];
  float out_tile[8*8] = {0}; // tile of output for this thread to compute, stored in registers
  // reg. buffers for one strip each from in and filts of 8 elements, for the same filts_ix_out_chan_elem
  float filts_strip[8]; // across output chans (stride is blk_filt_ix_sz )
  float in_strip[8]; // across patches (approx square block in x/y space, favoring x if sqrt() not integer)
  int32_t const blk_filt_ix_base = (blockIdx_x%1)*blk_filt_ix_sz;

  int32_t const blk_patch_ix_sz = 15*8;
  int32_t const blk_patch_ix_base = blockIdx_x*blk_patch_ix_sz;

  // iteratate over filter elements
  int32_t filts_off = blk_filt_ix_base;
  for( int32_t filts_ix_out_chan_elem = 0; filts_ix_out_chan_elem != (9408 / 64);
       ++filts_ix_out_chan_elem ) {
    __syncthreads();
    if( threadIdx_x < blk_filt_ix_sz ) { 
#ifdef NO_IOX // by default, we don't ever disable this, since it's seems about as good as it can be already
      //filts_smem[threadIdx_x] = threadIdx_x;
      filts_smem[threadIdx_x] = filts[threadIdx_x];
#else
      filts_smem[threadIdx_x] = filts[filts_off + threadIdx_x];
#endif
    }
    for( int32_t i = 0; i != 1; ++i ) {
      if( (threadIdx_x+blockDim_x*i) < blk_patch_ix_sz ) { 
	int32_t const t_smem_patch_ix = (blk_patch_ix_base+threadIdx_x+blockDim_x*i);

#ifdef NO_IO
	//float v = threadIdx_x;
	//float v = in[threadIdx_x];
	float v = in[filts_off + threadIdx_x];
#else
	float v = 0;
      int const smem_in_ix_y = ((t_smem_patch_ix/114)%114)*2+((filts_ix_out_chan_elem/7)%7) - 3;
      int const smem_in_ix_x = (t_smem_patch_ix%114)*2+(filts_ix_out_chan_elem%7) - 3;
      if(smem_in_ix_y >= 0 && smem_in_ix_x >= 0 && 
          (t_smem_patch_ix/12996) < 20 && 
         smem_in_ix_x < 227 && smem_in_ix_y < 227 ) {
        v = in[(t_smem_patch_ix/12996)*154587 +
          (filts_ix_out_chan_elem/49)*51529 +
          smem_in_ix_y*227 +
          smem_in_ix_x*1];
      };
#endif
	in_smem[threadIdx_x+blockDim_x*i] = v;
      }
    }
    filts_off += 64;
    __syncthreads();
#ifdef NO_IO
    // begin t_tile_dummy_loads
    filts_strip[0] = filts_smem[(threadIdx_x % 32) + 0];
    filts_strip[1] = filts_smem[(threadIdx_x % 32) + 1];
    filts_strip[2] = filts_smem[(threadIdx_x % 32) + 2];
    filts_strip[3] = filts_smem[(threadIdx_x % 32) + 3];
    filts_strip[4] = filts_smem[(threadIdx_x % 32) + 4];
    filts_strip[5] = filts_smem[(threadIdx_x % 32) + 5];
    filts_strip[6] = filts_smem[(threadIdx_x % 32) + 6];
    filts_strip[7] = filts_smem[(threadIdx_x % 32) + 7];
    in_strip[0] = in_smem[(threadIdx_x % 32) + 0];
    in_strip[1] = in_smem[(threadIdx_x % 32) + 1];
    in_strip[2] = in_smem[(threadIdx_x % 32) + 2];
    in_strip[3] = in_smem[(threadIdx_x % 32) + 3];
    in_strip[4] = in_smem[(threadIdx_x % 32) + 4];
    in_strip[5] = in_smem[(threadIdx_x % 32) + 5];
    in_strip[6] = in_smem[(threadIdx_x % 32) + 6];
    in_strip[7] = in_smem[(threadIdx_x % 32) + 7];
    // end t_tile_dummy_loads;
#else
    // begin t_tile_loads
    filts_strip[0] = filts_smem[(threadIdx_x%8)+0*8];
    filts_strip[1] = filts_smem[(threadIdx_x%8)+1*8];
    filts_strip[2] = filts_smem[(threadIdx_x%8)+2*8];
    filts_strip[3] = filts_smem[(threadIdx_x%8)+3*8];
    filts_strip[4] = filts_smem[(threadIdx_x%8)+4*8];
    filts_strip[5] = filts_smem[(threadIdx_x%8)+5*8];
    filts_strip[6] = filts_smem[(threadIdx_x%8)+6*8];
    filts_strip[7] = filts_smem[(threadIdx_x%8)+7*8];
    in_strip[0] = in_smem[8*(threadIdx_x/8)+0];
    in_strip[1] = in_smem[8*(threadIdx_x/8)+1];
    in_strip[2] = in_smem[8*(threadIdx_x/8)+2];
    in_strip[3] = in_smem[8*(threadIdx_x/8)+3];
    in_strip[4] = in_smem[8*(threadIdx_x/8)+4];
    in_strip[5] = in_smem[8*(threadIdx_x/8)+5];
    in_strip[6] = in_smem[8*(threadIdx_x/8)+6];
    in_strip[7] = in_smem[8*(threadIdx_x/8)+7];
    // end t_tile_loads;
#endif
    // (2) do 8^2 fmas into out_tile
    // begin t_tile_fmas
    out_tile[0] += filts_strip[0]*in_strip[0];
    out_tile[1] += filts_strip[1]*in_strip[0];
    out_tile[2] += filts_strip[2]*in_strip[0];
    out_tile[3] += filts_strip[3]*in_strip[0];
    out_tile[4] += filts_strip[4]*in_strip[0];
    out_tile[5] += filts_strip[5]*in_strip[0];
    out_tile[6] += filts_strip[6]*in_strip[0];
    out_tile[7] += filts_strip[7]*in_strip[0];
    out_tile[8] += filts_strip[0]*in_strip[1];
    out_tile[9] += filts_strip[1]*in_strip[1];
    out_tile[10] += filts_strip[2]*in_strip[1];
    out_tile[11] += filts_strip[3]*in_strip[1];
    out_tile[12] += filts_strip[4]*in_strip[1];
    out_tile[13] += filts_strip[5]*in_strip[1];
    out_tile[14] += filts_strip[6]*in_strip[1];
    out_tile[15] += filts_strip[7]*in_strip[1];
    out_tile[16] += filts_strip[0]*in_strip[2];
    out_tile[17] += filts_strip[1]*in_strip[2];
    out_tile[18] += filts_strip[2]*in_strip[2];
    out_tile[19] += filts_strip[3]*in_strip[2];
    out_tile[20] += filts_strip[4]*in_strip[2];
    out_tile[21] += filts_strip[5]*in_strip[2];
    out_tile[22] += filts_strip[6]*in_strip[2];
    out_tile[23] += filts_strip[7]*in_strip[2];
    out_tile[24] += filts_strip[0]*in_strip[3];
    out_tile[25] += filts_strip[1]*in_strip[3];
    out_tile[26] += filts_strip[2]*in_strip[3];
    out_tile[27] += filts_strip[3]*in_strip[3];
    out_tile[28] += filts_strip[4]*in_strip[3];
    out_tile[29] += filts_strip[5]*in_strip[3];
    out_tile[30] += filts_strip[6]*in_strip[3];
    out_tile[31] += filts_strip[7]*in_strip[3];
    out_tile[32] += filts_strip[0]*in_strip[4];
    out_tile[33] += filts_strip[1]*in_strip[4];
    out_tile[34] += filts_strip[2]*in_strip[4];
    out_tile[35] += filts_strip[3]*in_strip[4];
    out_tile[36] += filts_strip[4]*in_strip[4];
    out_tile[37] += filts_strip[5]*in_strip[4];
    out_tile[38] += filts_strip[6]*in_strip[4];
    out_tile[39] += filts_strip[7]*in_strip[4];
    out_tile[40] += filts_strip[0]*in_strip[5];
    out_tile[41] += filts_strip[1]*in_strip[5];
    out_tile[42] += filts_strip[2]*in_strip[5];
    out_tile[43] += filts_strip[3]*in_strip[5];
    out_tile[44] += filts_strip[4]*in_strip[5];
    out_tile[45] += filts_strip[5]*in_strip[5];
    out_tile[46] += filts_strip[6]*in_strip[5];
    out_tile[47] += filts_strip[7]*in_strip[5];
    out_tile[48] += filts_strip[0]*in_strip[6];
    out_tile[49] += filts_strip[1]*in_strip[6];
    out_tile[50] += filts_strip[2]*in_strip[6];
    out_tile[51] += filts_strip[3]*in_strip[6];
    out_tile[52] += filts_strip[4]*in_strip[6];
    out_tile[53] += filts_strip[5]*in_strip[6];
    out_tile[54] += filts_strip[6]*in_strip[6];
    out_tile[55] += filts_strip[7]*in_strip[6];
    out_tile[56] += filts_strip[0]*in_strip[7];
    out_tile[57] += filts_strip[1]*in_strip[7];
    out_tile[58] += filts_strip[2]*in_strip[7];
    out_tile[59] += filts_strip[3]*in_strip[7];
    out_tile[60] += filts_strip[4]*in_strip[7];
    out_tile[61] += filts_strip[5]*in_strip[7];
    out_tile[62] += filts_strip[6]*in_strip[7];
    out_tile[63] += filts_strip[7]*in_strip[7];
    // end t_tile_fmas;
  }

  // load per-block biases into smem
  __syncthreads();
  if( threadIdx_x < blk_filt_ix_sz ) { 
    int32_t const ocix_base = (blockIdx_x%1)*blk_filt_ix_sz;
    int32_t const load_reg = threadIdx_x / 8;
    int32_t const load_tile = threadIdx_x % 8;
    int32_t const ocix = ocix_base + load_tile*8 + load_reg;
    if( ocix < 64 ) { filts_smem[threadIdx_x] = biases[ ocix ]; }
    //int32_t const ocix_tile = (ocix / 8) % 8;
    //int32_t const ocix_reg = ocix % 8;
    //filts_smem[ocix_tile * 1 + ocix_reg * 8] = biases[ocix];
  }
  __syncthreads();
  // load biases into filts_strip
  // begin t_tile_loads
    filts_strip[0] = filts_smem[(threadIdx_x%8)+0*8];
    filts_strip[1] = filts_smem[(threadIdx_x%8)+1*8];
    filts_strip[2] = filts_smem[(threadIdx_x%8)+2*8];
    filts_strip[3] = filts_smem[(threadIdx_x%8)+3*8];
    filts_strip[4] = filts_smem[(threadIdx_x%8)+4*8];
    filts_strip[5] = filts_smem[(threadIdx_x%8)+5*8];
    filts_strip[6] = filts_smem[(threadIdx_x%8)+6*8];
    filts_strip[7] = filts_smem[(threadIdx_x%8)+7*8];
    in_strip[0] = in_smem[8*(threadIdx_x/8)+0];
    in_strip[1] = in_smem[8*(threadIdx_x/8)+1];
    in_strip[2] = in_smem[8*(threadIdx_x/8)+2];
    in_strip[3] = in_smem[8*(threadIdx_x/8)+3];
    in_strip[4] = in_smem[8*(threadIdx_x/8)+4];
    in_strip[5] = in_smem[8*(threadIdx_x/8)+5];
    in_strip[6] = in_smem[8*(threadIdx_x/8)+6];
    in_strip[7] = in_smem[8*(threadIdx_x/8)+7];
    // end t_tile_loads;

  // add bias to each elem of out_tile[] and store the results to out[]
#ifdef NO_IO
  // begin t_tile_dummy_stores
 out[0] = 0.0f
 + max(0.0f,out_tile[0] + filts_strip[0])
 + max(0.0f,out_tile[1] + filts_strip[1])
 + max(0.0f,out_tile[2] + filts_strip[2])
 + max(0.0f,out_tile[3] + filts_strip[3])
 + max(0.0f,out_tile[4] + filts_strip[4])
 + max(0.0f,out_tile[5] + filts_strip[5])
 + max(0.0f,out_tile[6] + filts_strip[6])
 + max(0.0f,out_tile[7] + filts_strip[7])
 + max(0.0f,out_tile[8] + filts_strip[0])
 + max(0.0f,out_tile[9] + filts_strip[1])
 + max(0.0f,out_tile[10] + filts_strip[2])
 + max(0.0f,out_tile[11] + filts_strip[3])
 + max(0.0f,out_tile[12] + filts_strip[4])
 + max(0.0f,out_tile[13] + filts_strip[5])
 + max(0.0f,out_tile[14] + filts_strip[6])
 + max(0.0f,out_tile[15] + filts_strip[7])
 + max(0.0f,out_tile[16] + filts_strip[0])
 + max(0.0f,out_tile[17] + filts_strip[1])
 + max(0.0f,out_tile[18] + filts_strip[2])
 + max(0.0f,out_tile[19] + filts_strip[3])
 + max(0.0f,out_tile[20] + filts_strip[4])
 + max(0.0f,out_tile[21] + filts_strip[5])
 + max(0.0f,out_tile[22] + filts_strip[6])
 + max(0.0f,out_tile[23] + filts_strip[7])
 + max(0.0f,out_tile[24] + filts_strip[0])
 + max(0.0f,out_tile[25] + filts_strip[1])
 + max(0.0f,out_tile[26] + filts_strip[2])
 + max(0.0f,out_tile[27] + filts_strip[3])
 + max(0.0f,out_tile[28] + filts_strip[4])
 + max(0.0f,out_tile[29] + filts_strip[5])
 + max(0.0f,out_tile[30] + filts_strip[6])
 + max(0.0f,out_tile[31] + filts_strip[7])
 + max(0.0f,out_tile[32] + filts_strip[0])
 + max(0.0f,out_tile[33] + filts_strip[1])
 + max(0.0f,out_tile[34] + filts_strip[2])
 + max(0.0f,out_tile[35] + filts_strip[3])
 + max(0.0f,out_tile[36] + filts_strip[4])
 + max(0.0f,out_tile[37] + filts_strip[5])
 + max(0.0f,out_tile[38] + filts_strip[6])
 + max(0.0f,out_tile[39] + filts_strip[7])
 + max(0.0f,out_tile[40] + filts_strip[0])
 + max(0.0f,out_tile[41] + filts_strip[1])
 + max(0.0f,out_tile[42] + filts_strip[2])
 + max(0.0f,out_tile[43] + filts_strip[3])
 + max(0.0f,out_tile[44] + filts_strip[4])
 + max(0.0f,out_tile[45] + filts_strip[5])
 + max(0.0f,out_tile[46] + filts_strip[6])
 + max(0.0f,out_tile[47] + filts_strip[7])
 + max(0.0f,out_tile[48] + filts_strip[0])
 + max(0.0f,out_tile[49] + filts_strip[1])
 + max(0.0f,out_tile[50] + filts_strip[2])
 + max(0.0f,out_tile[51] + filts_strip[3])
 + max(0.0f,out_tile[52] + filts_strip[4])
 + max(0.0f,out_tile[53] + filts_strip[5])
 + max(0.0f,out_tile[54] + filts_strip[6])
 + max(0.0f,out_tile[55] + filts_strip[7])
 + max(0.0f,out_tile[56] + filts_strip[0])
 + max(0.0f,out_tile[57] + filts_strip[1])
 + max(0.0f,out_tile[58] + filts_strip[2])
 + max(0.0f,out_tile[59] + filts_strip[3])
 + max(0.0f,out_tile[60] + filts_strip[4])
 + max(0.0f,out_tile[61] + filts_strip[5])
 + max(0.0f,out_tile[62] + filts_strip[6])
 + max(0.0f,out_tile[63] + filts_strip[7])
;
;
#else
  // begin t_tile_stores
  int32_t tpix[8];
  int32_t tcix[8];
  tpix[0] = ((((threadIdx_x/8)+blockIdx_x*15)*8+0)/12996)*831744 + 
   ( (((threadIdx_x/8)+blockIdx_x*15)*8+0) % 12996 ); // cache out patch ixs
   tpix[1] = ((((threadIdx_x/8)+blockIdx_x*15)*8+1)/12996)*831744 + 
   ( (((threadIdx_x/8)+blockIdx_x*15)*8+1) % 12996 ); // cache out patch ixs
   tpix[2] = ((((threadIdx_x/8)+blockIdx_x*15)*8+2)/12996)*831744 + 
   ( (((threadIdx_x/8)+blockIdx_x*15)*8+2) % 12996 ); // cache out patch ixs
   tpix[3] = ((((threadIdx_x/8)+blockIdx_x*15)*8+3)/12996)*831744 + 
   ( (((threadIdx_x/8)+blockIdx_x*15)*8+3) % 12996 ); // cache out patch ixs
   tpix[4] = ((((threadIdx_x/8)+blockIdx_x*15)*8+4)/12996)*831744 + 
   ( (((threadIdx_x/8)+blockIdx_x*15)*8+4) % 12996 ); // cache out patch ixs
   tpix[5] = ((((threadIdx_x/8)+blockIdx_x*15)*8+5)/12996)*831744 + 
   ( (((threadIdx_x/8)+blockIdx_x*15)*8+5) % 12996 ); // cache out patch ixs
   tpix[6] = ((((threadIdx_x/8)+blockIdx_x*15)*8+6)/12996)*831744 + 
   ( (((threadIdx_x/8)+blockIdx_x*15)*8+6) % 12996 ); // cache out patch ixs
   tpix[7] = ((((threadIdx_x/8)+blockIdx_x*15)*8+7)/12996)*831744 + 
   ( (((threadIdx_x/8)+blockIdx_x*15)*8+7) % 12996 ); // cache out patch ixs
   tcix[0] = ((((threadIdx_x%8)+(blockIdx_x%1)*8)*8)+0)*12996; // cache out chan ixs
  tcix[1] = ((((threadIdx_x%8)+(blockIdx_x%1)*8)*8)+1)*12996; // cache out chan ixs
  tcix[2] = ((((threadIdx_x%8)+(blockIdx_x%1)*8)*8)+2)*12996; // cache out chan ixs
  tcix[3] = ((((threadIdx_x%8)+(blockIdx_x%1)*8)*8)+3)*12996; // cache out chan ixs
  tcix[4] = ((((threadIdx_x%8)+(blockIdx_x%1)*8)*8)+4)*12996; // cache out chan ixs
  tcix[5] = ((((threadIdx_x%8)+(blockIdx_x%1)*8)*8)+5)*12996; // cache out chan ixs
  tcix[6] = ((((threadIdx_x%8)+(blockIdx_x%1)*8)*8)+6)*12996; // cache out chan ixs
  tcix[7] = ((((threadIdx_x%8)+(blockIdx_x%1)*8)*8)+7)*12996; // cache out chan ixs
  if( (((threadIdx_x/8)+blockIdx_x*15)*8+0) >= 259920 ) { return; } // this patch and the following are off-the-end patches, so don't store them.
if( tcix[0] < (64*12996) ) { out[ tpix[0] + tcix[0] ] = max(0.0f,out_tile[0] + filts_strip[0]); }
if( tcix[1] < (64*12996) ) { out[ tpix[0] + tcix[1] ] = max(0.0f,out_tile[1] + filts_strip[1]); }
if( tcix[2] < (64*12996) ) { out[ tpix[0] + tcix[2] ] = max(0.0f,out_tile[2] + filts_strip[2]); }
if( tcix[3] < (64*12996) ) { out[ tpix[0] + tcix[3] ] = max(0.0f,out_tile[3] + filts_strip[3]); }
if( tcix[4] < (64*12996) ) { out[ tpix[0] + tcix[4] ] = max(0.0f,out_tile[4] + filts_strip[4]); }
if( tcix[5] < (64*12996) ) { out[ tpix[0] + tcix[5] ] = max(0.0f,out_tile[5] + filts_strip[5]); }
if( tcix[6] < (64*12996) ) { out[ tpix[0] + tcix[6] ] = max(0.0f,out_tile[6] + filts_strip[6]); }
if( tcix[7] < (64*12996) ) { out[ tpix[0] + tcix[7] ] = max(0.0f,out_tile[7] + filts_strip[7]); }
  if( (((threadIdx_x/8)+blockIdx_x*15)*8+1) >= 259920 ) { return; } // this patch and the following are off-the-end patches, so don't store them.
if( tcix[0] < (64*12996) ) { out[ tpix[1] + tcix[0] ] = max(0.0f,out_tile[8] + filts_strip[0]); }
if( tcix[1] < (64*12996) ) { out[ tpix[1] + tcix[1] ] = max(0.0f,out_tile[9] + filts_strip[1]); }
if( tcix[2] < (64*12996) ) { out[ tpix[1] + tcix[2] ] = max(0.0f,out_tile[10] + filts_strip[2]); }
if( tcix[3] < (64*12996) ) { out[ tpix[1] + tcix[3] ] = max(0.0f,out_tile[11] + filts_strip[3]); }
if( tcix[4] < (64*12996) ) { out[ tpix[1] + tcix[4] ] = max(0.0f,out_tile[12] + filts_strip[4]); }
if( tcix[5] < (64*12996) ) { out[ tpix[1] + tcix[5] ] = max(0.0f,out_tile[13] + filts_strip[5]); }
if( tcix[6] < (64*12996) ) { out[ tpix[1] + tcix[6] ] = max(0.0f,out_tile[14] + filts_strip[6]); }
if( tcix[7] < (64*12996) ) { out[ tpix[1] + tcix[7] ] = max(0.0f,out_tile[15] + filts_strip[7]); }
  if( (((threadIdx_x/8)+blockIdx_x*15)*8+2) >= 259920 ) { return; } // this patch and the following are off-the-end patches, so don't store them.
if( tcix[0] < (64*12996) ) { out[ tpix[2] + tcix[0] ] = max(0.0f,out_tile[16] + filts_strip[0]); }
if( tcix[1] < (64*12996) ) { out[ tpix[2] + tcix[1] ] = max(0.0f,out_tile[17] + filts_strip[1]); }
if( tcix[2] < (64*12996) ) { out[ tpix[2] + tcix[2] ] = max(0.0f,out_tile[18] + filts_strip[2]); }
if( tcix[3] < (64*12996) ) { out[ tpix[2] + tcix[3] ] = max(0.0f,out_tile[19] + filts_strip[3]); }
if( tcix[4] < (64*12996) ) { out[ tpix[2] + tcix[4] ] = max(0.0f,out_tile[20] + filts_strip[4]); }
if( tcix[5] < (64*12996) ) { out[ tpix[2] + tcix[5] ] = max(0.0f,out_tile[21] + filts_strip[5]); }
if( tcix[6] < (64*12996) ) { out[ tpix[2] + tcix[6] ] = max(0.0f,out_tile[22] + filts_strip[6]); }
if( tcix[7] < (64*12996) ) { out[ tpix[2] + tcix[7] ] = max(0.0f,out_tile[23] + filts_strip[7]); }
  if( (((threadIdx_x/8)+blockIdx_x*15)*8+3) >= 259920 ) { return; } // this patch and the following are off-the-end patches, so don't store them.
if( tcix[0] < (64*12996) ) { out[ tpix[3] + tcix[0] ] = max(0.0f,out_tile[24] + filts_strip[0]); }
if( tcix[1] < (64*12996) ) { out[ tpix[3] + tcix[1] ] = max(0.0f,out_tile[25] + filts_strip[1]); }
if( tcix[2] < (64*12996) ) { out[ tpix[3] + tcix[2] ] = max(0.0f,out_tile[26] + filts_strip[2]); }
if( tcix[3] < (64*12996) ) { out[ tpix[3] + tcix[3] ] = max(0.0f,out_tile[27] + filts_strip[3]); }
if( tcix[4] < (64*12996) ) { out[ tpix[3] + tcix[4] ] = max(0.0f,out_tile[28] + filts_strip[4]); }
if( tcix[5] < (64*12996) ) { out[ tpix[3] + tcix[5] ] = max(0.0f,out_tile[29] + filts_strip[5]); }
if( tcix[6] < (64*12996) ) { out[ tpix[3] + tcix[6] ] = max(0.0f,out_tile[30] + filts_strip[6]); }
if( tcix[7] < (64*12996) ) { out[ tpix[3] + tcix[7] ] = max(0.0f,out_tile[31] + filts_strip[7]); }
  if( (((threadIdx_x/8)+blockIdx_x*15)*8+4) >= 259920 ) { return; } // this patch and the following are off-the-end patches, so don't store them.
if( tcix[0] < (64*12996) ) { out[ tpix[4] + tcix[0] ] = max(0.0f,out_tile[32] + filts_strip[0]); }
if( tcix[1] < (64*12996) ) { out[ tpix[4] + tcix[1] ] = max(0.0f,out_tile[33] + filts_strip[1]); }
if( tcix[2] < (64*12996) ) { out[ tpix[4] + tcix[2] ] = max(0.0f,out_tile[34] + filts_strip[2]); }
if( tcix[3] < (64*12996) ) { out[ tpix[4] + tcix[3] ] = max(0.0f,out_tile[35] + filts_strip[3]); }
if( tcix[4] < (64*12996) ) { out[ tpix[4] + tcix[4] ] = max(0.0f,out_tile[36] + filts_strip[4]); }
if( tcix[5] < (64*12996) ) { out[ tpix[4] + tcix[5] ] = max(0.0f,out_tile[37] + filts_strip[5]); }
if( tcix[6] < (64*12996) ) { out[ tpix[4] + tcix[6] ] = max(0.0f,out_tile[38] + filts_strip[6]); }
if( tcix[7] < (64*12996) ) { out[ tpix[4] + tcix[7] ] = max(0.0f,out_tile[39] + filts_strip[7]); }
  if( (((threadIdx_x/8)+blockIdx_x*15)*8+5) >= 259920 ) { return; } // this patch and the following are off-the-end patches, so don't store them.
if( tcix[0] < (64*12996) ) { out[ tpix[5] + tcix[0] ] = max(0.0f,out_tile[40] + filts_strip[0]); }
if( tcix[1] < (64*12996) ) { out[ tpix[5] + tcix[1] ] = max(0.0f,out_tile[41] + filts_strip[1]); }
if( tcix[2] < (64*12996) ) { out[ tpix[5] + tcix[2] ] = max(0.0f,out_tile[42] + filts_strip[2]); }
if( tcix[3] < (64*12996) ) { out[ tpix[5] + tcix[3] ] = max(0.0f,out_tile[43] + filts_strip[3]); }
if( tcix[4] < (64*12996) ) { out[ tpix[5] + tcix[4] ] = max(0.0f,out_tile[44] + filts_strip[4]); }
if( tcix[5] < (64*12996) ) { out[ tpix[5] + tcix[5] ] = max(0.0f,out_tile[45] + filts_strip[5]); }
if( tcix[6] < (64*12996) ) { out[ tpix[5] + tcix[6] ] = max(0.0f,out_tile[46] + filts_strip[6]); }
if( tcix[7] < (64*12996) ) { out[ tpix[5] + tcix[7] ] = max(0.0f,out_tile[47] + filts_strip[7]); }
  if( (((threadIdx_x/8)+blockIdx_x*15)*8+6) >= 259920 ) { return; } // this patch and the following are off-the-end patches, so don't store them.
if( tcix[0] < (64*12996) ) { out[ tpix[6] + tcix[0] ] = max(0.0f,out_tile[48] + filts_strip[0]); }
if( tcix[1] < (64*12996) ) { out[ tpix[6] + tcix[1] ] = max(0.0f,out_tile[49] + filts_strip[1]); }
if( tcix[2] < (64*12996) ) { out[ tpix[6] + tcix[2] ] = max(0.0f,out_tile[50] + filts_strip[2]); }
if( tcix[3] < (64*12996) ) { out[ tpix[6] + tcix[3] ] = max(0.0f,out_tile[51] + filts_strip[3]); }
if( tcix[4] < (64*12996) ) { out[ tpix[6] + tcix[4] ] = max(0.0f,out_tile[52] + filts_strip[4]); }
if( tcix[5] < (64*12996) ) { out[ tpix[6] + tcix[5] ] = max(0.0f,out_tile[53] + filts_strip[5]); }
if( tcix[6] < (64*12996) ) { out[ tpix[6] + tcix[6] ] = max(0.0f,out_tile[54] + filts_strip[6]); }
if( tcix[7] < (64*12996) ) { out[ tpix[6] + tcix[7] ] = max(0.0f,out_tile[55] + filts_strip[7]); }
  if( (((threadIdx_x/8)+blockIdx_x*15)*8+7) >= 259920 ) { return; } // this patch and the following are off-the-end patches, so don't store them.
if( tcix[0] < (64*12996) ) { out[ tpix[7] + tcix[0] ] = max(0.0f,out_tile[56] + filts_strip[0]); }
if( tcix[1] < (64*12996) ) { out[ tpix[7] + tcix[1] ] = max(0.0f,out_tile[57] + filts_strip[1]); }
if( tcix[2] < (64*12996) ) { out[ tpix[7] + tcix[2] ] = max(0.0f,out_tile[58] + filts_strip[2]); }
if( tcix[3] < (64*12996) ) { out[ tpix[7] + tcix[3] ] = max(0.0f,out_tile[59] + filts_strip[3]); }
if( tcix[4] < (64*12996) ) { out[ tpix[7] + tcix[4] ] = max(0.0f,out_tile[60] + filts_strip[4]); }
if( tcix[5] < (64*12996) ) { out[ tpix[7] + tcix[5] ] = max(0.0f,out_tile[61] + filts_strip[5]); }
if( tcix[6] < (64*12996) ) { out[ tpix[7] + tcix[6] ] = max(0.0f,out_tile[62] + filts_strip[6]); }
if( tcix[7] < (64*12996) ) { out[ tpix[7] + tcix[7] ] = max(0.0f,out_tile[63] + filts_strip[7]); }
  // end t_tile_stores;
#endif
}
