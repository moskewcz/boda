typedef int int32_t;
#define CUCL_GLOBAL_KERNEL extern "C" __global__
#define GASQ
#define GLOB_ID_1D (blockDim.x * blockIdx.x + threadIdx.x)
#define LOC_ID_1D (threadIdx.x)
#define GRP_ID_1D (blockIdx.x)
#define LOC_SZ_1D (blockDim.x)
#define LOCSHAR_MEM __shared__
#define LSMASQ
#define BARRIER_SYNC __syncthreads()
CUCL_GLOBAL_KERNEL void bconv__out_chan_1000__in_chan_1024__y_1__x_1__img_1__chan_1000( GASQ float const * const filts, // CUCL IN out_chan:in_chan:y:x
					  GASQ float const * const out_grad_loss, // CUCL IN img:chan:y:x
					 
					  GASQ float * const in_grad_loss ) // CUCL OUT img:chan:y:x
/* work */  // CUCL REF pels_blk:out_ix_blk:pels_tile:out_ix_tile:pels:out_ix
/* oix */  // CUCL REF in_chan:sy:sx
/* fioc */  // CUCL REF out_chan:ky:kx
{
  // CUCL IX pel_ix out_grad_loss use_dims=img:y:x
  // CUCL IX filt_elem_ix fioc
  // CUCL IX out_ix oix
  // CUCL IX GRP_ID_1D work use_dims=pels_blk:out_ix_blk
  // CUCL IX LOC_ID_1D work use_dims=pels_tile:out_ix_tile
  // note: <each thread handles> work use_dims=pels:out_out_ix; with pels_sz==out_chan_sz==t_tile_sz (currently); loops over in.chan==filts.in_chan

  LOCSHAR_MEM float in_smem[40];
  LOCSHAR_MEM float filts_smem[120];
  float out_tile[8*8] = {0}; // tile of output for this thread to compute, stored in registers
  // reg. buffers for one strip each from in and filts, for the same filts element
  float filts_strip[8]; // across output chans
  float in_strip[8]; // across pels (approx square block in x/y space, favoring x if sqrt() not integer)

  int32_t const blk_out_ix = (GRP_ID_1D%9)*15*8;
  int32_t const blk_pel_ix = (GRP_ID_1D/9)*5*8;

  for( int32_t filt_elem_ix = 0; filt_elem_ix != 1024; ++filt_elem_ix ) {
    BARRIER_SYNC;
    for( int32_t i = 0; i != 1; ++i ) {
      if( (LOC_ID_1D+LOC_SZ_1D*i) < 40 ) { 
	int32_t const pel_ix = (blk_pel_ix+LOC_ID_1D+LOC_SZ_1D*i);
	float v = 0;
	int const smem_in_ix_y = ((pel_ix/6)%6)+(filt_elem_ix%1) - 0;
	int const smem_in_ix_x = (pel_ix%6)+(filt_elem_ix%1) - 0;
	if(smem_in_ix_y >= 0 && smem_in_ix_x >= 0 && (pel_ix/36) < 1 &&
	   smem_in_ix_x < 6 && smem_in_ix_y < 6 ) {
	  v = out_grad_loss[(pel_ix/36)*36000 +
			    filt_elem_ix*36 +
			    smem_in_ix_y*6 +
			    smem_in_ix_x*1];
	}
	in_smem[LOC_ID_1D+LOC_SZ_1D*i] = v;
      }
    }
    for( int32_t i = 0; i != 2; ++i ) {
      if( (LOC_ID_1D+LOC_SZ_1D*i) < 120 ) { 
	int32_t const out_ix = (blk_out_ix+LOC_ID_1D+LOC_SZ_1D*i);
	float v = 0;
	int const smem_filt_ix_y = (out_ix%1)+(filt_elem_ix%1)*1;
	int const smem_filt_ix_x = (out_ix%1)+(filt_elem_ix%1)*1;
	if( out_ix < 1024 && filt_elem_ix < 1000
	    && smem_filt_ix_x < 1 && smem_filt_ix_y < 1 ) {
	  v = filts[filt_elem_ix*1024 +
		    out_ix*1 + 
		    smem_filt_ix_y*1 +
		    smem_filt_ix_x*1];
	}
	filts_smem[LOC_ID_1D+LOC_SZ_1D*i] = v;
      }
    }

    BARRIER_SYNC;
    // begin loads
   filts_strip[0] = filts_smem[(LOC_ID_1D%15)*8+0];
   filts_strip[1] = filts_smem[(LOC_ID_1D%15)*8+1];
   filts_strip[2] = filts_smem[(LOC_ID_1D%15)*8+2];
   filts_strip[3] = filts_smem[(LOC_ID_1D%15)*8+3];
   filts_strip[4] = filts_smem[(LOC_ID_1D%15)*8+4];
   filts_strip[5] = filts_smem[(LOC_ID_1D%15)*8+5];
   filts_strip[6] = filts_smem[(LOC_ID_1D%15)*8+6];
   filts_strip[7] = filts_smem[(LOC_ID_1D%15)*8+7];
   in_strip[0] = in_smem[(LOC_ID_1D/15)*8+0];
   in_strip[1] = in_smem[(LOC_ID_1D/15)*8+1];
   in_strip[2] = in_smem[(LOC_ID_1D/15)*8+2];
   in_strip[3] = in_smem[(LOC_ID_1D/15)*8+3];
   in_strip[4] = in_smem[(LOC_ID_1D/15)*8+4];
   in_strip[5] = in_smem[(LOC_ID_1D/15)*8+5];
   in_strip[6] = in_smem[(LOC_ID_1D/15)*8+6];
   in_strip[7] = in_smem[(LOC_ID_1D/15)*8+7];
    // end loads;
    // begin fmas
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
    // end fmas;
  }

  int32_t pel_ix = blk_pel_ix + (LOC_ID_1D/15)*8; // first pel_ix for this thread
  int32_t igl_y, igl_x;
  for( int32_t work_pel = 0; work_pel < 8; ++work_pel, ++pel_ix) {
    int32_t out_ix = blk_out_ix + (LOC_ID_1D%15)*8; // first out_ix for this thread
    // begin outs_to_filts_strip
   switch(work_pel) { 
   filts_strip[0] = out_tile[0];
   filts_strip[1] = out_tile[1];
   filts_strip[2] = out_tile[2];
   filts_strip[3] = out_tile[3];
   filts_strip[4] = out_tile[4];
   filts_strip[5] = out_tile[5];
   filts_strip[6] = out_tile[6];
   filts_strip[7] = out_tile[7];
   break;
   filts_strip[0] = out_tile[8];
   filts_strip[1] = out_tile[9];
   filts_strip[2] = out_tile[10];
   filts_strip[3] = out_tile[11];
   filts_strip[4] = out_tile[12];
   filts_strip[5] = out_tile[13];
   filts_strip[6] = out_tile[14];
   filts_strip[7] = out_tile[15];
   break;
   filts_strip[0] = out_tile[16];
   filts_strip[1] = out_tile[17];
   filts_strip[2] = out_tile[18];
   filts_strip[3] = out_tile[19];
   filts_strip[4] = out_tile[20];
   filts_strip[5] = out_tile[21];
   filts_strip[6] = out_tile[22];
   filts_strip[7] = out_tile[23];
   break;
   filts_strip[0] = out_tile[24];
   filts_strip[1] = out_tile[25];
   filts_strip[2] = out_tile[26];
   filts_strip[3] = out_tile[27];
   filts_strip[4] = out_tile[28];
   filts_strip[5] = out_tile[29];
   filts_strip[6] = out_tile[30];
   filts_strip[7] = out_tile[31];
   break;
   filts_strip[0] = out_tile[32];
   filts_strip[1] = out_tile[33];
   filts_strip[2] = out_tile[34];
   filts_strip[3] = out_tile[35];
   filts_strip[4] = out_tile[36];
   filts_strip[5] = out_tile[37];
   filts_strip[6] = out_tile[38];
   filts_strip[7] = out_tile[39];
   break;
   filts_strip[0] = out_tile[40];
   filts_strip[1] = out_tile[41];
   filts_strip[2] = out_tile[42];
   filts_strip[3] = out_tile[43];
   filts_strip[4] = out_tile[44];
   filts_strip[5] = out_tile[45];
   filts_strip[6] = out_tile[46];
   filts_strip[7] = out_tile[47];
   break;
   filts_strip[0] = out_tile[48];
   filts_strip[1] = out_tile[49];
   filts_strip[2] = out_tile[50];
   filts_strip[3] = out_tile[51];
   filts_strip[4] = out_tile[52];
   filts_strip[5] = out_tile[53];
   filts_strip[6] = out_tile[54];
   filts_strip[7] = out_tile[55];
   break;
   filts_strip[0] = out_tile[56];
   filts_strip[1] = out_tile[57];
   filts_strip[2] = out_tile[58];
   filts_strip[3] = out_tile[59];
   filts_strip[4] = out_tile[60];
   filts_strip[5] = out_tile[61];
   filts_strip[6] = out_tile[62];
   filts_strip[7] = out_tile[63];
   break;
   } 
    // end outs_to_filts_strip;
    // begin stores
   
  igl_y = (((pel_ix/6)%6)-0)*1+(out_ix%1);
  igl_x = ((pel_ix%6)-0)*1+(out_ix%1);
  if( igl_x >= 0 && igl_y >= 0 && igl_y < 6 && igl_x < 6 &&
      out_ix < 1024 && (pel_ix/36) < 1 ) {
    in_grad_loss[ (pel_ix/36)*36864 + out_ix*36 + 
		  igl_y*6 + igl_x*1] = filts_strip[0];
};
   ++out_ix;
   
  igl_y = (((pel_ix/6)%6)-0)*1+(out_ix%1);
  igl_x = ((pel_ix%6)-0)*1+(out_ix%1);
  if( igl_x >= 0 && igl_y >= 0 && igl_y < 6 && igl_x < 6 &&
      out_ix < 1024 && (pel_ix/36) < 1 ) {
    in_grad_loss[ (pel_ix/36)*36864 + out_ix*36 + 
		  igl_y*6 + igl_x*1] = filts_strip[1];
};
   ++out_ix;
   
  igl_y = (((pel_ix/6)%6)-0)*1+(out_ix%1);
  igl_x = ((pel_ix%6)-0)*1+(out_ix%1);
  if( igl_x >= 0 && igl_y >= 0 && igl_y < 6 && igl_x < 6 &&
      out_ix < 1024 && (pel_ix/36) < 1 ) {
    in_grad_loss[ (pel_ix/36)*36864 + out_ix*36 + 
		  igl_y*6 + igl_x*1] = filts_strip[2];
};
   ++out_ix;
   
  igl_y = (((pel_ix/6)%6)-0)*1+(out_ix%1);
  igl_x = ((pel_ix%6)-0)*1+(out_ix%1);
  if( igl_x >= 0 && igl_y >= 0 && igl_y < 6 && igl_x < 6 &&
      out_ix < 1024 && (pel_ix/36) < 1 ) {
    in_grad_loss[ (pel_ix/36)*36864 + out_ix*36 + 
		  igl_y*6 + igl_x*1] = filts_strip[3];
};
   ++out_ix;
   
  igl_y = (((pel_ix/6)%6)-0)*1+(out_ix%1);
  igl_x = ((pel_ix%6)-0)*1+(out_ix%1);
  if( igl_x >= 0 && igl_y >= 0 && igl_y < 6 && igl_x < 6 &&
      out_ix < 1024 && (pel_ix/36) < 1 ) {
    in_grad_loss[ (pel_ix/36)*36864 + out_ix*36 + 
		  igl_y*6 + igl_x*1] = filts_strip[4];
};
   ++out_ix;
   
  igl_y = (((pel_ix/6)%6)-0)*1+(out_ix%1);
  igl_x = ((pel_ix%6)-0)*1+(out_ix%1);
  if( igl_x >= 0 && igl_y >= 0 && igl_y < 6 && igl_x < 6 &&
      out_ix < 1024 && (pel_ix/36) < 1 ) {
    in_grad_loss[ (pel_ix/36)*36864 + out_ix*36 + 
		  igl_y*6 + igl_x*1] = filts_strip[5];
};
   ++out_ix;
   
  igl_y = (((pel_ix/6)%6)-0)*1+(out_ix%1);
  igl_x = ((pel_ix%6)-0)*1+(out_ix%1);
  if( igl_x >= 0 && igl_y >= 0 && igl_y < 6 && igl_x < 6 &&
      out_ix < 1024 && (pel_ix/36) < 1 ) {
    in_grad_loss[ (pel_ix/36)*36864 + out_ix*36 + 
		  igl_y*6 + igl_x*1] = filts_strip[6];
};
   ++out_ix;
   
  igl_y = (((pel_ix/6)%6)-0)*1+(out_ix%1);
  igl_x = ((pel_ix%6)-0)*1+(out_ix%1);
  if( igl_x >= 0 && igl_y >= 0 && igl_y < 6 && igl_x < 6 &&
      out_ix < 1024 && (pel_ix/36) < 1 ) {
    in_grad_loss[ (pel_ix/36)*36864 + out_ix*36 + 
		  igl_y*6 + igl_x*1] = filts_strip[7];
};
   ++out_ix;
    // end stores;
  }
}
