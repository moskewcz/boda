// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"cnn_op.H"
#include"str_util.H"
#include"gbt_tile.H"

namespace boda 
{

  // based on the semantics/type of the operation represented by conv_op_base_t and the code generation parameters,
  // annotate the operation with the information needed for the next stage of code generation. In particular, this code
  // selects an operation variant and selects tuning paramters for this operations for that variant.
  void add_cnn_codegen_annotations( conv_op_base_t * const op, op_tune_t const & op_tune ) {

    bool const enable_ipconv = op_tune.ipconv;
    bool const enable_k1conv = op_tune.k1conv;
    bool const enable_tconv = op_tune.tconv;
    bool const force_enable_tconv = (op_tune.tconv==2);
    u32_pt_t const & t_tile_sz = op_tune.MNt;
    uint32_t const max_tpb = op_tune.MNb.dims_prod(); // FIXME: pass as M/N parts, not product.
    dims_t ni_dims;
    dims_t no_dims = op->get_dims( op->coi->top_an(0) );
    u32_pt_t const no_sz = get_xy_dims( no_dims );
    if( !op->is(Concat_coi) ) { ni_dims = op->get_dims( op->coi->bot_an(0) ); } // empty for Concat where we shouldn't use it, otherwise first input
    bool const is_conv = op->is( Convolution_coi );
    bool const is_pool = op->is( Pooling_coi );
    dims_t in_dims;
    if( is_conv || is_pool || op->is( Spreading_coi ) || op->is( BckConv_coi ) ) {
      assert_st( !ni_dims.empty() );
      in_dims = ni_dims;
      op->dims_vals["in_ref"] = in_dims; // tconv needs the standard input dims for reference
      u32_pt_t kern_sz_;
      if( has( op->dims_vals, "kern_sz" ) ) { kern_sz_ = op->kern_sz(); }
      else {
	if( is_pool ) { kern_sz_ = get_xy_dims( ni_dims ); }
	else if( op->is( Spreading_coi ) ) { kern_sz_ = get_xy_dims( no_dims ); }
	else { assert_st(0); }
	op->dims_vals["kern_sz"] = dims_t{ {kern_sz_.d[1],kern_sz_.d[0]}, {"y","x"}, 1 }; // FIXME: not ideal ...
      } 
      if( is_conv && enable_ipconv && op->in_pad().is_zeros() && (get_xy_dims(no_dims) == u32_pt_t{1,1}) ) {
	op->set_cts( ipconv_str ); // single output per-chan-per-image: inner-product case
      } else if( is_conv && enable_k1conv && (kern_sz_ == u32_pt_t{1,1}) && (op->stride() == u32_pt_t{1,1}) 
		 && (no_sz.d[0] >= 6) && (no_sz.d[0] <= 300 ) && (no_dims.dsz("chan") >= 64) ) 
      { 
	if( !op->in_pad().is_zeros() ) { printf( "warning: can't use k1conv due only to non-zero padding on layer with kernel size 1\n" ); op->set_cts( conv_str ); }
	else { 
          if( op_tune.use_local_mem == 2 ) { op->set_cts( k1conv_simd_str ); }
          else { op->set_cts( k1conv_str ); }
        }
      }
      else if( is_conv && enable_tconv && (force_enable_tconv || ( kern_sz_.both_dims_le(op_tune.tconv_max_ksz)
								   && (kern_sz_.both_dims_ge(u32_pt_t{1,1}) && (no_sz.d[0] >= 6))))) {
        op->set_cts( tconv_str );
      }
      else { op->set_cts( conv_str ); }

      if( op->is( BckConv_coi ) ) {
	// note: since fwd strides are always N/1, bck 'strides' are always 1/N, meaning stride in the fwd sense will
	// always be 1 for the bck conv: 3x3@s2 -> 2x2@s1; 11x11@s4 -> 3x3@s1; 1x1@s1 -> 1x1@s1 ...
	u32_pt_t bck_kern_sz = ceil_div( kern_sz_, op->stride() ); 
	// if back kernel conv is convolved aligned to the - corner of output space, it yields results for the
	// post-padding input space region: [bck_in_off,bck_in_off+stride)
	u32_pt_t const bck_pad_in_off = (bck_kern_sz - u32_pt_t(1,1)) * op->stride();
	assert_st( bck_pad_in_off.dims_are_same() );
	// we don't need compute values for the input padding, so adjust to un-padded input space
	i32_pt_t bck_in_off = u32_to_i32( bck_pad_in_off ) - u32_to_i32(op->in_pad());
	assert_st( bck_in_off.dims_are_same() ); // not handled, since we want/need per-axis padding for that
	// now, calculate where we need to really start in output space to have the first results region inlcude 0
	i32_pt_t bck_in_pad = ceil_div( bck_in_off, op->stride() );
	// FIXME: 'too much' fwd-in-pad can the  bck-in-pad this negative. sensible, so handle?
	assert_st( bck_in_pad.both_dims_ge_zero() );
	// now, to get patch count, see how many in pels we're missing
	bck_in_off -= bck_in_pad * u32_to_i32(op->stride()); // first region calculated at - corner of padding out space
	// calculate number of extra pels needed to cover last pel in unpadded input space
	i32_pt_t bck_pels_sz = ceil_div( u32_to_i32(get_xy_dims(no_dims)) - (bck_in_off + u32_to_i32(op->stride())), op->stride() ); 
	bck_pels_sz += i32_pt_t(1,1); // include starting pixel
	assert_st( bck_pels_sz.both_dims_gt( i32_pt_t() ) );

	op->dims_vals["bck_in_pad"] = dims_t( vect_uint32_t{ uint32_t(bck_in_pad.d[1]), uint32_t(bck_in_pad.d[0]) }, 
					  vect_string{"y","x"}, 1 );
	op->dims_vals["bck_pad_in_off"] = dims_t( vect_uint32_t{ bck_pad_in_off.d[1], bck_pad_in_off.d[0] }, vect_string{"y","x"}, 1 );

	dims_t const & ogld = op->get_dims("out_grad_loss");
	dims_t const & fgld = op->get_dims("filts_grad_loss");

	gbt_tile_t gbt;
	op->dims_vals["oix"] = dims_t(  vect_uint32_t{ no_dims.dsz("chan"), op->stride().d[1], op->stride().d[0] }, 
				    vect_string{ "in_chan", "sy", "sx" }, 1 );
	op->dims_vals["pix"] = dims_t(  vect_uint32_t{ no_dims.dsz("img"), 
	      uint32_t(bck_pels_sz.d[1]), uint32_t(bck_pels_sz.d[0]) }, vect_string{ "img", "y", "x" }, 1 );
	gbt.init( t_tile_sz, max_tpb, u32_pt_t( op->dims_vals["pix"].dims_prod(), op->dims_vals["oix"].dims_prod()));
	dims_t work;
	work.add_dims( "pels_blk", gbt.num_blk.d[0] );
	work.add_dims( "out_ix_blk", gbt.num_blk.d[1] );
	work.add_dims( "pels_tile", gbt.thr_per_blk.d[0] );
	work.add_dims( "out_ix_tile", gbt.thr_per_blk.d[1] );
	work.add_dims( "pels", gbt.mn_per_thr.d[0], "out_ix", gbt.mn_per_thr.d[1] );
	work.calc_strides();
	op->dims_vals["work"] = work;
	op->dims_vals["fioc"] = dims_t( vect_uint32_t{ ogld.dsz("chan"), u32_ceil_div(kern_sz_.d[1],op->stride().d[1]), 
	      u32_ceil_div(kern_sz_.d[0],op->stride().d[0]) }, vect_string{"out_chan","ky","kx"}, 1 );
	  
	gbt_tile_t gbt_fb;
	gbt_fb.init( t_tile_sz, max_tpb, u32_pt_t( fgld.dsz("in_chan")*fgld.dsz("y")*fgld.dsz("x"), fgld.dsz("out_chan") ) );
	dims_t work_fb;
	work_fb.add_dims( "pels_blk", gbt_fb.num_blk.d[0] );
	work_fb.add_dims( "out_ix_blk", gbt_fb.num_blk.d[1] );
	work_fb.add_dims( "pels_tile", gbt_fb.thr_per_blk.d[0] );
	work_fb.add_dims( "out_ix_tile", gbt_fb.thr_per_blk.d[1] );
	work_fb.add_dims( "pels", gbt_fb.mn_per_thr.d[0], "out_ix", gbt_fb.mn_per_thr.d[1] );
	work_fb.calc_strides();
	op->dims_vals["work_fb"] = work_fb;
	op->dims_vals["fioc_fb"] = dims_t( vect_uint32_t{ ogld.dsz("img"), ogld.dsz("y"), ogld.dsz("x") },
				       vect_string{"img","y","x"}, 1 );

      }
      if( is_conv ) {
	// calc_blocking_conv()
	uint32_t const out_ix_sz = no_dims.dims_prod();
	uint32_t const pels_sz = out_ix_sz / no_dims.dsz("chan");
	assert_st( pels_sz * no_dims.dsz("chan") == out_ix_sz ); // by construction
	gbt_tile_t gbt;
	gbt.init( t_tile_sz, max_tpb, u32_pt_t( pels_sz, no_dims.dsz("chan") ) );
	dims_t work;
	uint32_t const lines_sz = no_dims.dsz("img") * no_sz.d[1];
	if( op->cts() == tconv_str ) {
	  assert( gbt.thr_per_blk.d[0] >= 2 ); // if 1, would imply tconv_blk_max_imgs = 1 (but not sensible?)
	  work.add_dims( "blk_bline", u32_ceil_div( lines_sz, gbt.thr_per_blk.d[0] ), 
			 "blk_bx", u32_ceil_div( no_sz.d[0], gbt.mn_per_thr.d[0] ) );
	  uint32_t tconv_blk_max_imgs = 0;
	  uint32_t blk_b_line = 0;
	  for( uint32_t i = 0; i != work.dsz("blk_bline"); ++i ) {
	    uint32_t const blk_e_line = blk_b_line + gbt.thr_per_blk.d[0] - 1;
	    uint32_t const blk_b_img = blk_b_line / no_sz.d[1];
	    uint32_t const blk_e_img = std::min( no_dims.dsz("img") - 1, blk_e_line / no_sz.d[1] );
	    uint32_t const blk_num_img = blk_e_img - blk_b_img + 1;
	    assert_st( blk_num_img );
	    max_eq( tconv_blk_max_imgs, blk_num_img );
	    blk_b_line = blk_e_line + 1;
	  }
	  assert_st( tconv_blk_max_imgs );
	  // calc conservative value (may be lower in general or for some num_imgs) and use as check:
	  uint32_t const conservative_conv_max_img_per_blk = 2 + ((gbt.thr_per_blk.d[0] - 2)/no_sz.d[1]); 
	  assert_st( tconv_blk_max_imgs <= conservative_conv_max_img_per_blk );
	  //printf( "no_sz.d[1]=%s thr_per_blk.d[0]=%s\n", str(no_sz.d[1]).c_str(), str(thr_per_blk.d[0]).c_str() );
	  //printf( "tconv_blk_max_imgs=%s\n", str(tconv_blk_max_imgs).c_str() );
	  assert( gbt.thr_per_blk.d[0] >= tconv_blk_max_imgs );
	  uint32_t const tconv_blk_max_in_lines = (gbt.thr_per_blk.d[0] - tconv_blk_max_imgs)*
	    op->stride().d[1] + kern_sz_.d[1]*tconv_blk_max_imgs;
	  uint32_t const tconv_blk_x_sz = (gbt.mn_per_thr.d[0] - 1)*op->stride().d[0] + kern_sz_.d[0];
	  // the tconv/in_tile_xpose format is for use when both ni_sz.d[0/1] are small multiple of
	  // gbt.mn_per_thr.d[0]/gbt.thr_per_blk.d[0] or >> than them (to avoid wasting too much work). each block will handle a
	  // (x,y) window of the output of size (gbt.mn_per_thr.d[0],gbt.thr_per_blk.d[0]) across bix_pels_blk_sz*gbt.mn_per_thr.d[0]
	  // output chans. in this case, we do not unroll across input chans, but we do unroll across kern_sz in X
	  // (and maybe in Y too for small kernels).  note: "out_ix" from in_tile_xpose becomes "in_ix" for tconv;
	  // from the perspective inside tconv: the blk_y and blk_x dims are in input image space, the other dims are
	  // in output space image space. other x/y's (in thread and block indexes) are all in output image space.
	  in_dims = dims_t( vect_uint32_t{
	      work.dsz("blk_bline"), work.dsz("blk_bx"), ni_dims.dsz("chan"), tconv_blk_max_in_lines, tconv_blk_x_sz },
	    vect_string{"blk_bline","blk_bx","blk_in_chan","blk_y","blk_x"}, 1 );
	} else {
	  work.add_dims( "pels_blk", gbt.num_blk.d[0] );
	}
	work.add_dims( "out_chan_blk", gbt.num_blk.d[1] );

	// dims of per-group work (defines # threads per local group)
	if( op->cts() == tconv_str ) { work.add_dims( "blk_y", gbt.thr_per_blk.d[0] ); }
	else { work.add_dims( "pels_tile", gbt.thr_per_blk.d[0] ); }
	work.add_dims(   "out_chan_tile", gbt.thr_per_blk.d[1] );

	work.add_dims( "pels", gbt.mn_per_thr.d[0], "out_chan", gbt.mn_per_thr.d[1] ); // dims of per-thread work
	if( op->cts() == ipconv_str ) { 
	  uint32_t fioc_tile = 4;
	  while( (fioc_tile < 32) && (fioc_tile*2*gbt.thr_per_blk.dims_prod()) <= 512 ) { fioc_tile *= 2; }
	  assert_st( (ni_dims.dsz("chan") % fioc_tile) == 0 );
	  work.add_dims( "fioc_tile", fioc_tile ); 
	} // unrolling/tiling of inner loop
	work.calc_strides();

        uint32_t out_chan_pad = op->dims_vals["filts"].dsz("out_chan"); // not padded yet but may be layer
        uint32_t in_chan_pad = ni_dims.dsz("chan"); // not padded yet but may be layer; note: == filts in_chan dim
	if( op->cts() == k1conv_str ) { 
	  uint32_t const in_blk_iter_chan_dim = op_tune.Kb;
	  // the k1conv/xpose_in format is for use when stride=1, kernel_sz=1, and in_pad=0. we treat all input pixels as one 1D
	  // vector across img:y:x, and divide them into blocks. we also block in the chan dim for unrolling.
	  in_dims = dims_t( vect_uint32_t{
	      work.dsz("pels_blk"), u32_ceil_div(ni_dims.dsz("chan"),in_blk_iter_chan_dim), in_blk_iter_chan_dim, work.dsz("pels_tile")*work.dsz("pels")}, 
	    vect_string{"blk","blk_iter","blk_iter_chan","blk_pel"}, 1 ); 
	} else if( op->cts() == k1conv_simd_str ) { 
          must_insert( op->str_vals, "vw", str(op_tune.vw) );
          // FIXME/NOTE: WIP to become simd, no-local-mem version of k1conv -- but initially with no SIMD for SIMD,
          // we'll need to pad in_chans to a multiple of Kb (which in turn should be a mult of vw), but for now we don't
          // need to.  we also xpose to pels:chans format for both the filts and input. each blk will handle:
          uint32_t const blk_pels = work.dsz("pels_tile")*work.dsz("pels");
          // if the output is not an exact number of pels blks, add enough images to at least fill the final partial blk
          uint32_t const in_blks = u32_ceil_div( pels_sz, blk_pels );
          uint32_t const img_pels = pels_sz / ni_dims.dsz("img");
          assert_st( img_pels * ni_dims.dsz("img") == pels_sz ); // by construction
          uint32_t const in_imgs_pad = u32_ceil_div( in_blks*blk_pels, img_pels );
	  must_insert( op->str_vals, "Kb", str(op_tune.Kb) );
          // if the output is not an exact number of out_chan blks, add enough out_chans to make it exact
          out_chan_pad = u32_ceil_align( out_chan_pad, work.dsz("out_chan_tile")*work.dsz("out_chan") );
          in_dims = dims_t( vect_uint32_t{ in_chan_pad, in_imgs_pad, ni_dims.dsz("y"), ni_dims.dsz("x") }, 
                            vect_string{"chan","img","y","x"}, 1 ); 
          // note: for now, we don't pad and/or xpose out, so the store code must handle that.
	}
	op->dims_vals["work"] = work;
	// k1conv and in_tile_xpose need the standard output dims for reference. curently this == the dims of "out",
	// but a later pass might change the desired output format by changing the "out" dims. however, the "out_ref"
	// dims will remain unchanged at this value.
	op->dims_vals["out_ref"] = no_dims; 
	// set final desired format for input. note: (1) original 'standard' format is stored as "in_ref" earlier (2)
	// the dims of "in" may now differ from the dims of the global/rtc variable in the arg_map that "in" is bound
	// to. our convention is that we expect or detect this in codegen and emit the need xform at that point. when
	// we do this, we may change the binding for "in" (e.g. to point to an xformed version of the original
	// variable).
	op->dims_vals["in"] = in_dims; 
	// 'standard' and desired/xformed filter dims. we don't currently xform the biases (although maybe we should).
	op->dims_vals["filts_ref"] = op->dims_vals["filts"];
        if( op->cts() == k1conv_simd_str ) {
          // match padded chans of input
	  op->dims_vals["filts"] = dims_t( 
            vect_uint32_t{ in_chan_pad, kern_sz_.d[1], kern_sz_.d[0], out_chan_pad }, 
            vect_string{"in_chan","y","x","out_chan"}, 1 ); 

        } else if( op->cts() != ipconv_str ) { 
	  op->dims_vals["filts"] = dims_t( vect_uint32_t{ work.dsz("out_chan_blk"),ni_dims.dsz("chan"), kern_sz_.d[1], kern_sz_.d[0],
		work.dsz("out_chan"),work.dsz("out_chan_tile")}, vect_string{"out_chan_blk","in_chan","y","x","out_chan_reg","out_chan_tile"}, 1 );
	}
	// dims_t( vect_uint32_t{out_chans}, vect_string{"out_chan"}, 1 );
      } // end if(is_conv)
    }
  }
  
#include"gen/cnn_op.H.nesi_gen.cc"
}
