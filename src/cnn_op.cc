// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"cnn_op.H"
#include"str_util.H"
#include"gbt_tile.H"
#include"nesi.H"
#include"lexp.H"
#include<sstream>

namespace boda 
{

  // based on the semantics/type of the operation represented by conv_op_base_t and the code generation parameters,
  // annotate the operation with the information needed for the next stage of code generation. In particular, this code
  // selects an operation variant and selects tuning paramters for this operations for that variant.
  void add_cnn_codegen_annotations( conv_op_base_t * const op, op_tune_t const & op_tune_,
                                    map_str_op_tune_t const *per_op_tune )
  {
    op_tune_t const * op_tune = &op_tune_;

    bool const enable_ipconv = op_tune->ipconv;
    bool const enable_k1conv = op_tune->k1conv;
    bool const enable_tconv = op_tune->tconv;
    bool const force_enable_tconv = (op_tune->tconv==2);
    dims_t ni_dims;
    dims_t no_dims = op->get_dims( op->coi->top_an(0) );
    u32_pt_t const no_sz = get_xy_dims( no_dims );
    if( !op->is(Concat_coi) ) { ni_dims = op->get_dims( op->coi->bot_an(0) ); } // empty for Concat where we shouldn't use it, otherwise first input
    bool const is_conv = op->is( Convolution_coi );
    bool const is_pool = op->is( Pooling_coi );
    dims_t in_dims;
    if( op->is( Dropout_coi ) || op->is( BckDropout_coi )){ op->set_dims("det_drop_seed",make_scalar_dims_t("uint32_t")); }

    if( is_conv || is_pool || op->is( Spreading_coi ) || op->is( BckConv_coi ) ) {
      assert_st( !ni_dims.empty() );
      in_dims = ni_dims;
      op->set_dims("in_ref",in_dims); // tconv needs the standard input dims for reference
      u32_pt_t kern_sz_;
      if( op->has("kern_sz") ) { kern_sz_ = op->kern_sz(); }
      else {
	if( is_pool ) { kern_sz_ = get_xy_dims( ni_dims ); }
	else if( op->is( Spreading_coi ) ) { kern_sz_ = get_xy_dims( no_dims ); }
	else { assert_st(0); }
	op->set_dims("kern_sz", dims_t{ {kern_sz_.d[1],kern_sz_.d[0]}, {"y","x"}, "none" } ); // FIXME: not ideal ...
      } 
      if( is_conv ) { // set func_name (aka variant) for conv case (others are set at bottom)
        if( op_tune->use_culibs ) { 
          op->set_func_name("cudnn_conv");
        } else if( enable_ipconv && op->in_pad().is_zeros() && (get_xy_dims(no_dims) == u32_pt_t{1,1}) ) {
          op->set_func_name( ipconv_str ); // single output per-chan-per-image: inner-product case
        } else if( enable_k1conv && (kern_sz_ == u32_pt_t{1,1}) && (op->stride() == u32_pt_t{1,1}) 
                   && (no_sz.d[0] >= 6) && (no_sz.d[0] <= 300 ) && (no_dims.dsz("chan") >= 64) ) 
        { 
          if( !op->in_pad().is_zeros() ) { printf( "warning: can't use k1conv due only to non-zero padding on layer with kernel size 1\n" ); op->set_func_name( conv_str ); }
          else { 
            if( op_tune->use_local_mem == 2 ) { op->set_func_name( k1conv_simd_str ); }
            else { op->set_func_name( k1conv_str ); }
          }
        }
        else if( enable_tconv && (force_enable_tconv || ( kern_sz_.both_dims_le(op_tune->tconv_max_ksz)
                                                          && (kern_sz_.both_dims_ge(u32_pt_t{1,1}) && (no_sz.d[0] >= 6))))) {
          op->set_func_name( tconv_str );
        }
        else { 
          if( op_tune->use_local_mem == 2 ) { op->set_func_name( conv_simd_str ); }
          else { op->set_func_name( conv_str ); }
        }
      }
      // NOTE/FIXME: this whole per_op_tune thing is hacky and experimental ... but we do need to do something. i guess
      // we're creeping toward what we need for supporting autotuning in the long run, but we need to figure out the
      // clean way to do things. there seem to really be multiple stages of tuning paramers. in this case, we use the
      // 'base' op_tune to select a variant (which uses some subset of the tuning params, mostly enable related ones),
      // and then if there's a per-variant op_tune we use it for some remaining params (mostly blocking related
      // ones). so maybe we need to split op_tune_t? or not?
      if( per_op_tune ) { op_tune = &get(*per_op_tune,op->get_func_name(),op_tune_); };

      u32_pt_t const & t_tile_sz = op_tune->MNt;
      uint32_t const max_tpb = op_tune->MNb.dims_prod(); // FIXME: pass as M/N parts, not product.

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

	op->set_dims("bck_in_pad",dims_t( vect_uint32_t{ uint32_t(bck_in_pad.d[1]), uint32_t(bck_in_pad.d[0]) }, 
                                          vect_string{"y","x"}, "none" ));
	op->set_dims("bck_pad_in_off",dims_t( vect_uint32_t{ bck_pad_in_off.d[1], bck_pad_in_off.d[0] }, vect_string{"y","x"}, "none" ));

	dims_t const & ogld = op->get_dims("out_grad_loss");
	dims_t const & fgld = op->get_dims("filts_grad_loss");

	gbt_tile_t gbt;
	op->set_dims("oix",dims_t(  vect_uint32_t{ no_dims.dsz("chan"), op->stride().d[1], op->stride().d[0] }, 
                                    vect_string{ "in_chan", "sy", "sx" }, "none" ));
	op->set_dims("pix",dims_t(  vect_uint32_t{ no_dims.dsz("img"), 
                uint32_t(bck_pels_sz.d[1]), uint32_t(bck_pels_sz.d[0]) }, vect_string{ "img", "y", "x" }, "none" ));
	gbt.init( t_tile_sz, max_tpb, u32_pt_t( op->get_dims("pix").dims_prod(), op->get_dims("oix").dims_prod()));
	dims_t work;
        work.tn = "none";
	work.add_dims( "pels_blk", gbt.num_blk.d[0] );
	work.add_dims( "out_ix_blk", gbt.num_blk.d[1] );
	work.add_dims( "pels_tile", gbt.thr_per_blk.d[0] );
	work.add_dims( "out_ix_tile", gbt.thr_per_blk.d[1] );
	work.add_dims( "pels", gbt.mn_per_thr.d[0], "out_ix", gbt.mn_per_thr.d[1] );
	work.calc_strides();
	op->set_dims("work",work);
	op->set_dims("fioc",dims_t( vect_uint32_t{ ogld.dsz("chan"), u32_ceil_div(kern_sz_.d[1],op->stride().d[1]), 
                u32_ceil_div(kern_sz_.d[0],op->stride().d[0]) }, vect_string{"out_chan","ky","kx"}, "none" ));
	  
	gbt_tile_t gbt_fb;
	gbt_fb.init( t_tile_sz, max_tpb, u32_pt_t( fgld.dsz("in_chan")*fgld.dsz("y")*fgld.dsz("x"), fgld.dsz("out_chan") ) );
	dims_t work_fb;
        work_fb.tn = "none";
	work_fb.add_dims( "pels_blk", gbt_fb.num_blk.d[0] );
	work_fb.add_dims( "out_ix_blk", gbt_fb.num_blk.d[1] );
	work_fb.add_dims( "pels_tile", gbt_fb.thr_per_blk.d[0] );
	work_fb.add_dims( "out_ix_tile", gbt_fb.thr_per_blk.d[1] );
	work_fb.add_dims( "pels", gbt_fb.mn_per_thr.d[0], "out_ix", gbt_fb.mn_per_thr.d[1] );
	work_fb.calc_strides();
	op->set_dims("work_fb",work_fb);
	op->set_dims("fioc_fb",dims_t( vect_uint32_t{ ogld.dsz("img"), ogld.dsz("y"), ogld.dsz("x") },
                                       vect_string{"img","y","x"}, "none" ));

      }
      if( is_conv && (op->get_func_name() != "cudnn_conv") ) { // skip blocking setup for cudnn_conv
        // 'standard' and desired/xformed filter dims. we don't currently xform the biases (although maybe we should).
        op->set_dims("filts_ref",op->get_dims("filts"));

	// calc_blocking_conv()
	uint32_t const out_ix_sz = no_dims.dims_prod();
	uint32_t const pels_sz = out_ix_sz / no_dims.dsz("chan");
	assert_st( pels_sz * no_dims.dsz("chan") == out_ix_sz ); // by construction
	gbt_tile_t gbt;
	gbt.init( t_tile_sz, max_tpb, u32_pt_t( pels_sz, no_dims.dsz("chan") ) );
	dims_t work;
        work.tn = "none";
	uint32_t const lines_sz = no_dims.dsz("img") * no_sz.d[1];
        if( (op->get_func_name() == tconv_str) || (op->get_func_name() == k1conv_str) ) { 
          op->set_dims("flags",make_scalar_dims_t("uint32_t")); // exactly these two variants have this debugging input
        }
	if( op->get_func_name() == tconv_str ) {
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
	    vect_string{"blk_bline","blk_bx","blk_in_chan","blk_y","blk_x"}, in_dims.tn );
	} else {
	  work.add_dims( "pels_blk", gbt.num_blk.d[0] );
	}
	work.add_dims( "out_chan_blk", gbt.num_blk.d[1] );

	// dims of per-group work (defines # threads per local group)
	if( op->get_func_name() == tconv_str ) { work.add_dims( "blk_y", gbt.thr_per_blk.d[0] ); }
	else { work.add_dims( "pels_tile", gbt.thr_per_blk.d[0] ); }
	work.add_dims(   "out_chan_tile", gbt.thr_per_blk.d[1] );

	work.add_dims( "pels", gbt.mn_per_thr.d[0], "out_chan", gbt.mn_per_thr.d[1] ); // dims of per-thread work
	if( op->get_func_name() == ipconv_str ) { 
	  uint32_t fioc_tile = 4;
	  while( (fioc_tile < 32) && (fioc_tile*2*gbt.thr_per_blk.dims_prod()) <= 512 ) { fioc_tile *= 2; }
	  assert_st( (ni_dims.dsz("chan") % fioc_tile) == 0 );
	  work.add_dims( "fioc_tile", fioc_tile ); 
	} // unrolling/tiling of inner loop
	work.calc_strides();

	if( op->get_func_name() == k1conv_str ) { 
	  uint32_t const in_blk_iter_chan_dim = op_tune->Kb;
	  // the k1conv/xpose_in format is for use when stride=1, kernel_sz=1, and in_pad=0. we treat all input pixels as one 1D
	  // vector across img:y:x, and divide them into blocks. we also block in the chan dim for unrolling.
	  in_dims = dims_t( vect_uint32_t{
	      work.dsz("pels_blk"), u32_ceil_div(ni_dims.dsz("chan"),in_blk_iter_chan_dim), in_blk_iter_chan_dim, work.dsz("pels_tile")*work.dsz("pels")}, 
	    vect_string{"blk","blk_iter","blk_iter_chan","blk_pel"}, in_dims.tn ); 
	} else if( op->get_func_name() == k1conv_simd_str ) { 
          // currently, the codegen/code doesn't support some cases. so, for those, bail. FIXME: how to sync this with the codegen?
          if( (work.dsz("pels") % op_tune->vw) != 0 ) { unsup_err( "k1conv_simd only supports work.pels being a multiple of vw" ); }
          if( (work.dsz("out_chan") % op_tune->vw) != 0 ) {unsup_err( "k1conv_simd only supports work.out_chan being a multiple of vw" ); }
          op->set_u32( "vw", op_tune->vw );
          // simd, no-local-mem version of k1conv.  we xpose to pels:chans format for the file, input, and output. we
          // pad the pels and out_chans to exactly match the blocking.
          uint32_t const pels_sz_pad = work.dsz("pels_blk")*work.dsz("pels_tile")*work.dsz("pels");
          assert_st( pels_sz_pad >= pels_sz );
          uint32_t const out_chan_pad = work.dsz("out_chan_blk")*work.dsz("out_chan_tile")*work.dsz("out_chan");
          assert_st( out_chan_pad >= op->get_dims("filts").dsz("out_chan") );
          // FIXME: pad in_chan to multiple of Kb?
          op->set_u32( "Kb", op_tune->Kb );
          uint32_t in_chan_pad = ni_dims.dsz("chan"); // not padded yet but may be layer; note: == filts in_chan dim
          in_dims = dims_t( vect_uint32_t{ in_chan_pad, pels_sz_pad }, vect_string{"chan","pel"}, in_dims.tn ); 
          // note: for now, we don't pad and/or xpose out, so the store code must handle that.
	  op->reset_dims("filts",dims_t( 
            vect_uint32_t{ in_chan_pad, kern_sz_.d[1], kern_sz_.d[0], out_chan_pad }, 
            vect_string{"in_chan","y","x","out_chan"}, op->get_dims("filts").tn )); 
	  op->reset_dims("out",dims_t( vect_uint32_t{ out_chan_pad, pels_sz_pad }, 
                                       vect_string{"chan","pel"}, op->get_dims("out").tn ));
	} else if( op->get_func_name() == conv_simd_str ) {
          op->set_u32( "vw", op_tune->vw );
          if( (work.dsz("pels") % op_tune->vw) != 0 ) { unsup_err( "conv_simd only supports work.pels being a multiple of vw" ); }
          if( (work.dsz("out_chan") % op_tune->vw) != 0 ) {unsup_err( "conv_simd only supports work.out_chan being a multiple of vw" ); }
          if( op_tune->Kb != 1 ) { unsup_err( "conv_simd only supports Kb == 1" ); } // FIXME: for now, no inner loop unroll ...
          // FIXME: pad in_chan to multiple of Kb?
          assert_st( op_tune->Kb == 1 ); // FIXME: for now, no inner loop unroll ...
          op->set_u32( "Kb", op_tune->Kb );
          uint32_t in_chan_pad = ni_dims.dsz("chan"); // not padded yet but may be layer; note: == filts in_chan dim
          // calculate padded x/y dims
          u32_pt_t in_xy = get_xy_dims( in_dims );
          in_xy += op->in_pad(); // add pre-padding (will also function as post-padding for prior y value
          // FIXME/NOTE: most things below work when the stride is different in X and Y. however, determining the
          // offsets into in inside the kernel is harder if we can't do it using a linear indexing system (i.e. in_pel =
          // out_pel*uniform_stride). so, for now, we only allow uniform X/Y stride, and note this in the kernel
          assert_st( op->stride().dims_are_same() ); 
          in_xy = ceil_align( in_xy, op->stride() ); // align to stride
          u32_pt_t out_xy = in_xy / op->stride(); // will divide exactly by construction
          // these are (for reference) the nested dims for the pel dim of in/out
	  op->set_dims("in_pels",dims_t( vect_uint32_t{ in_dims.dsz("img"), in_xy.d[1], in_xy.d[0] },
                                         vect_string{"img","y","x"}, "none" )); 
	  op->set_dims("out_pels",dims_t( vect_uint32_t{ in_dims.dsz("img"), out_xy.d[1], out_xy.d[0] },
                                          vect_string{"img","y","x"}, "none" )); // min output pels we must calculate
          work = dims_t(); // need to recalculate work. FIXME: need to refactor to make this cleaner across cases
          work.tn = "none";
          gbt_tile_t gbt;
          gbt.init( t_tile_sz, max_tpb, u32_pt_t( op->get_dims("out_pels").dims_prod(), no_dims.dsz("chan") ) );
	  work.add_dims( "pels_blk", gbt.num_blk.d[0], "out_chan_blk", gbt.num_blk.d[1] );
          work.add_dims( "pels_tile", gbt.thr_per_blk.d[0], "out_chan_tile", gbt.thr_per_blk.d[1] );
          work.add_dims( "pels", gbt.mn_per_thr.d[0], "out_chan", gbt.mn_per_thr.d[1] ); // dims of per-thread work
          work.calc_strides();

          uint32_t const pels_sz_pad = work.dsz("pels_blk")*work.dsz("pels_tile")*work.dsz("pels"); // final pels we will calc
          assert_st( pels_sz_pad >= op->get_dims("out_pels").dims_prod() );
          uint32_t const out_chan_pad = work.dsz("out_chan_blk")*work.dsz("out_chan_tile")*work.dsz("out_chan");
          assert_st( out_chan_pad >= op->get_dims("filts").dsz("out_chan") );

          // in needs final x/y padding + extra pels padding depending on the filter size. for correctness, we would
          // only need in_pad padding in X/Y, since no valid output should use more than that. however, to avoid invalid
          // memory access, we need (kern_sz_ - stride) pels of padding, since that's how far the last output pixel will
          // hang off the last image. so we max over those two quantities (either could be higher).
          u32_pt_t const fin_pad_xy = max( kern_sz_ - op->stride(), op->in_pad() );
          uint32_t filt_dep_extra_in_pad = (fin_pad_xy.d[1])*op->get_dims("in_pels").dstride("y") +
            (fin_pad_xy.d[0])*op->get_dims("in_pels").dstride("x"); 
          uint32_t final_in_pels_pad = u32_ceil_align( op->get_dims("in_pels").dims_prod()+filt_dep_extra_in_pad,op_tune->vw);
          in_dims = dims_t( vect_uint32_t{ in_chan_pad, final_in_pels_pad }, vect_string{"chan","pel"}, in_dims.tn ); 
          // note: for now, we don't pad and/or xpose out, so the store code must handle that.
	  op->reset_dims("filts",dims_t( 
            vect_uint32_t{ in_chan_pad, kern_sz_.d[1], kern_sz_.d[0], out_chan_pad }, 
            vect_string{"in_chan","y","x","out_chan"}, op->get_dims("filts").tn )); 
	  op->reset_dims("out",dims_t( vect_uint32_t{ out_chan_pad, pels_sz_pad }, vect_string{"chan","pel"}, 
                                     op->get_dims("out").tn )); 
        }
	op->set_dims("work",work);
	// k1conv and in_tile_xpose need the standard output dims for reference. curently this == the dims of "out",
	// but a later pass might change the desired output format by changing the "out" dims. however, the "out_ref"
	// dims will remain unchanged at this value.
	op->set_dims("out_ref",no_dims); 
	// set final desired format for input. note: (1) original 'standard' format is stored as "in_ref" earlier (2)
	// the dims of "in" may now differ from the dims of the global/rtc variable in the arg_map that "in" is bound
	// to. our convention is that we expect or detect this in codegen and emit the need xform at that point. when
	// we do this, we may change the binding for "in" (e.g. to point to an xformed version of the original
	// variable).
        op->reset_dims("in",in_dims); 
        if( is_k1_or_t_or_reg_conv( op->get_func_name() ) ) {
          op->reset_dims("filts",dims_t( vect_uint32_t{ work.dsz("out_chan_blk"),ni_dims.dsz("chan"), 
                  kern_sz_.d[1], kern_sz_.d[0],
                  work.dsz("out_chan"),work.dsz("out_chan_tile")}, vect_string{"out_chan_blk","in_chan","y","x",
                                                                       "out_chan_reg","out_chan_tile"}, 
              op->get_dims("filts").tn ));
          }
	// dims_t( vect_uint32_t{out_chans}, vect_string{"out_chan"}, 1 );
      } // end if(is_conv)
    }
    // if not set yet: set func name if a single value works, otherwise leave unset. FIXME: put these values into coi?
    if( 0 ) { }
    else if( op->is( Concat_coi ) ) { op->set_func_name( "copy" ); }
    else if( op->is( Split_coi ) ) { op->set_func_name( "split_copy" ); }
    else if( op->is( Reduce_coi ) ) { op->set_func_name( "reduce" ); }
    else if( op->is( Pooling_coi ) ) { op->set_func_name( "pool" ); }
    // else if( op->is( Convolution_coi ) ) { } // selected above already
    else if( op->is( ReLU_coi ) ) { op->set_func_name( "relu" ); }
    else if( op->is( LRN_coi ) ) { op->set_func_name( "lrn" ); }
    else if( op->is( BckLRN_coi ) ) { op->set_func_name( "bck_lrn" ); }
    else if( op->is( Dropout_coi ) ) { op->set_func_name( "dropout" ); }
    else if( op->is( BckDropout_coi ) ) { op->set_func_name( "dropout" ); }
    else if( op->is( Softmax_coi ) ) { op->set_func_name( "softmax" ); }
    // else if( op->is( SoftmaxWithLoss_coi ) ) { } // no single value
    else if( op->is( Spreading_coi ) ) { op->set_func_name( "spreading" ); }
    else if( op->is( ZeroIfNonPos_coi ) ) { op->set_func_name( op->get_type() ); } // wow! one with a consistent name! sigh.
    // else if( op->is( BckConv_coi ) ) { } // no single value
  }

  // FIXME: merge with add_cnn_codegen_annotations() wholesale? split into sub-funcs based on types with a dispatcher?
  void add_codegen_annotations( p_conv_op_base_t const & anno_op, op_tune_t const & op_tune, map_str_op_tune_t const *per_op_tune ) {
    if( anno_op->is( Convolution_coi ) ) {
      add_cnn_codegen_annotations( anno_op.get(), op_tune, per_op_tune );
      anno_op->set_u32( "conv_has_relu", 1 );
    } else if( anno_op->is( sgemm_coi ) ) {
      if( op_tune.use_culibs ) {
        anno_op->set_func_name("cublas_sgemm");
      } else {
        uint64_t const K = anno_op->get_dims("a").dsz("K"); // note == b.dsz("K")
        dims_t const & c = anno_op->get_dims("c");
        uint64_t const M_blk = op_tune.MNb.d[0] * op_tune.MNt.d[0];
        uint64_t const N_blk = op_tune.MNb.d[1] * op_tune.MNt.d[1];
        uint64_t const Mg = c.dsz("M") / M_blk;
        uint64_t const Ng = c.dsz("N") / N_blk;
        if( Mg * M_blk != c.dsz("M") ) { 
          rt_err( strprintf( "FIXME: currently, M=%s must be a multiple of M_blk=%s\n", 
                             str(c.dsz("M")).c_str(), str(M_blk).c_str() ) );
        }
        if( Ng * N_blk != c.dsz("N") ) { 
          rt_err( strprintf( "FIXME: currently, N=%s must be a multiple of N_blk=%s\n", 
                             str(c.dsz("N")).c_str(), str(N_blk).c_str() ) );
        }
        if( K % op_tune.Kb ) { 
          rt_err( strprintf( "FIXME: currently, K=%s must be a multiple of Kb=%s\n", 
                             str(K).c_str(), str(op_tune.Kb).c_str() ) );
        }

        dims_t work{ {(uint32_t)Mg,(uint32_t)Ng,op_tune.MNb.d[0],op_tune.MNb.d[1],op_tune.Kb,
              op_tune.MNt.d[0],op_tune.MNt.d[1]}, {"Mg","Ng","Mb","Nb","Kb","Mt","Nt"}, "none" };
        anno_op->set_dims( "work", work );
        anno_op->set_u32( "use_local_mem", op_tune.use_local_mem );
        anno_op->set_u32( "prof_variant", op_tune.prof_variant );
        anno_op->set_u32( "vw", op_tune.vw );
        if( op_tune.prof_variant ) { anno_op->set_func_name("sgemm_prof"); } 
        else {
          if( 0 ) { }
          else if( op_tune.use_local_mem == 0 ) { anno_op->set_func_name("sgemm_no_local"); }
          else if( op_tune.use_local_mem == 1 ) { anno_op->set_func_name("sgemm"); }
          else if( op_tune.use_local_mem == 2 ) { anno_op->set_func_name("sgemm_simd"); }
          else if( op_tune.use_local_mem == 3 ) { anno_op->set_func_name("sgemm_simd_local"); }
          else { rt_err( strprintf( "unknonw value for op_tune.use_local_mem of %s\n", 
                                    str(op_tune.use_local_mem).c_str() ) ); }
          
        }
      }	  
    }
  }

#include"gen/cnn_op.H.nesi_gen.cc"
}
