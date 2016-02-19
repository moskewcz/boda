// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"gbt_tile.H"
#include"str_util.H"
#include"has_conv_fwd.H"
#include"timers.H"
#include<boost/filesystem.hpp>
#include<boost/lexical_cast.hpp>
#include"lexp.H"
#include"conv_util.H"

#include"rtc_func_gen.H"
#include"rtc_compute.H"

namespace boda 
{
  struct quantize_ops_t : virtual public nesi // NESI(help="per-layer quantization options") 
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string name; //NESI(help="name of node to apply operation to",req=1)
    uint32_t max_val; //NESI(help="clamp value to this maximum",req=1)
    uint32_t keep_bits; //NESI(help="after clamping, keep this many high bits",req=1)
  };
  typedef vector< quantize_ops_t > vect_quantize_ops_t; 
  typedef shared_ptr< quantize_ops_t > p_quantize_ops_t; 
  typedef vector< p_quantize_ops_t > vect_p_quantize_ops_t;

  struct op_info_t;
  typedef shared_ptr< op_info_t > p_op_info_t; 
  typedef map< string, p_op_info_t > map_str_p_op_info_t;
  typedef shared_ptr< map_str_p_op_info_t > p_map_str_p_op_info_t; 

  string const k1conv_str = "k1conv"; string const tconv_str = "tconv"; string const ipconv_str = "ipconv"; string const conv_str = "conv";
  struct op_info_t {
    string tag;
    map_str_str template_var_values; // str->str templates+values to pass directly to generated code (e.g. lrn params)
    map_str_dims_t conv_ref_dims; // work + conv-type specific dims
    map_str_str arg_map; // map from func arg names to call-site arg names (in this case, just in global/rtc scope)
    string cts; // cts --> conv-type-str

    string get_arg( string const & an ) { return must_find( arg_map, an ); }
    dims_t const & get_arg_dims( string const & an ) { return must_find( conv_ref_dims, an ); }
    void set_arg( p_rtc_compute_t const & rtc, string const & an, string const & vn ) {
      must_insert( conv_ref_dims, an, rtc->get_var_dims_floats(vn) );
      must_insert( arg_map, an, vn );
    }
    void set_null_arg( string const & an ) { must_insert( conv_ref_dims, an, dims_t() ); }
    void erase_arg( string const & an ) { must_erase( conv_ref_dims, an ); must_erase( arg_map, an ); }
    void reset_arg( p_rtc_compute_t const & rtc, string const & an, string const & vn ) { 
      erase_arg(an); set_arg(rtc,an,vn);
    }

    void init( p_conv_pipe_t const & cp, p_conv_op_t const & cop, bool const & enable_ipconv,
	       bool const & enable_k1conv, bool const & enable_tconv, bool const & force_enable_tconv,
	       uint32_t const t_tile_sz ) {
      tag = cop->tag;
      template_var_values = cop->params;
      assert_st( cop->tops.size() >= 1 );
      assert_st( cop->bots.size() >= 1 );
      // add all bots/tops as ref dims and track the mapping from arg name to external (call-scope) name
      for( uint32_t i = 0; i != cop->bots.size(); ++i ) { 
	must_insert( conv_ref_dims, cop->coi->bot_an(i), cp->must_get_node( cop->bots[i] )->dims );
	must_insert( arg_map, cop->coi->bot_an(i), cop->bots[i] );
      }
      for( uint32_t i = 0; i != cop->tops.size(); ++i ) { 
	must_insert( conv_ref_dims, cop->coi->top_an(i), cp->must_get_node( cop->tops[i] )->dims ); 
	must_insert( arg_map, cop->coi->top_an(i), cop->tops[i] );
      }
      p_conv_node_t ni;
      p_conv_node_t no = cp->must_get_node( cop->tops[0] );
      u32_pt_t const no_sz = get_xy_dims( no->dims );
      if( !cop->is(Concat_coi) ) { ni = cp->must_get_node( cop->bots[0] ); } // null for Concat where we shouldn't use it, otherwise first input
      bool const is_conv = cop->is( Convolution_coi );
      bool const is_pool = cop->is( Pooling_coi );
      dims_t in_dims;
      if( is_conv || is_pool || cop->is( Spreading_coi ) || cop->is( BckConv_coi ) ) {
	assert_st( ni );
	in_dims = ni->dims;
	conv_ref_dims["in_ref"] = in_dims; // tconv needs the standard input dims for reference
	u32_pt_t kern_sz = cop->kern_sz;
	if( kern_sz.is_zeros() ) { // 'global' input special case
	  if( is_pool ) { kern_sz = get_xy_dims( ni->dims ); }
	  else if( cop->is( Spreading_coi ) ) { kern_sz = get_xy_dims( no->dims ); }
	  else { assert_st(0); }
	} 
	u32_pt_t const in_pad = cop->in_pad;
	u32_pt_t const stride = cop->stride;
	if( is_conv && enable_ipconv && in_pad.is_zeros() && (get_xy_dims(no->dims) == u32_pt_t{1,1}) ) {
	  cts = ipconv_str; // single output per-chan-per-image: inner-product case
	} else if( is_conv && enable_k1conv && (kern_sz == u32_pt_t{1,1}) && (stride == u32_pt_t{1,1}) 
	    && (no_sz.d[0] >= 6) && (no_sz.d[0] <= 300 ) && (no->dims.dsz("chan") >= 64) ) 
	{ 
	  if( !in_pad.is_zeros() ) { printf( "warning: can't use k1conv due only to non-zero padding on layer with kernel size 1\n" ); cts = conv_str; }
	  else { cts = k1conv_str; }
	}
	else if( is_conv && enable_tconv && (force_enable_tconv || ( kern_sz.both_dims_le(u32_pt_t{11,11})
								     && (kern_sz.both_dims_ge(u32_pt_t{1,1}) && (no_sz.d[0] >= 6))))) {
		   cts = tconv_str; }
	else { cts = conv_str; }

	conv_ref_dims["kern_sz"] = dims_t( vect_uint32_t{ kern_sz.d[1], kern_sz.d[0] }, vect_string{"y","x"}, 1 );
	conv_ref_dims["stride"] = dims_t( vect_uint32_t{ stride.d[1], stride.d[0] }, vect_string{"y","x"}, 1 );
	conv_ref_dims["in_pad"] = dims_t( vect_uint32_t{ in_pad.d[1], in_pad.d[0] }, vect_string{"y","x"}, 1 );

	if( cop->is( BckConv_coi ) ) {
	  // note: since fwd strides are always N/1, bck 'strides' are always 1/N, meaning stride in the fwd sense will
	  // always be 1 for the bck conv: 3x3@s2 -> 2x2@s1; 11x11@s4 -> 3x3@s1; 1x1@s1 -> 1x1@s1 ...
	  u32_pt_t bck_kern_sz = ceil_div( kern_sz, stride ); 
	  // if back kernel conv is convolved aligned to the - corner of output space, it yields results for the
	  // post-padding input space region: [bck_in_off,bck_in_off+stride)
	  u32_pt_t const bck_pad_in_off = (bck_kern_sz - u32_pt_t(1,1)) * stride;
	  assert_st( bck_pad_in_off.dims_are_same() );
	  // we don't need compute values for the input padding, so adjust to un-padded input space
	  i32_pt_t bck_in_off = u32_to_i32( bck_pad_in_off ) - u32_to_i32(in_pad);
	  assert_st( bck_in_off.dims_are_same() ); // not handled, since we want/need per-axis padding for that
	  // now, calculate where we need to really start in output space to have the first results region inlcude 0
	  i32_pt_t bck_in_pad = ceil_div( bck_in_off, stride );
	  // FIXME: 'too much' fwd-in-pad can the  bck-in-pad this negative. sensible, so handle?
	  assert_st( bck_in_pad.both_dims_ge_zero() );
	  // now, to get patch count, see how many in pels we're missing
	  bck_in_off -= bck_in_pad * u32_to_i32(stride); // first region calculated at - corner of padding out space
	  // calculate number of extra pels needed to cover last pel in unpadded input space
	  i32_pt_t bck_pels_sz = ceil_div( u32_to_i32(get_xy_dims(no->dims)) - (bck_in_off + u32_to_i32(stride)), stride ); 
	  bck_pels_sz += i32_pt_t(1,1); // include starting pixel
	  assert_st( bck_pels_sz.both_dims_gt( i32_pt_t() ) );

	  conv_ref_dims["bck_in_pad"] = dims_t( vect_uint32_t{ bck_in_pad.d[1], bck_in_pad.d[0] }, vect_string{"y","x"}, 1 );
	  conv_ref_dims["bck_pad_in_off"] = dims_t( vect_uint32_t{ bck_pad_in_off.d[1], bck_pad_in_off.d[0] }, vect_string{"y","x"}, 1 );

	  p_conv_node_t ogl = cp->must_get_node(get_arg("out_grad_loss"));
	  p_conv_node_t fgl = cp->must_get_node(get_arg("filts_grad_loss"));

	  gbt_tile_t gbt;
	  conv_ref_dims["oix"] = dims_t(  vect_uint32_t{ no->dims.dsz("chan"), stride.d[1], stride.d[0] }, 
					  vect_string{ "in_chan", "sy", "sx" }, 1 );
	  conv_ref_dims["pix"] = dims_t(  vect_uint32_t{ no->dims.dsz("img"), 
		uint32_t(bck_pels_sz.d[1]), uint32_t(bck_pels_sz.d[0]) }, vect_string{ "img", "y", "x" }, 1 );
	  gbt.init( t_tile_sz, 128, u32_pt_t( conv_ref_dims["pix"].dims_prod(), conv_ref_dims["oix"].dims_prod()));
	  dims_t work;
	  work.add_dims( "pels_blk", gbt.num_blk.d[0] );
	  work.add_dims( "out_ix_blk", gbt.num_blk.d[1] );
	  work.add_dims( "pels_tile", gbt.thr_per_blk.d[0] );
	  work.add_dims( "out_ix_tile", gbt.thr_per_blk.d[1] );
	  work.add_dims( "pels", gbt.mn_per_thr.d[0], "out_ix", gbt.mn_per_thr.d[1] );
	  work.calc_strides();
	  conv_ref_dims["work"] = work;
	  conv_ref_dims["fioc"] = dims_t( vect_uint32_t{ ogl->dims.dsz("chan"), u32_ceil_div(kern_sz.d[1],stride.d[1]), 
		u32_ceil_div(kern_sz.d[0],stride.d[0]) }, vect_string{"out_chan","ky","kx"}, 1 );
	  
	  gbt_tile_t gbt_fb;
	  gbt_fb.init( t_tile_sz, 128, u32_pt_t( fgl->dims.dsz("in_chan")*fgl->dims.dsz("y")*fgl->dims.dsz("x"), 
						 fgl->dims.dsz("out_chan") ) );
	  dims_t work_fb;
	  work_fb.add_dims( "pels_blk", gbt_fb.num_blk.d[0] );
	  work_fb.add_dims( "out_ix_blk", gbt_fb.num_blk.d[1] );
	  work_fb.add_dims( "pels_tile", gbt_fb.thr_per_blk.d[0] );
	  work_fb.add_dims( "out_ix_tile", gbt_fb.thr_per_blk.d[1] );
	  work_fb.add_dims( "pels", gbt_fb.mn_per_thr.d[0], "out_ix", gbt_fb.mn_per_thr.d[1] );
	  work_fb.calc_strides();
	  conv_ref_dims["work_fb"] = work_fb;
	  conv_ref_dims["fioc_fb"] = dims_t( vect_uint32_t{ ogl->dims.dsz("img"), ogl->dims.dsz("y"), ogl->dims.dsz("x") },
					     vect_string{"img","y","x"}, 1 );

	}
	if( is_conv ) {
	  // calc_blocking_conv()
	  uint32_t const out_ix_sz = no->dims.dims_prod();
	  uint32_t const pels_sz = out_ix_sz / no->dims.dsz("chan");
	  assert_st( pels_sz * no->dims.dsz("chan") == out_ix_sz ); // by construction
	  gbt_tile_t gbt;
	  gbt.init( t_tile_sz, 128, u32_pt_t( pels_sz, no->dims.dsz("chan") ) );
	  dims_t work;
	  uint32_t const lines_sz = no->dims.dsz("img") * no_sz.d[1];
	  if( cts == tconv_str ) {
	    assert( gbt.thr_per_blk.d[0] >= 2 ); // if 1, would imply tconv_blk_max_imgs = 1 (but not sensible?)
	    work.add_dims( "blk_bline", u32_ceil_div( lines_sz, gbt.thr_per_blk.d[0] ), 
			   "blk_bx", u32_ceil_div( no_sz.d[0], gbt.mn_per_thr.d[0] ) );
	    uint32_t tconv_blk_max_imgs = 0;
	    uint32_t blk_b_line = 0;
	    for( uint32_t i = 0; i != work.dsz("blk_bline"); ++i ) {
	      uint32_t const blk_e_line = blk_b_line + gbt.thr_per_blk.d[0] - 1;
	      uint32_t const blk_b_img = blk_b_line / no_sz.d[1];
	      uint32_t const blk_e_img = std::min( no->dims.dsz("img") - 1, blk_e_line / no_sz.d[1] );
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
	    uint32_t const tconv_blk_max_in_lines = (gbt.thr_per_blk.d[0] - tconv_blk_max_imgs)*stride.d[1] + kern_sz.d[1]*tconv_blk_max_imgs;
	    uint32_t const tconv_blk_x_sz = (gbt.mn_per_thr.d[0] - 1)*stride.d[0] + kern_sz.d[0];
	    // the tconv/in_tile_xpose format is for use when both ni_sz.d[0/1] are small multiple of
	    // gbt.mn_per_thr.d[0]/gbt.thr_per_blk.d[0] or >> than them (to avoid wasting too much work). each block will handle a
	    // (x,y) window of the output of size (gbt.mn_per_thr.d[0],gbt.thr_per_blk.d[0]) across bix_pels_blk_sz*gbt.mn_per_thr.d[0]
	    // output chans. in this case, we do not unroll across input chans, but we do unroll across kern_sz in X
	    // (and maybe in Y too for small kernels).  note: "out_ix" from in_tile_xpose becomes "in_ix" for tconv;
	    // from the perspective inside tconv: the blk_y and blk_x dims are in input image space, the other dims are
	    // in output space image space. other x/y's (in thread and block indexes) are all in output image space.
	    in_dims = dims_t( vect_uint32_t{
		work.dsz("blk_bline"), work.dsz("blk_bx"), ni->dims.dsz("chan"), tconv_blk_max_in_lines, tconv_blk_x_sz },
	      vect_string{"blk_bline","blk_bx","blk_in_chan","blk_y","blk_x"}, 1 );
	  } else {
	    work.add_dims( "pels_blk", gbt.num_blk.d[0] );
	  }
	  work.add_dims( "out_chan_blk", gbt.num_blk.d[1] );

	  // dims of per-group work (defines # threads per local group)
	  if( cts == tconv_str ) { work.add_dims( "blk_y", gbt.thr_per_blk.d[0] ); }
	  else { work.add_dims( "pels_tile", gbt.thr_per_blk.d[0] ); }
	  work.add_dims(   "out_chan_tile", gbt.thr_per_blk.d[1] );

	  work.add_dims( "pels", gbt.mn_per_thr.d[0], "out_chan", gbt.mn_per_thr.d[1] ); // dims of per-thread work
	  if( cts == ipconv_str ) { 
	    uint32_t fioc_tile = 4;
	    while( (fioc_tile < 32) && (fioc_tile*2*gbt.thr_per_blk.dims_prod()) <= 512 ) { fioc_tile *= 2; }
	    assert_st( (ni->dims.dsz("chan") % fioc_tile) == 0 );
	    work.add_dims( "fioc_tile", fioc_tile ); 
	  } // unrolling/tiling of inner loop
	  work.calc_strides();

	  if( cts == k1conv_str ) { 
	    uint32_t const in_blk_iter_chan_dim = 8; // FIXME: make into param?
	    // the k1conv/xpose_in format is for use when stride=1, kernel_sz=1, and in_pad=0. we treat all input pixels as one 1D
	    // vector across img:y:x, and divide them into blocks. we also block in the chan dim for unrolling.
	    in_dims = dims_t( vect_uint32_t{
		work.dsz("pels_blk"), u32_ceil_div(ni->dims.dsz("chan"),in_blk_iter_chan_dim), in_blk_iter_chan_dim, work.dsz("pels_tile")*work.dsz("pels")}, 
	      vect_string{"blk","blk_iter","blk_iter_chan","blk_pel"}, 1 ); 
	  }
	  conv_ref_dims["work"] = work;
	  conv_ref_dims["out_ref"] = no->dims; // k1conv and in_tile_xpose need the standard output dims for reference
	  conv_ref_dims["in_xp"] = in_dims; // cached final desired format for input (original 'standard' format is stored as "in_ref" earlier)
	  // 'standard' and desired/xformed filter dims. we don't currently xform the biases (although maybe we should).
	  conv_ref_dims["filts_xp"] = dims_t( vect_uint32_t{ work.dsz("out_chan_blk"),ni->dims.dsz("chan"), kern_sz.d[1], kern_sz.d[0],
		work.dsz("out_chan"),work.dsz("out_chan_tile")}, vect_string{"out_chan_blk","in_chan","y","x","out_chan_reg","out_chan_tile"}, 1 );
	  // dims_t( vect_uint32_t{cop->out_chans}, vect_string{"out_chan"}, 1 );
	} // end if(is_conv)
      }
    }
  };


  struct conv_pipe_fwd_t : virtual public nesi, public has_conv_fwd_t // NESI(help="compute conv pipe forward using rtc",
			   // bases=["has_conv_fwd_t"], type_id="rtc" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    uint32_t enable_lineinfo; //NESI(default=0,help="if 1, enable lineinfo for ptx compilation")
    uint32_t enable_stats; //NESI(default=0,help="if 1, dump stats")
    uint32_t enable_prof; //NESI(default=1,help="if 1, enable profiling")
    uint32_t enable_double_run; //NESI(default=0,help="if 1, run ops an extra time before the timed run (doubles run time, might improve timing quality/repeatability).")
    string per_call_fn; //NESI(default="",help="if non-empty, write per-call profiling (timing via events) to given file.")
    vect_string def; // NESI(help="#define STR 1 in generated code")
    vect_p_quantize_ops_t quantize; //NESI(help="per-layer quantize options")
    uint32_t show_compile_log; //NESI(default=0,help="if 1, print compilation log")
    uint32_t show_rtc_calls; //NESI(default=0,help="if 1, print rtc calls")
    uint32_t show_func_attrs; //NESI(default=0,help="if 1, print func attrs after load")
    uint32_t enable_k1conv; //NESI(default=0,help="if 1, enable experimental k1conv special case")
    uint32_t enable_ipconv; //NESI(default=0,help="if 1, enable ipconv special case")
    uint32_t enable_tconv; //NESI(default=0,help="if 1, enable tconv special case")
    uint32_t enable_bconv; //NESI(default=0,help="if 1, enable bconv")
    uint32_t force_enable_tconv; //NESI(default=0,help="if 1, force-enable experimental tconv special case even for not-sensible sizes")
    uint32_t enable_write_xpose; //NESI(default=0,help="if 1, enable experimental k1conv write xposing")
    uint32_t force_zero_bias; //NESI(default=0,help="if 1, force biases to zero")
    uint32_t flags; //NESI(default=0,help="dynamic flags to pass to kernels that request them (often to trick compiler)")
    uint32_t t_tile_sz; //NESI(default=8,help="register blocking tile size: compute t_tile_sz^2 outputs in registers per thread")
    vect_string dump_vars; // NESI(help="dump out values of these vars after forward")
    uint32_t enable_bwai_test; //NESI(default=0,help="if 1, generate an call to bwai")

    p_conv_pipe_t cp;
    p_map_str_p_op_info_t op_infos;

    vect_string op_param_names;
    set_string filts_names;
    set_string inxp_names;
    set_string force_zero_names;

    vect_string stats_names;
    map_str_float_t stats_map;

    p_rtc_compute_t rtc; //NESI(default="(be=nvrtc)",help="rtc back-end to use")

    vect_rcg_func_call_t init_calls;
    vect_rcg_func_call_t fwd_calls;
    

    virtual void init( p_conv_pipe_t const & cp_ );
    virtual void run_fwd( vect_string const & to_set_vns, p_map_str_p_nda_float_t const & fwd, vect_string const & to_get_vns );
    vect_uint32_t dropout_cixs;
    virtual void set_det_drop_seed( uint32_t const & det_drop_seed_ ) { 
      // sigh.
      for( vect_uint32_t::const_iterator i = dropout_cixs.begin(); i != dropout_cixs.end(); ++i ) {
	assert( (*i) < fwd_calls.size() );
	rcg_func_call_t & rcg = fwd_calls[*i];
	assert_st( rcg.u32_args.size() == 1 );
	rcg.u32_args[0] = det_drop_seed_; 
      }
    }

    void update_stats( void );
    string dump_var( string const & n );
    virtual string get_info_log( void );
  protected:
    vect_string gen_op_stats( string const & top_in );
    void gen_op_quantize( string const & top_in, uint32_t const & max_val, uint32_t const & keep_bits );

    rtc_codegen_t codegen;

    string gen_func( rtc_func_sig_t const & rfs );
    void gen_call( string const & fn, p_op_info_t const & oi );
    string gen_apply_func_to_var( string const & in_an, string const & in_var, 
				  string const & ret_an, dims_t const & ret_dims, string const & func );
    void gen_node_var( string const & name, string const & node_name );
    void gen_op( p_conv_op_t const & cop );
    void gen_ops_rec( string const & node_name );

    void run_rfc( rcg_func_call_t & rfc );
  };

  // FIXME: i'm not too happy about the duplication between here and the kernel version
  float stats_reduce( string const & stats_name, float const & v1, float const & v2 ) { 
    if( 0 ) { }
    else if( endswith(stats_name,"min_out_sz_1") ) { return std::min(v1,v2); }
    else if( endswith(stats_name,"max_out_sz_1") ) { return std::max(v1,v2); }
    else if( endswith(stats_name,"sum_out_sz_1") ) { return v1 + v2; }
    else if( endswith(stats_name,"hist_out_sz_1") ) { return v1 + v2; }
    else if( endswith(stats_name,"cnt_out_sz_1") ) { return v1 + v2; }
    else { assert_st(0); }
  }

  void conv_pipe_fwd_t::update_stats( void ) {
    for( vect_string::const_iterator i = stats_names.begin(); i != stats_names.end(); ++i ) {
      p_nda_float_t nda = rtc->copy_var_as_flat_nda( *i );
      assert_st( nda->elems.sz == 1 );
      float v = nda->elems[0];
      if( has( stats_map, *i ) ) { v = stats_reduce( *i, v, stats_map[*i] ); }
      stats_map[*i] = v;
    }
  }

  string conv_pipe_fwd_t::dump_var( string const & n ) {
    string ret;
    p_nda_float_t nda = rtc->create_nda_from_var( n );
    // dump nda
    ret += strprintf( "dumping var '%s'\n", str(n).c_str() );
    for( dims_iter_t di( nda->dims ) ; ; )  {
      ret += strprintf( "[%s]: %s\n", nda->dims.ix_str(di.di,1).c_str(), str(nda->at(di.di)).c_str() );
      if( !di.next() ) { break; }
    }
    return ret;
  }

  string conv_pipe_fwd_t::get_info_log( void ) {
    string ret;
    for( vect_string::const_iterator i = dump_vars.begin(); i != dump_vars.end(); ++i ) { 
      ret += dump_var( *i ); 
    }
    for( map_str_float_t::const_iterator i = stats_map.begin(); i != stats_map.end(); ++i ) {
      ret += strprintf( "%s=%s\n", str(i->first).c_str(), str(i->second).c_str() );
    }
    return ret;
  }

  vect_string conv_pipe_fwd_t::gen_op_stats( string const & top_in ) {
    vect_string const reds{ "min","max","sum","hist","cnt" }; // FIXME: dup'd with kernel code
    uint32_t in_sz = rtc->get_var_dims_floats( top_in ).dims_prod(); // treat input as flat
    uint32_t primary_in = 1;
    assert_st( in_sz );
    dims_t arg_dims( {0}, {"v"}, 1 ); // all vars are single-dim with wild/any size
    map_str_dims_t ref_dims; // note:constant after initial setup
    vect_string cur_ins;
    for( uint32_t i = 0; i != reds.size(); ++i ) {  // input dims (const); initial inputs
      must_insert( ref_dims, reds[i] + "_in", arg_dims ); 
      must_insert( ref_dims, reds[i] + "_out", arg_dims ); 
      cur_ins.push_back( top_in ); 
    } 
    while( in_sz > 1 ) {
      string const func = gen_func( rtc_func_sig_t{ "var_stats", ref_dims } );
      vect_string cur_outs;
      //vect_string args = cur_ins;
      vect_string out_args;
      uint32_t const out_sz = u32_ceil_div( in_sz, must_find(codegen.rtc_func_names_map,func)->tpb );
      for( uint32_t i = 0; i != reds.size(); ++i ) { 
	string cur_out = top_in + "_" + reds[i] + "_out_sz_" + str(out_sz);
	rtc->create_var_with_dims_floats( cur_out, dims_t{ {out_sz}, {"v"}, 1 } );
	cur_outs.push_back( cur_out );
	//args.push_back( cur_out );
      }
      map_str_str arg_map;
      assert_st( cur_ins.size() == reds.size() );
      for( uint32_t i = 0; i != reds.size(); ++i ) { must_insert( arg_map, reds[i]+"_in", cur_ins[i] ); }
      assert_st( cur_outs.size() == reds.size() );
      for( uint32_t i = 0; i != reds.size(); ++i ) { must_insert( arg_map, reds[i]+"_out", cur_outs[i] ); }
      fwd_calls.push_back( rcg_func_call_t{ func, "var_stats", arg_map, {in_sz, primary_in} } );
      cur_ins = cur_outs;
      in_sz = out_sz;
      primary_in = 0;
    }
    assert_st( in_sz == 1 );
    return cur_ins;
  }

  void conv_pipe_fwd_t::gen_op_quantize( string const & top_in, uint32_t const & max_val, uint32_t const & keep_bits ) {
    uint32_t drop_bits = 0;
    while( max_val > (1U<<(keep_bits+drop_bits)) ) { ++drop_bits; }
    uint32_t drop_mask = ((1<<drop_bits)-1);
    string const func = gen_func( rtc_func_sig_t{ "quantize", {{"out",rtc->get_var_dims_floats(top_in)}} } );
    fwd_calls.push_back( rcg_func_call_t{ func, "quantize", map_str_str{{"out",top_in}}, {max_val,drop_mask} } );
  }

  // this assumes that in_var is valid/calculated, and returns ret_var=func(in_var). it assumes that func is a stateless
  // unary operator (with two args: {in,out}), so that only one unique ret_var need only be generated per unique
  // in_var/func<ret_dims> pair. ret_var is named in_var+"__"+func
  string conv_pipe_fwd_t::gen_apply_func_to_var( string const & in_an, string const & in_var, 
						 string const & ret_an, dims_t const & ret_dims, string const & func )
  {
    string const ret_var = in_var + "__" + func;
    bool const did_ins = inxp_names.insert( ret_var ).second;
    if( did_ins ) { // newly-seen/used ret_var, so create and calc it here
      rtc->create_var_with_dims_floats( ret_var, ret_dims );
      fwd_calls.push_back( rcg_func_call_t{ func, in_var + "__inxp", map_str_str{{in_an,in_var},{ret_an,ret_var}}} );
    }
    return ret_var;
  }

  void conv_pipe_fwd_t::gen_op( p_conv_op_t const & cop ) {
    p_op_info_t const & oi = must_find( *op_infos, cop->tag );
    if( cop->is( Concat_coi ) ) {      
      uint32_t chans_out_done = 0;
      for( uint32_t bi = 0; bi != cop->bots.size(); ++bi ) {
	dims_t & dims_in = cp->must_get_node( cop->bots[bi] )->dims;
	assert_st( get_xy_dims( dims_in ) == get_xy_dims( oi->get_arg_dims("out") ) );
	assert_st( chans_out_done+dims_in.dsz("chan") <= oi->get_arg_dims("out").dsz("chan") );
	// note: oi->template_var_values is overwritten each iter; also, oi->cop->tag+"__copy" is reused for all calls (FIXME either/both?)
        oi->template_var_values = { {"ocix",str(chans_out_done)} };
	oi->set_arg( rtc, "in", cop->bots[bi] );
	gen_call( "copy", oi );
	chans_out_done += dims_in.dsz("chan");
	oi->erase_arg( "in" );
      }
      assert_st( chans_out_done == oi->get_arg_dims("out").dsz("chan") );
    } else if( cop->is( Split_coi ) ) { // FIXME: pretty dup'd with Concat above ... generalize/merge/share?
      uint32_t chans_in_done = 0;
      for( uint32_t ti = 0; ti != cop->tops.size(); ++ti ) {
	dims_t & dims_out = cp->must_get_node( cop->tops[ti] )->dims;
	assert_st( get_xy_dims( dims_out ) == get_xy_dims( oi->get_arg_dims("in") ) );
	assert_st( chans_in_done+dims_out.dsz("chan") <= oi->get_arg_dims("in").dsz("chan") );
	// note: oi->template_var_values is overwritten each iter; also, oi->cop->tag+"__copy" is reused for all calls (FIXME either/both?)
        oi->template_var_values = { {"icix",str(chans_in_done)} };
	oi->set_arg( rtc, "out", cop->tops[ti] );
	gen_call( "split_copy", oi );
	chans_in_done += dims_out.dsz("chan");
	oi->erase_arg( "out" );
      }
      assert_st( chans_in_done == oi->get_arg_dims("in").dsz("chan") );
    } else if( cop->is( Reduce_coi ) ) {
      gen_call( "reduce", oi );
    } else if( cop->is( Pooling_coi ) ) {
      if( cop->u32_param("emit_out_in_yx") == 1 ) {
	string const out_in_yx = oi->get_arg("out") + "_in_yx"; 
	rtc->create_var_with_dims_floats( out_in_yx, oi->get_arg_dims("out") ); // same size as out
	oi->set_arg( rtc, "out_in_yx", out_in_yx );
      } else {
	assert_st( cop->u32_param("emit_out_in_yx") == 0 );
	oi->set_null_arg( "out_in_yx" );
      }
      gen_call( "pool", oi );
    } else if( cop->is( Convolution_coi ) ) {
      op_param_names.push_back( oi->get_arg("filts") );
      op_param_names.push_back( oi->get_arg("biases") );
      if( force_zero_bias ) { force_zero_names.insert( oi->get_arg("biases") ); }
      string const in_id = oi->arg_map["in"];
      oi->reset_arg( rtc, "in", oi->get_arg("in") ); // reset in dims from rtc, since may now differ from conv_pipe dims
      if( oi->cts != ipconv_str ) { // ipconv uses untransformed filts
	string const filts_xp_fn = gen_func( rtc_func_sig_t{ "xpose_filts", oi->conv_ref_dims, oi->template_var_values } );
	oi->reset_arg( rtc, "filts", gen_apply_func_to_var( "filts", oi->get_arg("filts"), 
							    "filts_xp", oi->get_arg_dims("filts_xp"), filts_xp_fn ) );
      }
      //in_arg_ids.push_back( cop->bots[2] ); // biases
      if( oi->cts == tconv_str ) {
	string const xp_fn = gen_func( rtc_func_sig_t{ "in_tile_xpose", oi->conv_ref_dims, oi->template_var_values } );
	oi->reset_arg( rtc, "in", gen_apply_func_to_var( "in", oi->get_arg("in"),
							 "in_xp", oi->get_arg_dims("in_xp"), xp_fn ) );
      } else if( oi->cts == k1conv_str ) {
	if( oi->get_arg_dims("in") != oi->get_arg_dims("in_xp") ) { 
	  // if dims not exactly right, assume they are 'normal' dims and convert. FIXME: fails if unexpected format.
	  string const xp_fn = gen_func( rtc_func_sig_t{ "xpose_in", oi->conv_ref_dims, oi->template_var_values } );
	  oi->reset_arg( rtc, "in", gen_apply_func_to_var( "in", oi->get_arg("in"),
							   "in_xp", oi->get_arg_dims("in_xp"), xp_fn ) );
	} 	
      } 
      dims_t no_dims = oi->conv_ref_dims["out_ref"];
      if( oi->cts == k1conv_str ) { 
	p_op_info_t noi;
	p_conv_node_t no = cp->must_get_node( oi->get_arg("out") );
	if( no->in_place_ops.empty() && ( no->bot_for.size() == 1) ) { // if output feeds single non-in-place operation
	  noi = must_find( *op_infos, no->bot_for[0] ); // next operation
	  if( enable_write_xpose && (noi->cts == k1conv_str) ) { no_dims = noi->get_arg_dims("in"); }
	}
      }
      rtc->create_var_with_dims_floats( oi->get_arg("out"), no_dims );
      oi->reset_arg( rtc, "out", oi->get_arg("out") );
      gen_call( oi->cts, oi );
    } else if( cop->is( ReLU_coi ) ) {
      assert_st( oi->get_arg("in") == oi->get_arg("out") ); // check that this is a single in-out in-place operation
      oi->set_arg( rtc, "inout", oi->get_arg("in") );
      gen_call( "relu", oi );
    } else if( cop->is( LRN_coi ) ) {
      assert_st( oi->get_arg_dims("in") == oi->get_arg_dims("out") ); // FIXME: better place/way for this check?
      if( cop->u32_param("emit_out_scale_base") == 1 ) {
	string const out_scale_base = oi->get_arg("out") + "_scale_base"; 
	rtc->create_var_with_dims_floats( out_scale_base, oi->get_arg_dims("out") ); // same size as out
	oi->set_arg( rtc, "out_scale_base", out_scale_base );
      } else {
	assert_st( cop->u32_param("emit_out_scale_base") == 0 );
	oi->set_null_arg( "out_scale_base" );
      }
      gen_call( "lrn", oi );
    } else if( cop->is( BckLRN_coi ) ) {
      oi->set_arg( rtc, "out_scale_base", oi->get_arg("out") + "_scale_base" ); // generated by matching LRN op
      gen_call( "bck_lrn", oi );
    } else if( cop->is( Dropout_coi ) ) {
      assert_st( oi->get_arg("in") == oi->get_arg("out") ); // check that this is a single in-out in-place operation
      oi->set_arg( rtc, "inout", oi->get_arg("in") );
      gen_call( "dropout", oi );
      // FIXME: move this check (and others like it) to conv_util.cc or similar?
      double const dropout_ratio = lc_str_d( oi->template_var_values["dropout_ratio"] );
      assert_st( dropout_ratio > 0.0 );
      assert_st( dropout_ratio < 1.0 );
      fwd_calls.back().u32_args.push_back( 0 ); // see update code elsewhere. yeah, not the cleanest approach.
      dropout_cixs.push_back( fwd_calls.size() - 1 );
    } else if( cop->is( BckDropout_coi ) ) {
      assert_st( oi->get_arg("in") == oi->get_arg("out") ); // check that this is a single in-out in-place operation
      oi->set_arg( rtc, "inout", oi->get_arg("in") );
      gen_call( "dropout", oi ); // Backwards of dropout is dropout
      fwd_calls.back().u32_args.push_back( 0 ); // see update code elsewhere. yeah, not the cleanest approach.
      dropout_cixs.push_back( fwd_calls.size() - 1 );
    } else if( cop->is( Softmax_coi ) ) {
      gen_call( "softmax", oi );
    } else if( cop->is( SoftmaxWithLoss_coi ) ) {
      string const prob_node_name = cop->tag + "_prob";
      gen_node_var( prob_node_name, oi->get_arg("in") );
      oi->set_arg( rtc, "prob", prob_node_name );
      string const loss_per_pel = oi->get_arg("loss") + "_per_pel"; // same size as label
      gen_node_var( loss_per_pel, oi->get_arg("label") );
      oi->set_arg( rtc, "loss_per_pel", loss_per_pel );
      gen_call( "softmax", oi );
      gen_call( "sm_grad_and_loss", oi  );
      gen_call( "sum_loss_over_imgs", oi );
    } else if( cop->is( Spreading_coi ) ) {
      oi->set_arg( rtc, "out_in_yx", oi->get_arg("out") + "_in_yx" ); // generated by matching Pooling op
      gen_call( "spreading", oi );
    } else if( cop->is( ZeroIfNonPos_coi ) ) {
      gen_call( cop->type, oi );
    } else if( cop->is( BckConv_coi ) ) { 
      // { in, filts, biases, out_grad_loss } --> { in_grad_loss, filts_grad_loss, biases_grad_loss }
      string ogl_vn = oi->get_arg("out_grad_loss");
      string ogl_fn = "BckConv_in_grad_loss";
      string fgl_fn = "BckConv_filts_grad_loss";
      assert_st( oi->cts == conv_str );
      if( enable_bconv ) {
#if 0
	dims_t const & ogl_dims = rtc->get_var_dims_floats( ogl_vn );
	dims_t const & ogl_xp_dims = ogl_dims; // oi->conv_ref_dims["out_grad_loss"];
	string ogl_xp_fn = gen_func( rtc_func_sig_t{ "btconv_ogl_xpose", {ogl_dims,ogl_xp_dims}, 
	      oi->conv_ref_dims, oi->template_var_values } );
	ogl_vn = gen_apply_func_to_var( ogl_vn, ogl_xp_dims, ogl_xp_fn );
#endif
	ogl_fn = "bconv";
	fgl_fn = "bconv_fb";
      }
      gen_call( ogl_fn, oi );
      gen_call( "BckConv_biases_grad_loss", oi );
      gen_call( fgl_fn, oi );
    } else { rt_err( "gen_op: unhandled op of type: " + cop->type ); }
  }

  struct cnn_custom_codegen_t : public custom_codegen_t {

    virtual void gen_op( rtc_call_gen_t * rcg, string const & op_name ) {
      // *** custom codegen hooks ***
      if( op_name == "conv" ) { gen_op_conv(rcg); } 
      else if( op_name == "ipconv" ) { gen_op_ipconv(rcg); } 
      else if( op_name == "k1conv" ) { gen_op_k1conv(rcg); } 
      else if( op_name == "tconv" ) { gen_op_tconv(rcg); } 
      else if( op_name == "bwai" ) { gen_op_bwai(rcg); } 
      else if( op_name == "bconv" ) { gen_op_bconv(rcg); } 
      else if( op_name == "bconv_fb" ) { gen_op_bconv_fb(rcg); } 
      else if( op_name == "reduce" ) { gen_op_reduce(rcg); } 
    }

    void gen_op_reduce( rtc_call_gen_t * rcg ) {
      for( uint32_t i = 0; i != rcg->flat_arg_decls.size() - 1; ++i ) { 
	rcg->line( "ins_ops", "v += ins_"+str(i)+"[GLOB_ID_1D];" ); 
      }
    }

    string add_bias_then_maybe_relu( rtc_call_gen_t * rcg, dims_t const & work, uint32_t const & tx, uint32_t const ty ) { 
      string const ve = strprintf( "(out_tile[%s] + filts_strip[%s])", str((ty*work.dsz("out_chan")+tx)).c_str(), str(tx).c_str() );
      return rcg->rfs.get_u32_tvv("conv_has_relu") ? ( "max(0.0f,"+ve+")" ) : ve;
    }    

    void gen_op_bconv( rtc_call_gen_t * rcg ) {
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      uint32_t const in_smem_sz = work.dsz("pels_tile")*work.dsz("pels");
      rcg->set( "in_smem_sz", str(in_smem_sz) );
      uint32_t const in_smem_load_iter = u32_ceil_div( in_smem_sz, rcg->tpb );
      rcg->set( "in_smem_load_iter", str(in_smem_load_iter) );    

      uint32_t const filts_smem_sz = work.dsz("out_ix_tile")*work.dsz("out_ix");
      rcg->set( "filts_smem_sz", str(filts_smem_sz) );
      uint32_t const filts_smem_load_iter = u32_ceil_div( filts_smem_sz, rcg->tpb );
      rcg->set( "filts_smem_load_iter", str(filts_smem_load_iter) );    

      for( uint32_t tx = 0; tx != work.dsz( "out_ix" ); ++tx ) {
	rcg->line( "loads", strprintf( "filts_strip[%s] = filts_smem[%%(LOC_ID_1D_out_ix_tile)*%%(work_out_ix_dim)+%s];",
					 str(tx).c_str(), str(tx).c_str() ) );
      }
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) { // note: could merge with above loop, but we want to use ty for consistency
	rcg->line( "loads", strprintf( "in_strip[%s] = in_smem[%%(LOC_ID_1D_pels_tile)*%%(work_pels_dim)+%s];",
					 str(ty).c_str(), str(ty).c_str() ) );
      }

      rcg->line( "outs_to_filts_strip", "switch(work_pel) { " );
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) {
	rcg->line( "outs_to_filts_strip", "case "+str(ty)+":" );
	for( uint32_t tx = 0; tx != work.dsz( "out_ix" ); ++tx ) {
	  uint32_t const rix = ty*work.dsz("out_ix")+tx;
	  rcg->line( "fmas", strprintf( "out_tile[%s] += filts_strip[%s]*in_strip[%s];", 
					  str(rix).c_str(), str(tx).c_str(), str(ty).c_str() ) );
	  rcg->line( "outs_to_filts_strip", strprintf( "filts_strip[%s] = out_tile[%s];", 
					    str(tx).c_str(), str(rix).c_str() ) );	  
	}
	rcg->line( "outs_to_filts_strip", "break;" );
      }
      rcg->line( "outs_to_filts_strip", "} " );

      string store_expr = R"foo(
  igl_y = (%(pel_ix_y)-%(bck_in_pad_y_dim))*%(stride_y_dim)+%(out_ix_sy)-%(in_pad_y_dim)+%(bck_pad_in_off_y_dim);
  igl_x = (%(pel_ix_x)-%(bck_in_pad_x_dim))*%(stride_x_dim)+%(out_ix_sx)-%(in_pad_x_dim)+%(bck_pad_in_off_x_dim);
  if( igl_x >= 0 && igl_y >= 0 && igl_y < %(in_grad_loss_y_dim) && igl_x < %(in_grad_loss_x_dim) &&
      %(out_ix_in_chan) < %(in_grad_loss_chan_dim) && %(pel_ix_img) < %(in_grad_loss_img_dim) ) {
    in_grad_loss[ %(pel_ix_img)*%(in_grad_loss_img_sz) + %(out_ix_in_chan)*%(in_grad_loss_chan_sz) + 
		  igl_y*%(in_grad_loss_y_sz) + igl_x*%(in_grad_loss_x_sz)] = filts_strip[)foo";
      for( uint32_t tx = 0; tx != work.dsz( "out_ix" ); ++tx ) {
	rcg->line( "stores", store_expr + strprintf( "%s];\n};", str(tx).c_str() ) );
	rcg->line( "stores", "++out_ix;" );
      }
    }

    void gen_op_bconv_fb( rtc_call_gen_t * rcg ) {
      dims_t const & work = rcg->get_arg_dims_by_name( "work_fb" );
      uint32_t const in_smem_sz = work.dsz("pels_tile")*work.dsz("pels");
      rcg->set( "in_smem_sz", str(in_smem_sz) );
      uint32_t const in_smem_load_iter = u32_ceil_div( in_smem_sz, rcg->tpb );
      rcg->set( "in_smem_load_iter", str(in_smem_load_iter) );    

      uint32_t const filts_smem_sz = work.dsz("out_ix_tile")*work.dsz("out_ix");
      rcg->set( "filts_smem_sz", str(filts_smem_sz) );
      uint32_t const filts_smem_load_iter = u32_ceil_div( filts_smem_sz, rcg->tpb );
      rcg->set( "filts_smem_load_iter", str(filts_smem_load_iter) );    

      for( uint32_t tx = 0; tx != work.dsz( "out_ix" ); ++tx ) {
	rcg->line( "loads", strprintf( "filts_strip[%s] = filts_smem[%%(LOC_ID_1D_out_ix_tile)*%%(work_fb_out_ix_dim)+%s];",
					 str(tx).c_str(), str(tx).c_str() ) );
      }
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) { // note: could merge with above loop, but we want to use ty for consistency
	rcg->line( "loads", strprintf( "in_strip[%s] = in_smem[%%(LOC_ID_1D_pels_tile)*%%(work_fb_pels_dim)+%s];",
					 str(ty).c_str(), str(ty).c_str() ) );
      }

      rcg->line( "outs_to_filts_strip", "switch(work_pel) { " );
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) {
	rcg->line( "outs_to_filts_strip", "case "+str(ty)+":" );
	for( uint32_t tx = 0; tx != work.dsz( "out_ix" ); ++tx ) {
	  uint32_t const rix = ty*work.dsz("out_ix")+tx;
	  rcg->line( "fmas", strprintf( "out_tile[%s] += filts_strip[%s]*in_strip[%s];", 
					  str(rix).c_str(), str(tx).c_str(), str(ty).c_str() ) );
	  rcg->line( "outs_to_filts_strip", strprintf( "filts_strip[%s] = out_tile[%s];", 
					    str(tx).c_str(), str(rix).c_str() ) );	  
	}
	rcg->line( "outs_to_filts_strip", "break;" );
      }
      rcg->line( "outs_to_filts_strip", "} " );

      string store_expr = R"foo(
  if( %(pel_ix_in_chan) < %(filts_grad_loss_in_chan_dim) && %(out_ix_out_chan) < %(filts_grad_loss_out_chan_dim) ) {
    filts_grad_loss[ %(out_ix_out_chan)*%(filts_grad_loss_out_chan_sz) + %(pel_ix_in_chan)*%(filts_grad_loss_in_chan_sz) + 
		  %(pel_ix_y)*%(filts_grad_loss_y_sz) + %(pel_ix_x)*%(filts_grad_loss_x_sz)] = filts_strip[)foo";
      for( uint32_t tx = 0; tx != work.dsz( "out_ix" ); ++tx ) {
	rcg->line( "stores", store_expr + strprintf( "%s];\n};", str(tx).c_str() ) );
	rcg->line( "stores", "++out_ix;" );
      }
    }
    
    void gen_filts_smem_loads( rtc_call_gen_t * rcg, uint32_t const filts_smem_sz ) { // note: filts_smem_sz must == tvv %(filts_smem_sz)
      uint32_t const out_chan_smem_load_iter = u32_ceil_div( filts_smem_sz, rcg->tpb );    
      for( uint32_t i = 0; i != out_chan_smem_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string eif;
	if( (i+1)*rcg->tpb > filts_smem_sz ) { 
	  rcg->line( "filts_smem_loads", "if( "+ixe+" < %(filts_smem_sz) ) {" );eif = "}";}
	// note: load is (always) contiguous
	rcg->line( "filts_smem_loads", strprintf("filts_smem[%s] = filts[filts_off+(%%(tpb)*%s)];%s",ixe.c_str(),str(i).c_str(),eif.c_str()) );
      }
      // number of out chans per block; note: == work_out_chan_tile_dim*work_out_chan_dim
      uint32_t const filts_x_sz = rcg->get_arg_dims_by_name("filts").dstride("x"); 
      uint32_t const out_chan_bias_smem_load_iter = u32_ceil_div( filts_x_sz, rcg->tpb );
      rcg->set( "out_chan_bias_smem_load_iter", str(out_chan_bias_smem_load_iter) );

      rcg->line( "biases_smem_loads","int32_t ocix; int32_t const ocix_base = %(GRP_ID_1D_out_chan_blk)*%(filts_x_sz);" );
      for( uint32_t i = 0; i != out_chan_bias_smem_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string eif;
	rcg->line( "biases_smem_loads", strprintf( "ocix = ocix_base + (%s %%%% %%(work_out_chan_tile_dim))*%%(work_out_chan_dim) + ( %s / %%(work_out_chan_tile_dim) );", ixe.c_str(), ixe.c_str() ) );
	if( (i+1)*rcg->tpb > filts_x_sz ) { 
	  rcg->line( "biases_smem_loads", "if( "+ixe+" < %(filts_x_sz) ) {" );eif = "}";}
	// note: load is (always) contiguous
	rcg->line( "biases_smem_loads", strprintf("if( ocix < %%(biases_out_chan_dim) ) {filts_smem[%s] = biases[ocix];}%s",ixe.c_str(),eif.c_str()) );
      }

    }

    void gen_op_conv( rtc_call_gen_t * rcg ) {
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      dims_t const & filts = rcg->get_arg_dims_by_name( "filts" );
      uint32_t const filts_smem_sz = filts.dstride("x");
      rcg->set( "filts_smem_sz", str(filts_smem_sz) );
      gen_filts_smem_loads( rcg, filts_smem_sz );

      uint32_t const pel_smem_load_iter = u32_ceil_div( (work.dsz( "pels" ) * work.dsz( "pels_tile" )), rcg->tpb );
      rcg->set( "pel_smem_load_iter", str(pel_smem_load_iter) );
      rcg->set( "out_chan_tile", 
		"(%(LOC_ID_1D_out_chan_tile)+%(GRP_ID_1D_out_chan_blk)*%(work_out_chan_tile_dim))");
      rcg->set( "pel_tile",
		"(%(LOC_ID_1D_pels_tile)+%(GRP_ID_1D_pels_blk)*%(work_pels_tile_dim))");
      rcg->set( "out_chan_ix","(%(out_chan_tile)*%(work_out_chan_dim))" );
      for( uint32_t i = 0; i != work.dsz( "pels" ); ++i ) {
	insert_nda_ix_exprs( rcg->tf_exprs, "pel_ix_" + str(i), must_find(rcg->all_ix_dims,"out_pel_ix"),
			     strprintf( "(%%(pel_tile)*%%(work_pels_dim)+%s)", str(i).c_str() ) );
      }
      string const get_in = strprintf( 
	"float v = 0;\n"
	"      int const smem_in_ix_y = %%(out_pel_ix_y)*%%(stride_y_dim)+%%(filts_ix_out_chan_elem_y) - %%(in_pad_y_dim);\n"
	"      int const smem_in_ix_x = %%(out_pel_ix_x)*%%(stride_x_dim)+%%(filts_ix_out_chan_elem_x) - %%(in_pad_x_dim);\n"
	"      if(smem_in_ix_y >= 0 && smem_in_ix_x >= 0 && \n"
	"          %%(out_pel_ix_img) < %%(in_img_dim) && \n"
	"         smem_in_ix_x < %%(in_x_dim) && smem_in_ix_y < %%(in_y_dim) ) {\n"
	"        v = in[%%(out_pel_ix_img)*%%(in_img_sz) +\n"
	"          %%(filts_ix_out_chan_elem_in_chan)*%%(in_chan_sz) +\n"
	"          smem_in_ix_y*%%(in_y_sz) +\n"
	"          smem_in_ix_x*%%(in_x_sz)];\n" 
	"      }"
				       );
      rcg->set( "get_in", get_in );
      for( uint32_t tx = 0; tx != work.dsz( "out_chan" ); ++tx ) {
	rcg->line( "loads", strprintf( "filts_strip[%s] = filts_smem[%%(LOC_ID_1D_out_chan_tile)+%s*%%(work_out_chan_tile_dim)];",
				       str(tx).c_str(), str(tx).c_str() ) );
      }
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) { // note: could merge with above loop, but we want to use ty for consistency
	rcg->line( "loads", strprintf( "in_strip[%s] = in_smem[%%(LOC_ID_1D_pels_tile)*%%(work_pels_dim)+%s];",
					 str(ty).c_str(), str(ty).c_str() ) );
      }
      rcg->line( "stores", "int32_t tpix[%(work_pels_dim)];");
      rcg->line( "stores", "int32_t tcix[%(work_out_chan_dim)];");
      // FIXME: should somehow assert that both out_ix and pel_ix_N have the same dims here
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) { 
	rcg->line( "stores", 
		   strprintf( "tpix[%s] = %%(pel_ix_%s_img)*%%(out_img_sz) + "
			      "( %%(pel_ix_%s_x_nomod) %%%% (%%(out_y_dim)*%%(out_x_dim)) ); // cache out pel ixs ", // note: y:x adj-dim opt.
				str(ty).c_str(), str(ty).c_str(), str(ty).c_str() ) );
      }
      for( uint32_t ty = 0; ty != work.dsz( "out_chan" ); ++ty ) { 
	rcg->line( "stores", strprintf( "  tcix[%s] = (%%(out_chan_ix)+%s)*%%(out_chan_sz); // cache out chan ixs",
					  str(ty).c_str(), str(ty).c_str() ) );
      }	
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) {
	rcg->line( "stores", "if( %(pel_ix_"+str(ty)+"_x_nomod) >= %(pel_ix_0_dims_prod) ) { return; } "
		     "// this pel and the following are off-the-end pels, so don't store them." );
	for( uint32_t tx = 0; tx != work.dsz( "out_chan" ); ++tx ) {
	  rcg->line( "fmas", strprintf( "out_tile[%s] += filts_strip[%s]*in_strip[%s];", 
					  str((ty*work.dsz( "out_chan" )+tx)).c_str(), str(tx).c_str(), str(ty).c_str() ) );
	  rcg->line( "stores", strprintf( "if( tcix[%s] < (%%(out_chan_dim)*%%(out_chan_sz)) ) { out[ tpix[%s] + tcix[%s] ] = %s; }",
					    str(tx).c_str(), str(ty).c_str(), str(tx).c_str(), 
					  add_bias_then_maybe_relu(rcg,work,tx,ty).c_str() ) );
	}
      }
    }

    void gen_op_ipconv( rtc_call_gen_t * rcg ) {
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      //dims_t const & filts = get_arg_dims_by_name( "filts" );
      uint32_t const filts_smem_sz = work.dsz("out_chan_tile")*work.dsz("out_chan")*work.dsz("fioc_tile");
      rcg->set( "filts_smem_sz", str(filts_smem_sz) );
      uint32_t const out_chan_smem_load_iter = u32_ceil_div( filts_smem_sz, rcg->tpb );    
      for( uint32_t i = 0; i != out_chan_smem_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string const filt_ix = "( LOC_ID_1D/%(work_fioc_tile_dim) + %(tpb)/%(work_fioc_tile_dim)* "+str(i)+")";
	string eif;
	// FIXME: can load garbage when ((out_chan_dim % filts_per_blk) != 0). pad output? add conditionals here? ignore?
	if( (i+1)*rcg->tpb > filts_smem_sz ) { 
	  rcg->line( "filts_smem_loads", "if( "+ixe+" < %(filts_smem_sz) ) {" );eif = "}";}
	rcg->line( "filts_smem_loads", strprintf("filts_smem[%s] = filts[filts_off+(%s*%%(filts_out_chan_sz))];%s",ixe.c_str(),filt_ix.c_str(),eif.c_str()) );
      }

      uint32_t const in_smem_sz = work.dsz("pels_tile")*work.dsz("pels")*work.dsz("fioc_tile");
      rcg->set( "in_smem_sz", str(in_smem_sz) );
      uint32_t const in_smem_load_iter = u32_ceil_div( in_smem_sz, rcg->tpb );    
      // currently, ipconv can only handle one output point per image, and assume the filt and in data-layouts are the
      // same (hence the name ipconv, for inner-product-conv).
      for( uint32_t i = 0; i != in_smem_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string const img_ix = "( LOC_ID_1D/%(work_fioc_tile_dim) + %(tpb)/%(work_fioc_tile_dim)* "+str(i)+")";
	string eif;
	// FIXME: can load garbage when ((in_img_dim % imgs_per_blk) != 0). pad input? add conditionals here? ignore?
	if( (i+1)*rcg->tpb > in_smem_sz ) { 
	  rcg->line( "in_smem_loads", "if( "+ixe+" < %(in_smem_sz) ) {" );eif = "}";}
	rcg->line( "in_smem_loads", strprintf("in_smem[%s] = in[in_off+(%s*%%(in_img_sz))];%s",ixe.c_str(),img_ix.c_str(),eif.c_str()) );
      }

      for( uint32_t tx = 0; tx != work.dsz( "out_chan" ); ++tx ) {
	rcg->line( "loads", strprintf( "filts_strip[%s] = filts_smem_off[%s*%%(work_fioc_tile_dim)];",
					 str(tx).c_str(), str(tx).c_str() ) );
      }
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) { // note: could merge with above loop, but we want to use ty for consistency
	rcg->line( "loads", strprintf( "in_strip[%s] = in_smem_off[%s*%%(work_fioc_tile_dim)];",
					 str(ty).c_str(), str(ty).c_str() ) );
      }
      rcg->line( "outs_to_filts_strip", "if( (in_pel+work_pel) >= %(in_img_dim) ) { return; } "
		   "// this pel and the following are off-the-end pels, so don't store them." );
      rcg->line( "outs_to_filts_strip", "switch(work_pel) { " );
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) {
	rcg->line( "outs_to_filts_strip", "case "+str(ty)+":" );
	for( uint32_t tx = 0; tx != work.dsz( "out_chan" ); ++tx ) {
	  rcg->line( "fmas", strprintf( "out_tile[%s] += filts_strip[%s]*in_strip[%s];", 
					  str((ty*work.dsz( "out_chan" )+tx)).c_str(), str(tx).c_str(), str(ty).c_str() ) );
	  rcg->line( "outs_to_filts_strip", strprintf( "filts_strip[%s] = out_tile[%s];", 
					    str(tx).c_str(), str((ty*work.dsz("out_chan")+tx)).c_str() ) );	  
	}
	rcg->line( "outs_to_filts_strip", "break;" );
      }
      rcg->line( "outs_to_filts_strip", "} " );

      for( uint32_t tx = 0; tx != work.dsz( "out_chan" ); ++tx ) {
	string ve = strprintf( "(filts_strip[%s] + biases[ocix+%s])", str(tx).c_str(), str(tx).c_str() );
	ve = rcg->rfs.get_u32_tvv("conv_has_relu") ? ( "max(0.0f,"+ve+")" ) : ve;
	for( uint32_t wb = work.dsz("fioc_tile") / 2; wb; wb /= 2 ) {
	  rcg->line( "stores", strprintf( "filts_strip[%s] += __shfl_down( filts_strip[%s], %s, %s );", 
					    str(tx).c_str(), str(tx).c_str(), str(wb).c_str(), 
					    str( work.dsz("fioc_tile") ).c_str() ) );
	}
	rcg->line( "stores", strprintf( "if( (%%(LOC_ID_1D_fioc_tile) == 0 ) && ((ocix + %s) < %%(out_chan_dim)) ) "
					  "{ out[out_off + %s*%%(out_chan_sz)] = %s; }", 
					  str(tx).c_str(), str(tx).c_str(), str(ve).c_str() ) );
      }
    }

    void gen_op_bwai( rtc_call_gen_t * rcg ) {
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      uint32_t const a_sm_sz = work.dsz("Mb")*work.dsz("Mt")*work.dsz("Kb");
      rcg->set( "a_sm_sz", str(a_sm_sz) );
      uint32_t const a_sm_load_iter = u32_ceil_div( a_sm_sz, rcg->tpb );    
      for( uint32_t i = 0; i != a_sm_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string const filt_ix = "( LOC_ID_1D/%(work_Kb_dim) + %(tpb)/%(work_Kb_dim)* "+str(i)+")";
	string eif;
	if( (i+1)*rcg->tpb > a_sm_sz ) { 
	  rcg->line( "sm_loads", "if( "+ixe+" < %(a_sm_sz) ) {" );eif = "}";}
	rcg->line( "sm_loads", strprintf("a_sm[%s] = a[a_off+(%s*%%(a_M_sz))];%s",ixe.c_str(),filt_ix.c_str(),eif.c_str()) );
      }

      uint32_t const b_sm_sz = work.dsz("Mb")*work.dsz("Mt")*work.dsz("Kb");
      rcg->set( "b_sm_sz", str(b_sm_sz) );

    }

    void gen_op_k1conv( rtc_call_gen_t * rcg ) {
      assert_st( get_xy_dims( rcg->get_arg_dims_by_name( "in_pad" ) ).is_zeros() );
      assert_st( (get_xy_dims( rcg->get_arg_dims_by_name( "stride" ) ) == u32_pt_t{1,1}) );
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      dims_t const & filts = rcg->get_arg_dims_by_name( "filts" );
      assert_st( filts.dsz("x") == 1 ); assert_st( filts.dsz("y") == 1 );
      dims_t const & in = rcg->get_arg_dims_by_name( "in" );
      dims_t const & out = rcg->get_arg_dims_by_name( "out" );
      // calculate needed smem sizes (and total kernel needed smem size)
      // note: filts and in smem are used concurrently, then just all of all_smem as an output buffer
      uint32_t const filts_smem_sz = filts.dstride("in_chan")*in.dsz("blk_iter_chan");
      rcg->set( "filts_smem_sz", str(filts_smem_sz) );
      uint32_t const out_smem_sz = work.dsz("pels_tile")*work.dsz("out_chan_tile")*work.dsz("pels"); // note: == oi->tpb*work.dsz("pels")
      rcg->set( "out_smem_sz", str(out_smem_sz) ); // note: unused, but assumed that all_smem_sz >= out_smem_sz
      uint32_t const all_smem_sz = std::max( out_smem_sz, filts_smem_sz+in.dstride("blk_iter") ); // note: %(in_blk_iter_sz) == in_smem_sz
      rcg->set( "all_smem_sz", str(all_smem_sz) );

      // generate smem loads
      gen_filts_smem_loads( rcg, filts_smem_sz );
      uint32_t const in_smem_load_iter = u32_ceil_div( in.dstride("blk_iter"), rcg->tpb );    
      for( uint32_t i = 0; i != in_smem_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string eif;
	if( (i+1)*rcg->tpb > in.dstride("blk_iter") ) { rcg->line( "smem_loads", "if( "+ixe+" < %(in_blk_iter_sz)) { ");eif = "}";}
	rcg->line( "smem_loads", strprintf("    in_smem[%s] = in[ blk_in_ix_base + (%%(tpb)*%s) ];%s\n",
					     ixe.c_str(),str(i).c_str(),eif.c_str()) );
      }
      rcg->set( "out_chan_tile", "(%(GRP_ID_1D_out_chan_blk)*%(work_out_chan_tile_dim)+%(LOC_ID_1D_out_chan_tile))");
      rcg->set( "out_chan_ix","(%(out_chan_tile)*%(work_out_chan_dim))" );

      // rcg->line( "stores", "  if( %(out_line_img) >= %(out_ix_img_dim) ) { return; } "; // not possible due to no-partial-imgs-per-block
      // FIXME: should somehow assert that both out_ix and pel_ix_N have the same dims here
      // FIXME: out_pel must be per-tpix (again)
      if( out.get_dim_by_name("blk") ) { // aka if(write_xposed) -- if this dim names exists in the output, we know to write in xposed format
	// padded # of in chans of next layer  == out.dsz("blk_iter")*out.dsz("blk_iter_chan")
	// padded # of out chans of this layer == work.dsz("out_chan_blk")*work.dsz("out_chan_tile")*work.dsz("out_chan")
	// if these are ==, we don't have to worry about bounds-checking our writes to out in the chan dim
	assert_st( work.dsz("out_chan_blk")*work.dsz("out_chan_tile")*work.dsz("out_chan") == out.dsz("blk_iter")*out.dsz("blk_iter_chan") );
	// padded # of in pels of next layer:  == out.dsz("blk")*out.dsz("blk_pel")
	// padded # of out pels of this layer: == work.dsz("pels_blk")*work.dsz("pels_tile")*work.dsz("pels")
	// if these are ==, we don't have to worry about bounds-checking our writes to out in the pel dim
	assert_st( work.dsz("pels_blk")*work.dsz("pels_tile")*work.dsz("pels") == out.dsz("blk")*out.dsz("blk_pel") );

	// we assume out_blk_pel_dim (== noi->thr_per_blk.d[0]*t_tile_sz) is divisible by t_tile_sz. but let's check it explicitly:
	// FIXME_WXP: i don't see where we assume this, and hence i dunno what t_tile_sz refers to below. poop. assert is removed for now:
	// assert_st( (out.dsz("blk_pel") % t_tile_sz) == 0 );
	// we assume the out chans are a single (span of) dims in out. FIXME: check this?. FIXME_WXP: what does this even mean?

	//rcg->line( "stores", "  int32_t xpbuf[%(work_out_chan_dim)];\n";
	// FIXME: assumes (for GRP_ID_1D_pels_blk*... term) that input and output block have same # of pels ... too strong?
	assert_st( out.dsz("blk_pel") == in.dsz("blk_pel") );
	rcg->line( "stores", "int32_t const out_ix = (%(GRP_ID_1D_out_chan_blk)*%(work_out_chan_tile_dim)*%(work_out_chan_dim))*%(out_blk_iter_chan_sz) + %(GRP_ID_1D_pels_blk)*%(out_blk_sz);" ); 
	rcg->line( "stores", "int32_t xpbuf_rd_pel;" );
	rcg->line( "stores", "int32_t xpbuf_rd_chan;" );

	for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	  // transpose each thread's tx'th out_chan (= work_out_chan_dim out chans across all threads) into xpbuf (again across all threads)
	  // such that we can do (mostly) sequential writes to global memory for this set of work_out_chan_dim out chans
	  rcg->line( "stores", "  BARRIER_SYNC;" );
	  for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) { // out_tile[] (registers) -> all_smem[]
	    rcg->line( "stores", strprintf( "out_smem_off[%%(tpb)*%s] = %s;", str(ty).c_str(), 
					    add_bias_then_maybe_relu(rcg,work,tx,ty).c_str() ) );
	  }
	  rcg->line( "stores", "  BARRIER_SYNC;" );
	  for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) { // all_smem[] -> [xpbuf[] (registers)] -> out[] (global)
	    // here, we reshape the threads so that the total threads across iterations (%(tbp)*work.dsz("pels")) covers
	    // the space of the data in out_smem as a (simple) 2D array ordered as chan:pel. thus, for each thread, we
	    // have a single chan and pel index that we must read from smem and write to global memory. this mapping is
	    // such that the writes to global memory are somewhat sequential (with jumps at chan boundaries). however,
	    // for the reads from smem we just calculate the correct index and hope for the best. note that the actual
	    // output chan indexes read/written to here are strided by %(work_out_chan_dim) and offset by tx.
	    string const obe = "(LOC_ID_1D + %(tpb)*"+str(ty)+")";
	    rcg->line( "stores", "  xpbuf_rd_pel = "+obe+" %% %(out_blk_pel_dim) ;" );
	    rcg->line( "stores", "  xpbuf_rd_chan = "+obe+" / %(out_blk_pel_dim) ;" );
	    rcg->line( "stores", strprintf( "out[out_ix + xpbuf_rd_pel + (xpbuf_rd_chan*%%(work_out_chan_dim)+%s)*%%(out_blk_iter_chan_sz)] = "
					      "all_smem[xpbuf_rd_chan+(xpbuf_rd_pel %%%% %%(work_pels_dim))*%%(tpb)"
					      "+ (xpbuf_rd_pel / %%(work_pels_dim))*%%(work_out_chan_tile_dim) ];",
					      str(tx).c_str() ) );
	  }
	  for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) { // xpbuf[] registers -> out[] (global)
	    // TODO/UNUSED?
	  }	
	}
      } else {
	rcg->line( "stores", "  int32_t tpix[%(work_pels_dim)];" );
	rcg->line( "stores", "  int32_t tcix[%(work_out_chan_dim)];" );
	for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) { 
	  insert_nda_ix_exprs( rcg->tf_exprs, "out_pel_" + str(ty), must_find(rcg->all_ix_dims,"out_ref_pel"),
			       "( (%(GRP_ID_1D_pels_blk)*%(work_pels_tile_dim) + %(LOC_ID_1D_pels_tile))*%(work_pels_dim) + "+str(ty)+" )" );
	  rcg->line( "stores", strprintf( "  tpix[%s] = %%(out_pel_%s_img)*%%(out_img_sz) + "
					    " %%(out_pel_%s_x)*%%(out_x_sz) + %%(out_pel_%s_y)*%%(out_y_sz) " // FIXME_WXP:restore: y:x adj-dim opt?
					    "  ; // cache out pel ixs",
					    str(ty).c_str(), str(ty).c_str(), str(ty).c_str(), str(ty).c_str() ) );
	}
	for( uint32_t ty = 0; ty != work.dsz("out_chan"); ++ty ) { 
	  rcg->line( "stores", strprintf( "  tcix[%s] = (%%(out_chan_ix)+%s)*%%(out_chan_sz); // cache out chan ixs",
					    str(ty).c_str(), str(ty).c_str() ) );
	}
	for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	  rcg->line( "stores", "  if( %(out_pel_"+str(ty)+"_img) >= %(out_img_dim) ) { return; } "
		       "// this pel and the following are off-the-end pels, so don't store them." );
	  for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	    rcg->line( "stores", strprintf( "if( tcix[%s] < (%%(out_chan_dim)*%%(out_chan_sz)) ) { out[ tpix[%s] + tcix[%s] ] = %s; }",
					      str(tx).c_str(), str(ty).c_str(), str(tx).c_str(), 
					    add_bias_then_maybe_relu(rcg,work,tx,ty).c_str() ) );
	  }
	}
      }
      for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	  rcg->line( "dummy_stores", strprintf( "out_off[%s] = %s;", 
						  str((ty*work.dsz("out_chan")+tx)*rcg->tpb).c_str(), 
						add_bias_then_maybe_relu(rcg,work,tx,ty).c_str() ) );
	}
      }
      for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	rcg->line( "bias_loads", strprintf( "filts_strip[%s] = filts_smem_off[%s*%%(work_out_chan_tile_dim)];", 
					      str(tx).c_str(), str(tx).c_str() ) );
      }
      assert_st( in.dsz("blk_pel") == work.dsz("pels_tile")*work.dsz("pels") ); // by input xform design
      for( uint32_t ict = 0; ict != in.dsz("blk_iter_chan"); ++ict ) {
	for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	  rcg->line( "inner_loop_body", strprintf( "filts_strip[%s] = filts_smem_off[(%s*%%(filts_in_chan_sz))+%s*%%(work_out_chan_tile_dim)];", 
						     str(tx).c_str(), str(ict).c_str(), str(tx).c_str() ) );
	}
	for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) { 
	  rcg->line( "inner_loop_body", strprintf( "in_strip[%s] = in_smem_off[(%s*%%(in_blk_pel_dim)+%s)];",
						     str(ty).c_str(), str(ict).c_str(), str(ty).c_str() ) );
	}
	for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	  for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	    rcg->line( "inner_loop_body", strprintf( "out_tile[%s] += filts_strip[%s]*in_strip[%s];", 
						       str((ty*work.dsz("out_chan")+tx)).c_str(), str(tx).c_str(), str(ty).c_str() ) );
	  }
	}
      }
      rcg->has_final_flags_arg = 1;
    }

    void gen_op_tconv( rtc_call_gen_t * rcg ) {
      dims_t const & stride = rcg->get_arg_dims_by_name( "stride" );
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      dims_t const & filts = rcg->get_arg_dims_by_name( "filts" );
      dims_t const & in = rcg->get_arg_dims_by_name( "in" );
      uint32_t const filts_smem_sz = filts.dstride("y");
      rcg->set( "filts_smem_sz", str(filts_smem_sz) );
      gen_filts_smem_loads( rcg, filts_smem_sz );
      rcg->line( "filts_smem_loads", "filts_off += %(filts_smem_sz);" );
      uint32_t const in_smem_load_iter = u32_ceil_div( in.dstride("blk_in_chan"), rcg->tpb );  // in smem loads
      for( uint32_t i = 0; i != in_smem_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string eif;
	if( (i+1)*rcg->tpb > in.dstride("blk_in_chan") ) { rcg->line( "in_smem_loads", "if( "+ixe+" < %(in_blk_in_chan_sz)) { " );eif = "}";}
	rcg->line( "in_smem_loads", strprintf("in_smem[%s] = in[ blk_in_ix_base + (%%(tpb)*%s) ];%s",	ixe.c_str(),str(i).c_str(),eif.c_str()));
      }
      rcg->line( "in_smem_loads", "blk_in_ix_base += %(in_blk_in_chan_sz);" );

      for( uint32_t i = 0; i != in.dsz("blk_x"); ++i ) {
	rcg->line( "inner_loop_body", strprintf( "in_strip[%s] = in_smem_off[%s];", str(i).c_str(), str(i).c_str() ) );
      }
      assert_st( work.dsz("out_chan_tile") == filts.dsz("out_chan_tile") ); // also == %(filts_out_chan_reg_sz)
      for( uint32_t kx = 0; kx != filts.dsz("x"); ++kx ) {
	for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	  rcg->line( "inner_loop_body", strprintf( "filts_strip[%s] = filts_smem_off[%s*%%(filts_x_sz)+%s*%%(filts_out_chan_reg_sz)];", 
						     str(tx).c_str(), str(kx).c_str(), str(tx).c_str() ) );
	}
	for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	  for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	    rcg->line( "inner_loop_body", strprintf( "out_tile[%s] += filts_strip[%s]*in_strip[%s];",
						     str((ty*work.dsz("out_chan")+tx)).c_str(), 
						     str(tx).c_str(), str(ty*stride.dsz("x")+kx).c_str()));
	  }
	}
      }
      for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	rcg->line( "bias_loads", strprintf( "filts_strip[%s] = filts_smem_off[%s*%%(filts_out_chan_reg_sz)];", str(tx).c_str(), str(tx).c_str() ) );
      }
      //rcg->line( "stores", "  if( %(out_line_y) >= %(out_ix_y_sz) ) { return; }" ); // not possible
      rcg->line( "stores", "if( %(out_line_img) >= %(out_img_dim) ) { return; }" );
      rcg->line( "stores", "int32_t out_x = %(GRP_ID_1D_blk_bx)*%(work_pels_dim);" );
      rcg->line( "stores", "int32_t out_chan = (%(GRP_ID_1D_out_chan_blk)*%(work_out_chan_tile_dim) + %(LOC_ID_1D_out_chan_tile))*%(work_out_chan_dim);" );
      rcg->line( "stores", "GASQ float * out_off = out + %(out_line_img)*%(out_img_sz) + out_chan*%(out_chan_sz) + "
		   "%(out_line_y)*%(out_y_sz) + out_x*%(out_x_sz) ;" );

      for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	rcg->line( "stores", "if( (out_x + "+str(ty)+") >= %(out_x_dim) ) { return; } "
		     "// this x value and the following are off-the-end pels, so don't store them." );
	for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
#if 1
	  string const ve = add_bias_then_maybe_relu(rcg,work,tx,ty);

#else
	  string const ve = strprintf( "(filts_strip[%s])", str(tx).c_str() );
#endif
	  rcg->line( "stores", strprintf( "if( (out_chan + %s) < %%(out_chan_dim) ) { "
						   "out_off[ %s*%%(out_chan_sz) + %s*%%(out_x_sz) ] = %s; }",
						   str(tx).c_str(), str(tx).c_str(), str(ty).c_str(), ve.c_str() ) );
	}
      }
      rcg->has_final_flags_arg = 1;
    }
  };

  string conv_pipe_fwd_t::gen_func( rtc_func_sig_t const & rfs ) { 
    cnn_custom_codegen_t ccc; 
    return codegen.gen_func( &ccc, rfs ); 
  }

  void conv_pipe_fwd_t::gen_call( string const & fn, p_op_info_t const & oi ) { 
    // note: we generally assume all strides are 0 (uncalculated), and assume no (non-explicit) padding. it's unclear if
    // this is the best idea. note: we assume that all arg dims are already availible
    rtc_func_sig_t rfs;
    rfs.fn = fn;
    rfs.template_var_values = oi->template_var_values;
    rfs.ref_dims = oi->conv_ref_dims;
    string const & gen_fn = gen_func( rfs );
    fwd_calls.push_back( rcg_func_call_t{ gen_fn, oi->tag, oi->arg_map } );
  }

  // gen_node_var() creates a var directly corresponding to a pipe node.  usually, but not always, name == node_node; in
  // that case the var is directly mirroring a pipe node
  void conv_pipe_fwd_t::gen_node_var( string const & name, string const & node_name ) { 
    rtc->create_var_with_dims_floats( name, cp->must_get_node(node_name)->dims );
  }

  // quantize command line example:
  // export QOPTS="keep_bits=8,quantize=(_=(name=conv1,max_val=4096),_=(name=conv2,max_val=1024),_=(name=conv3,max_val=1024),_=(name=conv4,max_val=512),_=(name=conv5,max_val=512))

  // CUDA_VISIBLE_DEVICES=0 DISABLE_CUDNN=0 time boda test_lmdb --model-name=alexnet_ng_conv --num-to-read=1000 --run-cnet="(in_sz=(img=20),ptt_fn=%(models_dir)/%(model_name)/train_val.prototxt,trained_fn=%(models_dir)/%(model_name)/best.caffemodel,out_node_name=fc8-conv,compute_mode=1,conv_fwd=(mode=rtc,enable_stats=0,show_rtc_calls=0,${QOPTS}))"

  void conv_pipe_fwd_t::gen_ops_rec( string const & node_name ) {
    p_conv_node_t node = cp->must_get_node( node_name );
    if( node->top_for.empty() ) { gen_node_var( node_name, node_name ); }
    else { assert( node->top_for.size() == 1 ); } // multiple writers not handled

    // in-place ops for this node
    for( vect_p_conv_op_t::const_iterator j = node->in_place_ops.begin(); j != node->in_place_ops.end(); ++j ) { 
      gen_op( *j ); 
    }
    // generate stats gathering call
    // printf( "node_name=%s\n", str(node_name).c_str() );
    for( vect_p_quantize_ops_t::const_iterator i = quantize.begin(); i != quantize.end(); ++i ) {
      if( node_name != (*i)->name ) { continue; }
      gen_op_quantize( node_name, (*i)->max_val, (*i)->keep_bits );
    }
    if( enable_stats ) {
      vect_string new_stats_names = gen_op_stats( node_name );
      stats_names.insert( stats_names.end(), new_stats_names.begin(), new_stats_names.end() );
    }

    for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) {
      p_conv_op_t const & cop = cp->get_op( *i );
      if( !cop->on_seen_bot() ) { continue; } // wait till we've seen all bottoms
      for( vect_string::const_iterator j = cop->tops.begin(); j != cop->tops.end(); ++j ) {  // generate output nodes
	if( !cop->is(Convolution_coi) ) { gen_node_var( *j, *j ); } // only if not conv (which explicitly/manually creates node var)
      }
      gen_op( cop );
      for( vect_string::const_iterator j = cop->tops.begin(); j != cop->tops.end(); ++j ) { gen_ops_rec( *j ); }
    }
  }

  void conv_pipe_fwd_t::init( p_conv_pipe_t const & cp_ ) {
    cp = cp_;
    assert_st( cp );
    // note: the following modifies cp, so it's not clear that is exactly the right place for it to go.
    cp->fuse_relus_and_maybe_enable_write_xposed(); 
    op_infos.reset( new map_str_p_op_info_t );
    for( map_str_p_conv_op_t::iterator i = cp->convs->begin(); i != cp->convs->end(); ++i ) { 
      p_op_info_t & oi = (*op_infos)[i->first];
      assert_st( !oi );
      oi = make_shared< op_info_t >();
      oi->init( cp, i->second, enable_ipconv, enable_k1conv, enable_tconv, force_enable_tconv, t_tile_sz );
    }
    rtc->init();
    for( vect_string::const_iterator i = def.begin(); i != def.end(); ++i ) { 
      codegen.rtc_prog_str += "#define "+*i+" 1\n"; 
    }
    cp->topo_visit_setup();
    for( set_string::const_iterator i = cp->bots.begin(); i != cp->bots.end(); ++i ) { gen_ops_rec( *i ); }

    if( enable_bwai_test ) { // test bwai gen
      assert_st(0);
#if 0
      rtc->create_var_with_dims_floats( "a", dims_t{ {1000,1024}, {"M","K"}, 1 } );
      rtc->create_var_with_dims_floats( "b", dims_t{ {1000,1024}, {"N","K"}, 1 } );
      rtc->create_var_with_dims_floats( "c", dims_t{ {1000,1000}, {"M","N"}, 1 } );
      map_str_dims_t bwai_ref_dims;
      bwai_ref_dims["work"] = dims_t{ {10,10,10,10,32,10,10}, {"Mg","Ng","Mb","Nb","Kb","Mt","Nt"}, 1 };
      gen_call( "bwai", map_str_str(), "bwai_sgemm", {"a","b","c"}, bwai_ref_dims, 0 );
#endif
    }
    rtc->compile( codegen.rtc_prog_str, show_compile_log, enable_lineinfo );
    for( rtc_func_names_map_t::iterator i = codegen.rtc_func_names_map.begin(); i != codegen.rtc_func_names_map.end(); ++i ) { rtc->check_runnable( i->first, show_func_attrs ); }
    rtc->copy_ndas_to_vars( op_param_names, *cp->op_params ); // copy op_params in (FIXME/note: implicit  on names)
    for( set_string::const_iterator i = force_zero_names.begin(); i != force_zero_names.end(); ++i ) { rtc->set_var_to_zero( *i ); }
    for( vect_rcg_func_call_t::iterator i = init_calls.begin(); i != init_calls.end(); ++i ) { run_rfc( *i ); } // init-time-only calls
    rtc->finish_and_sync();
  }

  void conv_pipe_fwd_t::run_rfc( rcg_func_call_t & rfc ) { codegen.run_rfc( rtc, show_rtc_calls, rfc, flags );  }

  void conv_pipe_fwd_t::run_fwd( vect_string const & to_set_vns, p_map_str_p_nda_float_t const & fwd, vect_string const & to_get_vns ) {
    if( enable_double_run ) {
      // optional: run fwd rfc's one for testing/flushing/cache setup. note: ~*doubles* total run time ...
      for( vect_rcg_func_call_t::iterator i = fwd_calls.begin(); i != fwd_calls.end(); ++i ) { run_rfc( *i ); }
    }
    timer_t t("conv_pipe_fwd_t::run_fwd");
    if( enable_prof ) { rtc->profile_start(); }
    //printf("run_fwd() begin\n");
    rtc->copy_ndas_to_vars( to_set_vns, *fwd ); // copy sources in
    //printf("run_fwd() exec\n");
    for( vect_rcg_func_call_t::iterator i = fwd_calls.begin(); i != fwd_calls.end(); ++i ) { run_rfc( *i ); }
    rtc->finish_and_sync();
    float const compute_dur = fwd_calls.empty() ? 0.0f : rtc->get_dur( fwd_calls.front().call_id, fwd_calls.back().call_id );
    if( enable_prof ) { rtc->profile_stop(); }
    if( !per_call_fn.empty() ) {
      p_ofstream out = ofs_open( per_call_fn );
      (*out) << strprintf("net.args.runtime=%s\n", str(compute_dur/1000.0).c_str() );
      for( vect_rcg_func_call_t::iterator i = fwd_calls.begin(); i != fwd_calls.end(); ++i ) {
	rcg_func_call_t & rfc = *i;
	if( rfc.call_tag.empty() ) { continue; }
	float const rfc_dur = rtc->get_dur( rfc.call_id, rfc.call_id );
	(*out) << strprintf( "per_layer_time['%s']=per_layer_time.get('%s',0.0) + %s # %s \n", 
			     str(rfc.call_tag).c_str(), str(rfc.call_tag).c_str(), str(rfc_dur/1000.0).c_str(), rfc.rtc_func_name.c_str() );
      }
      cp->dump_ops( *out );
    }
    //printf("run_fwd() copy out\n");
    rtc->copy_vars_to_ndas( to_get_vns, *fwd ); // copy requested vars out
    update_stats();
    rtc->release_per_call_id_data();
    //printf("run_fwd() done\n");
  }
  
#include"gen/rtc_fwd.cc.nesi_gen.cc"
}
