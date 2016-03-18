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
    map_str_str str_vals; // str->str templates+values to pass directly to generated code (e.g. lrn params)
    map_str_dims_t dims_vals; // work + conv-type specific dims
    map_str_str arg_map; // map from func arg names to call-site arg names (in this case, just in global/rtc scope)
    string cts; // cts --> conv-type-str

    string get_arg( string const & an ) { return must_find( arg_map, an ); }
    dims_t const & get_arg_dims( string const & an ) { return must_find( dims_vals, an ); }
    void set_arg( p_rtc_compute_t const & rtc, string const & an, string const & vn ) {
      must_insert( dims_vals, an, rtc->get_var_dims_floats(vn) );
      must_insert( arg_map, an, vn );
    }
    void set_null_arg( string const & an ) { must_insert( dims_vals, an, dims_t() ); }
    void erase_arg( string const & an ) { must_erase( dims_vals, an ); must_erase( arg_map, an ); }
    void reset_arg( p_rtc_compute_t const & rtc, string const & an, string const & vn ) { 
      erase_arg(an); set_arg(rtc,an,vn);
    }

    void init( p_conv_pipe_t const & cp, p_conv_op_t const & cop, bool const & enable_ipconv,
	       bool const & enable_k1conv, bool const & enable_tconv, bool const & force_enable_tconv,
	       uint32_t const t_tile_sz ) {
      tag = cop->tag;
      str_vals = cop->str_vals;
      dims_vals = cop->dims_vals;
      assert_st( cop->tops.size() >= 1 );
      assert_st( cop->bots.size() >= 1 );
      // add all bots/tops as ref dims and track the mapping from arg name to external (call-scope) name
      for( uint32_t i = 0; i != cop->bots.size(); ++i ) { 
	must_insert( dims_vals, cop->coi->bot_an(i), cp->must_get_node( cop->bots[i] )->dims );
	must_insert( arg_map, cop->coi->bot_an(i), cop->bots[i] );
      }
      for( uint32_t i = 0; i != cop->tops.size(); ++i ) { 
	must_insert( dims_vals, cop->coi->top_an(i), cp->must_get_node( cop->tops[i] )->dims ); 
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
	dims_vals["in_ref"] = in_dims; // tconv needs the standard input dims for reference
	u32_pt_t kern_sz;
	if( has( cop->dims_vals, "kern_sz" ) ) { kern_sz = cop->kern_sz(); }
	else {
	  if( is_pool ) { kern_sz = get_xy_dims( ni->dims ); }
	  else if( cop->is( Spreading_coi ) ) { kern_sz = get_xy_dims( no->dims ); }
	  else { assert_st(0); }
	  dims_vals["kern_sz"] = dims_t{ {kern_sz.d[1],kern_sz.d[0]}, {"y","x"}, 1 }; // FIXME: not ideal ...
	} 
	u32_pt_t const in_pad = cop->in_pad();
	u32_pt_t const stride = cop->stride();
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

	  dims_vals["bck_in_pad"] = dims_t( vect_uint32_t{ uint32_t(bck_in_pad.d[1]), uint32_t(bck_in_pad.d[0]) }, 
						vect_string{"y","x"}, 1 );
	  dims_vals["bck_pad_in_off"] = dims_t( vect_uint32_t{ bck_pad_in_off.d[1], bck_pad_in_off.d[0] }, vect_string{"y","x"}, 1 );

	  p_conv_node_t ogl = cp->must_get_node(get_arg("out_grad_loss"));
	  p_conv_node_t fgl = cp->must_get_node(get_arg("filts_grad_loss"));

	  gbt_tile_t gbt;
	  dims_vals["oix"] = dims_t(  vect_uint32_t{ no->dims.dsz("chan"), stride.d[1], stride.d[0] }, 
					  vect_string{ "in_chan", "sy", "sx" }, 1 );
	  dims_vals["pix"] = dims_t(  vect_uint32_t{ no->dims.dsz("img"), 
		uint32_t(bck_pels_sz.d[1]), uint32_t(bck_pels_sz.d[0]) }, vect_string{ "img", "y", "x" }, 1 );
	  gbt.init( t_tile_sz, 128, u32_pt_t( dims_vals["pix"].dims_prod(), dims_vals["oix"].dims_prod()));
	  dims_t work;
	  work.add_dims( "pels_blk", gbt.num_blk.d[0] );
	  work.add_dims( "out_ix_blk", gbt.num_blk.d[1] );
	  work.add_dims( "pels_tile", gbt.thr_per_blk.d[0] );
	  work.add_dims( "out_ix_tile", gbt.thr_per_blk.d[1] );
	  work.add_dims( "pels", gbt.mn_per_thr.d[0], "out_ix", gbt.mn_per_thr.d[1] );
	  work.calc_strides();
	  dims_vals["work"] = work;
	  dims_vals["fioc"] = dims_t( vect_uint32_t{ ogl->dims.dsz("chan"), u32_ceil_div(kern_sz.d[1],stride.d[1]), 
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
	  dims_vals["work_fb"] = work_fb;
	  dims_vals["fioc_fb"] = dims_t( vect_uint32_t{ ogl->dims.dsz("img"), ogl->dims.dsz("y"), ogl->dims.dsz("x") },
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
	  dims_vals["work"] = work;
	  dims_vals["out_ref"] = no->dims; // k1conv and in_tile_xpose need the standard output dims for reference
	  dims_vals["in_xp"] = in_dims; // cached final desired format for input (original 'standard' format is stored as "in_ref" earlier)
	  // 'standard' and desired/xformed filter dims. we don't currently xform the biases (although maybe we should).
	  dims_vals["filts_xp"] = dims_t( vect_uint32_t{ work.dsz("out_chan_blk"),ni->dims.dsz("chan"), kern_sz.d[1], kern_sz.d[0],
		work.dsz("out_chan"),work.dsz("out_chan_tile")}, vect_string{"out_chan_blk","in_chan","y","x","out_chan_reg","out_chan_tile"}, 1 );
	  // dims_t( vect_uint32_t{cop->out_chans}, vect_string{"out_chan"}, 1 );
	} // end if(is_conv)
      }
    }
  };

  typedef shared_ptr< dims_t > p_dims_t; 

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

    filename_t rtc_func_sigs_fn; //NESI(default="rtc_func_sigs.txt",help="file to hold all generated func signatures")
    p_dims_t dummy_dims; // NESI(help="HACK: dummy NESI var of type dims_t (otherwise unused) to force tinfo generation. see map_str_T FIXME in nesi.cc")

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
    map_str_dims_t dims_vals; // note:constant after initial setup
    vect_string cur_ins;
    for( uint32_t i = 0; i != reds.size(); ++i ) {  // input dims (const); initial inputs
      must_insert( dims_vals, reds[i] + "_in", arg_dims ); 
      must_insert( dims_vals, reds[i] + "_out", arg_dims ); 
      cur_ins.push_back( top_in ); 
    } 
    while( in_sz > 1 ) {
      string const func = gen_func( rtc_func_sig_t{ "var_stats", dims_vals, {} } );
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
    string const func = gen_func( rtc_func_sig_t{ "quantize", {{"out",rtc->get_var_dims_floats(top_in)}}, {} } );
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
	// note: oi->str_vals is overwritten each iter; also, oi->cop->tag+"__copy" is reused for all calls (FIXME either/both?)
        oi->str_vals = { {"ocix",str(chans_out_done)} };
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
	// note: oi->str_vals is overwritten each iter; also, oi->cop->tag+"__copy" is reused for all calls (FIXME either/both?)
        oi->str_vals = { {"icix",str(chans_in_done)} };
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
	string const filts_xp_fn = gen_func( rtc_func_sig_t{ "xpose_filts", oi->dims_vals, oi->str_vals } );
	oi->reset_arg( rtc, "filts", gen_apply_func_to_var( "filts", oi->get_arg("filts"), 
							    "filts_xp", oi->get_arg_dims("filts_xp"), filts_xp_fn ) );
      }
      //in_arg_ids.push_back( cop->bots[2] ); // biases
      if( oi->cts == tconv_str ) {
	string const xp_fn = gen_func( rtc_func_sig_t{ "in_tile_xpose", oi->dims_vals, oi->str_vals } );
	oi->reset_arg( rtc, "in", gen_apply_func_to_var( "in", oi->get_arg("in"),
							 "in_xp", oi->get_arg_dims("in_xp"), xp_fn ) );
      } else if( oi->cts == k1conv_str ) {
	if( oi->get_arg_dims("in") != oi->get_arg_dims("in_xp") ) { 
	  // if dims not exactly right, assume they are 'normal' dims and convert. FIXME: fails if unexpected format.
	  string const xp_fn = gen_func( rtc_func_sig_t{ "xpose_in", oi->dims_vals, oi->str_vals } );
	  oi->reset_arg( rtc, "in", gen_apply_func_to_var( "in", oi->get_arg("in"),
							   "in_xp", oi->get_arg_dims("in_xp"), xp_fn ) );
	} 	
      } 
      dims_t no_dims = oi->dims_vals["out_ref"];
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
      double const dropout_ratio = lc_str_d( oi->str_vals["dropout_ratio"] );
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
	dims_t const & ogl_xp_dims = ogl_dims; // oi->dims_vals["out_grad_loss"];
	string ogl_xp_fn = gen_func( rtc_func_sig_t{ "btconv_ogl_xpose", {ogl_dims,ogl_xp_dims}, 
	      oi->dims_vals, oi->str_vals } );
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

  string conv_pipe_fwd_t::gen_func( rtc_func_sig_t const & rfs ) { 
    p_custom_codegen_t ccc = make_cnn_custom_codegen_t();
    return codegen.gen_func( ccc.get(), rfs ); 
  }

  void conv_pipe_fwd_t::gen_call( string const & fn, p_op_info_t const & oi ) { 
    // note: we generally assume all strides are 0 (uncalculated), and assume no (non-explicit) padding. it's unclear if
    // this is the best idea. note: we assume that all arg dims are already availible
    rtc_func_sig_t rfs( fn, oi->dims_vals, oi->str_vals );
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
    //codegen.write_rtc_func_sigs( rtc_func_sigs_fn );
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
    rtc->finish_and_sync();
    if( enable_prof ) { rtc->profile_start(); }
    //printf("run_fwd() begin\n");
    {
      timer_t t("conv_pipe_fwd_t::set_vars");
      rtc->copy_ndas_to_vars( to_set_vns, *fwd ); // copy sources in
      rtc->finish_and_sync();
    }
    //printf("run_fwd() exec\n");
    {
      timer_t t("conv_pipe_fwd_t::run_fwd");
      for( vect_rcg_func_call_t::iterator i = fwd_calls.begin(); i != fwd_calls.end(); ++i ) { run_rfc( *i ); }
      rtc->finish_and_sync();
    }
    //printf("run_fwd() copy out\n");
    {
      timer_t t("conv_pipe_fwd_t::get_vars");
      rtc->copy_vars_to_ndas( to_get_vns, *fwd ); // copy requested vars out
      rtc->finish_and_sync();
    }
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
    update_stats();
    rtc->release_per_call_id_data();
    rtc->finish_and_sync();
    //printf("run_fwd() done\n");
  }
  
#include"gen/rtc_fwd.cc.nesi_gen.cc"

}
