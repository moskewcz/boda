// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"has_conv_fwd.H"
#include"timers.H"
#include<boost/filesystem.hpp>
#include<boost/lexical_cast.hpp>
#include"lexp.H"
#include"conv_util.H"

#include"rtc_compute.H"

namespace boda 
{
  using boost::filesystem::path;

  typedef map< string, uint32_t > map_str_u32_t;
  typedef map< string, float > map_str_float_t;

  struct rtc_func_t { 
    string name;
    bool finalized;
    bool has_final_flags_arg;
    vect_uint32_t arg_sizes;
    uint32_t tpb;
    uint32_t blks;
  };
  typedef map< string, rtc_func_t > rtc_funcs_t;

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

  struct rtc_func_param_info_t { 
    string name; 
    string val; 
    bool operator == ( rtc_func_param_info_t const & o ) const { return name == o.name && val == o.val; }
    bool operator < ( rtc_func_param_info_t const & o ) const { 
      if( name != o.name ) { return name < o.name; }
      if( val != o.val ) { return val < o.val; }
      return 0;
    }
  };
  struct rtc_u32_param_info_t { string name; uint32_t val; };
  typedef vector< rtc_func_param_info_t > vect_rtc_func_param_info_t; 

  struct op_info_t {
    // --- phase 1 info --- filled in during init, independantly for all operations, in no particular order
    vect_rtc_func_param_info_t template_var_values; // str->str templates+values to pass directly to generated code (e.g. lrn params)
    dims_t work;
    dims_t in_dims;

    p_conv_op_t cop;
    p_conv_node_t no;
    bool is_conv;
    bool is_pool;
    p_conv_node_t ni;

    // valid if: is_conv == 1
    bool conv_has_relu;
    uint32_t in_pad;
    uint32_t kern_sz;
    uint32_t stride;
    dims_t no_dims; // dims for conv output. defaults to dims for no from conv_pipe, may be overidden in op generation

    bool is_k1conv;
    bool is_s1conv;
    bool is_tconv; // FIXME/TODO: mostly unused

    // --- phase 2 info --- filled in during breadth-first inputs->outputs creation phase (i.e. gen_op())
    // when filling these in, we can assume all phase 1 + phase 2 parent info exists.
    bool single_k1conv_output;

    void init( p_conv_pipe_t const & cp, p_conv_op_t const & cop_, uint32_t const & num_imgs,
	       bool const & enable_k1conv, bool const & enable_s1conv, bool const & enable_tconv, bool const & force_enable_tconv,
	       uint32_t const t_tile_sz ) {
      cop = cop_;
      assert_st( cop->tops.size() >= 1 );
      assert_st( cop->bots.size() >= 1 );
      no = cp->must_get_node( cop->tops[0] );
      if( !cop->is(Concat_coi) ) { ni = cp->must_get_node( cop->bots[0] ); } // null for Concat where we shouldn't use it, otherwise first input

      is_conv = cop->is( Convolution_coi );
      is_pool = cop->is( Pooling_coi );
      // if the output node's first in_place op is a ReLU, fuse it into this conv. a matching conditional later will omit the relu

      if( is_conv || is_pool || cop->is( Spreading_coi ) || cop->is( BckConv_coi ) ) {
	no_dims = no->cio.dims(num_imgs);
	in_dims = ni->cio.dims(num_imgs);
	conv_has_relu = (no->in_place_ops.size() > 0) && (no->in_place_ops[0]->is(ReLU_coi));
	if( conv_has_relu ) { no->in_place_ops.erase( no->in_place_ops.begin() ); } // remove fused relu
	// for now, we only attempt to handle the (common) case of uniform padding, kernel size, and stride
	assert_st( cop->in_pad.bnds_are_same() );
	assert_st( cop->in_pad.p[0].dims_are_same() );
	assert_st( cop->stride.dims_are_same() );
	u32_pt_t kern_sz_ = cop->kern_sz;
	if( kern_sz_.is_zeros() ) { kern_sz_ = ni->cio.sz; } // 'global' input special case
	assert_st( kern_sz_.dims_are_same() );

	in_pad = cop->in_pad.p[0].d[0];
	kern_sz = kern_sz_.d[0];
	stride = cop->stride.d[0];
	// also, for now, we'll only handle square inputs. however, this is probably too limiting for more than initial tests.
	assert_st( ni->cio.sz.dims_are_same() );
	is_k1conv = 0;
	is_s1conv = 0;
	is_tconv = 0;
	if( is_conv && enable_k1conv && (kern_sz == 1) && (stride == 1) 
	    && (no->cio.sz.d[0] >= 6) && (no->cio.sz.d[0] <= 300 ) && (no->cio.chans >= 64) ) 
	{ 
	  if( in_pad != 0 ) {
	    printf( "warning: can't use k1conv due only to non-zero padding on layer with kernel size 1\n" );
	  } else { is_k1conv = 1; }
	}
	else if( is_conv && enable_tconv && (force_enable_tconv || ((kern_sz <= 11) && (kern_sz > 1) && (no->cio.sz.d[0] >= 6))) )
	{ 
	  is_tconv = 1;
	}
	else if( is_conv && enable_s1conv && (stride == 1) && (kern_sz <= 5) && (kern_sz > 1) 
		 && (no->cio.sz.d[0] >= 6) && (no->cio.sz.d[0] <= 300 ) && (no->cio.chans >= 64) ) 
	{ 
	  is_s1conv = 1;
	}

	single_k1conv_output = 0; // may be set to 1 in phase 2, but default to 0 here

      }
     
      if( !is_conv ) { return; }
      // calc_blocking_conv()
      op_info_t * const oi = this;
      uint32_t const out_ix_sz = num_imgs * oi->no->cio.num_pels();

      // for reg blocking
      uint32_t const out_chan_tile_sz = u32_ceil_div( oi->no->cio.chans, t_tile_sz );

      uint32_t tix_pels_tile_sz_incr = 1;
      if( oi->is_s1conv ) {
	uint32_t const line_x_tile_sz = u32_ceil_div( oi->no->cio.sz.d[0], t_tile_sz );
	tix_pels_tile_sz_incr = line_x_tile_sz;
      }

      uint32_t const max_tpb = 128; // treated as a target, but not be exceeded
      uint32_t const goal_work_out_chan_tile_dim = 16; // sqrt( tpb ) above, more or less, but tweakable
      // determine block geometry in terms of WxH where the W is over out_chan_tile_sz (typ. ~64-1024+ / 8) and the H is
      // over pel_size (probably large-ish, at least in the cases we care most about perf for). ideally, we want
      // blocks with size sqrt(tpb) tiles. but, we can't (usefully) use a W smaller than the oi->no->cio.chans.
      uint32_t const work_out_chan_tile_dim = std::min( goal_work_out_chan_tile_dim, out_chan_tile_sz );
      uint32_t tix_pels_tile_sz = 0;
      uint32_t best_tbp = 0;
      while( 1 ) {
	uint32_t const maybe_tbp = (tix_pels_tile_sz+tix_pels_tile_sz_incr) * work_out_chan_tile_dim; // recalculate proposed tpb
	if( maybe_tbp > max_tpb ) { break; }
	tix_pels_tile_sz += tix_pels_tile_sz_incr;
	best_tbp = maybe_tbp;
      }
      assert_st( best_tbp );
      assert_st( best_tbp <= max_tpb );

      uint32_t const lines_sz = num_imgs * oi->no->cio.sz.d[1];
      if( oi->is_s1conv ) {
	assert_st( lines_sz * oi->no->cio.sz.d[0] * oi->no->cio.chans == out_ix_sz ); // by construction
	work.add_dims( "lines_blk", u32_ceil_div( lines_sz*tix_pels_tile_sz_incr, tix_pels_tile_sz ) );
      } else if( oi->is_tconv ) {
	assert( tix_pels_tile_sz >= 2 ); // if 1, would imply tconv_blk_max_imgs = 1 (but not sensible?)
	work.add_dims( "blk_bline", u32_ceil_div( lines_sz, tix_pels_tile_sz ), "blk_bx", u32_ceil_div( oi->no->cio.sz.d[0], t_tile_sz ) );
	uint32_t tconv_blk_max_imgs = 0;
	uint32_t blk_b_line = 0;
	for( uint32_t i = 0; i != work.dsz("blk_bline"); ++i ) {
	  uint32_t const blk_e_line = blk_b_line + tix_pels_tile_sz - 1;
	  uint32_t const blk_b_img = blk_b_line / oi->no->cio.sz.d[1];
	  uint32_t const blk_e_img = std::min( num_imgs - 1, blk_e_line / oi->no->cio.sz.d[1] );
	  uint32_t const blk_num_img = blk_e_img - blk_b_img + 1;
	  assert_st( blk_num_img );
	  max_eq( tconv_blk_max_imgs, blk_num_img );
	  blk_b_line = blk_e_line + 1;
	}
	assert_st( tconv_blk_max_imgs );
	// calc conservative value (may be lower in general or for some num_imgs) and use as check:
	uint32_t const conservative_conv_max_img_per_blk = 2 + ((tix_pels_tile_sz - 2)/oi->no->cio.sz.d[1]); 
	assert_st( tconv_blk_max_imgs <= conservative_conv_max_img_per_blk );
	//printf( "oi->no->cio.sz.d[1]=%s tix_pels_tile_sz=%s\n", str(oi->no->cio.sz.d[1]).c_str(), str(tix_pels_tile_sz).c_str() );
	//printf( "tconv_blk_max_imgs=%s\n", str(tconv_blk_max_imgs).c_str() );
	assert( tix_pels_tile_sz >= tconv_blk_max_imgs );
	uint32_t const tconv_blk_max_in_lines = (tix_pels_tile_sz - tconv_blk_max_imgs)*stride + kern_sz*tconv_blk_max_imgs;
	uint32_t const tconv_blk_x_sz = (t_tile_sz - 1)*stride + kern_sz;
	// the tconv/in_tile_xpose format is for use when both oi->ni->cio.sz.d[0/1] are small multiple of
	// t_tile_sz/tix_pels_tile_sz or >> than them (to avoid wasting too much work). each block will handle a (x,y)
	// window of the output of size (t_tile_sz,tix_pels_tile_sz) across bix_pels_blk_sz*t_tile_sz output chans. in
	// this case, we do not unroll across input chans, but we do unroll across kern_sz in X (and maybe in Y too for
	// small kernels).
	// note: "out_ix" from in_tile_xpose becomes "in_ix" for tconv; 
	// from the perspective inside tconv: the blk_y and blk_x dims are in input image space, the other dims are in output space
	// image space. other x/y's (in thread and block indexes) are all in output image space.
	in_dims = dims_t( vect_uint32_t{
	    work.dsz("blk_bline"), work.dsz("blk_bx"), oi->ni->cio.chans, tconv_blk_max_in_lines, tconv_blk_x_sz },
	  vect_string{"blk_bline","blk_bx","blk_in_chan","blk_y","blk_x"}, 1 );
      } else {
	uint32_t const pels_sz = out_ix_sz / oi->no->cio.chans;
	assert_st( pels_sz * oi->no->cio.chans == out_ix_sz ); // by construction
	uint32_t const pels_tile_sz = u32_ceil_div( pels_sz, t_tile_sz );
	work.add_dims( "pels_blk", u32_ceil_div( pels_tile_sz, tix_pels_tile_sz ) );
      }
      work.add_dims( "out_chan_blk", u32_ceil_div( out_chan_tile_sz, work_out_chan_tile_dim ) );

      // dims of per-group work (defines # threads per local group)
      if( oi->is_s1conv ) { 
	uint32_t const line_x_tile_sz = u32_ceil_div( oi->no->cio.sz.d[0], t_tile_sz );
	uint32_t const blk_num_lines = u32_ceil_div( tix_pels_tile_sz, line_x_tile_sz );
	assert_st( blk_num_lines * line_x_tile_sz == tix_pels_tile_sz ); // should be exact div by construction
	work.add_dims( "line", blk_num_lines, "line_x_tile", line_x_tile_sz );
      } 
      else if( oi->is_tconv ) { work.add_dims( "blk_y", tix_pels_tile_sz ); }
      else { work.add_dims( "pels_tile", tix_pels_tile_sz ); }
      work.add_dims(   "out_chan_tile",work_out_chan_tile_dim );

      work.add_dims( "pels",t_tile_sz,   "out_chan",t_tile_sz   ); // dims of per-thread work
      work.calc_strides();
      template_var_values = {{"conv_has_relu",str(conv_has_relu)},{"stride",str(stride)},{"in_pad",str(in_pad)}}; 

      if( is_k1conv ) { 
	uint32_t const in_blk_iter_chan_dim = 8; // FIXME: make into param?
	// the k1conv/xpose_in format is for use when stride=1, kernel_sz=1, and in_pad=0. we treat all input pixels as one 1D
	// vector across img:y:x, and divide them into blocks. we also block in the chan dim for unrolling.
	in_dims = dims_t( vect_uint32_t{
  oi->work.dsz("pels_blk"), u32_ceil_div(oi->ni->cio.chans,in_blk_iter_chan_dim), in_blk_iter_chan_dim, oi->work.dsz("pels_tile")*oi->work.dsz("pels")}, 
	  vect_string{"blk","blk_iter","blk_iter_chan","blk_pel"}, 1 ); 
      }

    }   
  };
  typedef shared_ptr< op_info_t > p_op_info_t; 
  typedef map< string, p_op_info_t > map_str_p_op_info_t;
  typedef shared_ptr< map_str_p_op_info_t > p_map_str_p_op_info_t; 

  struct rtc_func_sig_t { 
    string fn;
    vect_dims_t args;
    vect_rtc_func_param_info_t template_var_values; // str->str templates+values to pass directly to generated code (e.g. lrn params)
    bool operator < ( rtc_func_sig_t const & o ) const { 
      if( fn != o.fn ) { return fn < o.fn; }
      if( args != o.args ) { return args < o.args; }
      if( template_var_values != o.template_var_values ) { return template_var_values < o.template_var_values; }
      return 0;
    }
    
    uint32_t get_u32_tvv( string const & tvn ) {
      for( vect_rtc_func_param_info_t::const_iterator i = template_var_values.begin(); i != template_var_values.end(); ++i ) {
	if( i->name == tvn ) { return lc_str_u32( i->val ); }
      }
      rt_err("INTERNAL_ERROR: unknown template var name in codegen: '"+tvn+"'. typo?");
    }

    string gen_unused_fn( set_string & fns ) const {
      string maybe_fn_base = fn;
      set_string unique_dims;
      for( vect_dims_t::const_iterator i = args.begin(); i != args.end(); ++i ) {
	dims_t const & dims = *i;
	for( uint32_t i = 0; i != dims.sz(); ++i ) {
	  string const & dn = dims.names(i);
	  bool const did_ins = unique_dims.insert( dn ).second;
	  if( did_ins ) { maybe_fn_base += "__"+dn+"_"+str(dims.dims(i)); }
	}
      }
      string maybe_fn = maybe_fn_base;
      uint32_t uix = 0;
      while( has( fns, maybe_fn ) ) { ++uix; maybe_fn = maybe_fn_base + "__namegenconflict_" + str(uix); }
      must_insert( fns, maybe_fn );
      return maybe_fn;
    }
  };
  inline std::ostream & operator << ( std::ostream & out, rtc_func_sig_t const & o ) {
    return out << strprintf( "o.fn=%s o.args=%s", str(o.fn).c_str(), str(o.args).c_str() );
  }


  typedef map< rtc_func_sig_t, string > rtc_func_sigs_map_t;

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
    uint32_t enable_s1conv; //NESI(default=0,help="if 1, enable experimental s1conv special case")
    uint32_t enable_k1conv; //NESI(default=0,help="if 1, enable experimental k1conv special case")
    uint32_t enable_tconv; //NESI(default=0,help="if 1, enable experimental tconv special case")
    uint32_t force_enable_tconv; //NESI(default=0,help="if 1, force-enable experimental tconv special case even for not-sensible sizes")
    uint32_t enable_write_xpose; //NESI(default=0,help="if 1, enable experimental k1conv write xposing")
    uint32_t force_zero_bias; //NESI(default=0,help="if 1, force biases to zero")
    uint32_t flags; //NESI(default=0,help="dynamic flags to pass to kernels that request them (often to trick compiler)")
    uint32_t t_tile_sz; //NESI(default=8,help="register blocking tile size: compute t_tile_sz^2 outputs in registers per thread")
    vect_string dump_vars; // NESI(help="dump out values of these vars after forward")

    p_conv_pipe_t cp;
    p_map_str_p_op_info_t op_infos;

    uint32_t num_imgs;
    vect_string op_param_names;
    set_string filts_names;
    set_string inxp_names;
    set_string force_zero_names;

    vect_string stats_names;
    map_str_float_t stats_map;

    p_rtc_compute_t rtc; //NESI(default="(be=nvrtc)",help="rtc back-end to use")

    string rtc_prog_str;
    vect_rtc_func_call_t init_calls;
    vect_rtc_func_call_t fwd_calls;
    rtc_funcs_t rtc_funcs;
    

    virtual void init( p_conv_pipe_t const & cp_, uint32_t const & num_imgs_ );
    virtual void run_fwd( p_map_str_p_nda_float_t const & fwd, vect_string const & to_get_vns );

    void update_stats( void );
    string dump_var( string const & n );
    virtual string get_info_log( void );
  protected:
    vect_string gen_op_stats( string const & top_in );
    void gen_op_quantize( string const & top_in, uint32_t const & max_val, uint32_t const & keep_bits );

    // gen_call() related data
    set_string used_rtc_func_names;
    rtc_func_sigs_map_t rtc_func_sigs_map;
    string gen_func( rtc_func_sig_t const & rfs );
    void gen_call( string const & fn, p_op_info_t const & oi, vect_string const & args, 
		   vect_dims_t const & ref_dims = vect_dims_t(), bool const is_init_call = 0 );

    void gen_conv_filts( p_op_info_t const & oi ); // setup nodes and xforms for conv op filts/biases
    string gen_apply_func_to_var( string const & in_var, dims_t const & ret_dims, string const & func );

    void gen_node_var( string const & name, string const & node_name );
    void gen_op( p_conv_op_t const & cop );
    void gen_ops_rec( string const & node_name );

    void run_rfc( rtc_func_call_t & rfc );
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
    p_nda_float_t nda = rtc->copy_var_as_flat_nda( n );
    // dump nda
    ret += strprintf( "dupming var '%s'\n", str(n).c_str() );
    for( uint32_t i = 0; i != nda->dims.dims_prod(); ++i ) {
      ret += strprintf( "i=%s v=%s\n", str(i).c_str(), str(nda->cm_at1(i)).c_str() );
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
    vect_dims_t args_dims; // note:constant after initial setup
    vect_string cur_ins;
    for( uint32_t i = 0; i != reds.size(); ++i ) { args_dims.push_back( arg_dims ); cur_ins.push_back( top_in ); } // input dims (const); initial inputs
    for( uint32_t i = 0; i != reds.size(); ++i ) { args_dims.push_back( arg_dims ); } // output dims (const)
    while( in_sz > 1 ) {
      string const func = gen_func( rtc_func_sig_t{ "var_stats", args_dims, {} } );
      vect_string cur_outs;
      vect_string args = cur_ins;
      vect_string out_args;
      uint32_t const out_sz = u32_ceil_div( in_sz, must_find(rtc_funcs,func).tpb );
      for( uint32_t i = 0; i != reds.size(); ++i ) { 
	string cur_out = top_in + "_" + reds[i] + "_out_sz_" + str(out_sz);
	rtc->create_var_with_dims_floats( cur_out, dims_t{ {out_sz}, {"v"}, 1 } );
	cur_outs.push_back( cur_out );
	args.push_back( cur_out );
      }
      fwd_calls.push_back( rtc_func_call_t{ func, args,{},{}, {in_sz, primary_in}, "var_stats" } );
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
    string const func = gen_func( rtc_func_sig_t{ "quantize", {rtc->get_var_dims_floats(top_in)}, {} } );
    fwd_calls.push_back( rtc_func_call_t{ func, {},{top_in},{}, {max_val,drop_mask}, "quantize" } );
  }

  // setup nodes and xforms for conv op filts/biases. note: tracks oi->cop->tag (operation name) to do this only
  // once-per-op. since currently each op has a unique name and unique set of filts, this is okay/correct. but we'd need
  // to adjust it accordingly if various things are sharable/custom-namable.
  void conv_pipe_fwd_t::gen_conv_filts( p_op_info_t const & oi ) {
    bool const did_ins = filts_names.insert( oi->cop->tag ).second; 
    if( !did_ins ) { return; } // filts already set up for this op

    string const filts_id = oi->cop->tag + "_filts";
    dims_t const filts_dims( vect_uint32_t{oi->cop->out_chans,oi->ni->cio.chans,oi->kern_sz,oi->kern_sz}, 
			     vect_string{"out_chan","in_chan","y","x"}, 1 );
    string const filtsxp_id = filts_id + "_xposed";
    dims_t const filtsxp_dims( vect_uint32_t{oi->work.dsz("out_chan_blk"),oi->ni->cio.chans,oi->kern_sz,oi->kern_sz,t_tile_sz,
	  oi->work.dsz("out_chan_tile")}, 
				vect_string{"out_chan_blk","in_chan","y","x","out_chan_reg","out_chan_tile"}, 1 );
    string const biases_id = oi->cop->tag + "_biases";
    dims_t const biases_dims( vect_uint32_t{oi->cop->out_chans}, vect_string{"out_chan"}, 1 );

    rtc->create_var_with_dims_floats( filts_id, filts_dims );
    op_param_names.push_back( filts_id );
    rtc->create_var_with_dims_floats( filtsxp_id, filtsxp_dims );
    // FIXME: probably not right to use the conv op's oi here. but params should be empty; oi->call_tag is okay to use
    // (since the called func is appended). let's revisit when we're further along with gen_call() refactoring ...
    gen_call( "xpose_filts", oi, { filts_id, filtsxp_id }, {}, 1 ); 
    rtc->create_var_with_dims_floats( biases_id, biases_dims );
    op_param_names.push_back( biases_id );
  }

  // this assumes that in_var is valid/calculated, and returns ret_var=func(in_var). it assumes that func is a stateless
  // unary operator (with two args: {in,out}), so that only one unique ret_var need only be generated per unique
  // in_var/func<ret_dims> pair. ret_var is named in_var+"__"+func
  string conv_pipe_fwd_t::gen_apply_func_to_var( string const & in_var, dims_t const & ret_dims, string const & func ) {
    string const ret_var = in_var + "__" + func;
    bool const did_ins = inxp_names.insert( ret_var ).second;
    if( did_ins ) { // newly-seen/used ret_var, so create and calc it here
      rtc->create_var_with_dims_floats( ret_var, ret_dims );
      fwd_calls.push_back( rtc_func_call_t{ func, {in_var,ret_var}, {}, {}, {}, in_var + "__inxp" } );
    }
    return ret_var;
  }

  void conv_pipe_fwd_t::gen_op( p_conv_op_t const & cop ) {
    p_op_info_t const & oi = must_find( *op_infos, cop->tag );
    p_op_info_t poi;
    if( oi->ni && !oi->ni->top_for.empty() ) {
      assert_st( oi->ni->top_for.size() == 1 );
      poi = must_find( *op_infos, oi->ni->top_for[0] ); // single unique parent operation, needed for poi->single_k1conv_output
    }

    if( cop->is( Concat_coi ) ) {      
      uint32_t chans_out_done = 0;
      for( uint32_t bi = 0; bi != cop->bots.size(); ++bi ) {
	conv_io_t & cio_in = cp->must_get_node( cop->bots[bi] )->cio;
	assert_st( cio_in.sz == oi->no->cio.sz );
	assert_st( chans_out_done+cio_in.chans <= oi->no->cio.chans );
	// note: oi->template_var_values is overwritten each iter; also, oi->cop->tag+"__copy" is reused for all calls (FIXME either/both?)
        oi->template_var_values = { {"ocix",str(chans_out_done)} }; 
	gen_call( "copy", oi, {cop->bots[bi], cop->tops[0]} );
	chans_out_done += cio_in.chans;
      }
      assert_st( chans_out_done == oi->no->cio.chans );
      return;
    }

    if( oi->is_pool ) {
      oi->template_var_values = {{"kern_sz",str(oi->kern_sz)},{"stride",str(oi->stride)},{"avg_pool",str(oi->cop->avg_pool)},{"in_pad",str(oi->in_pad)}};
      oi->template_var_values.push_back( {"op", oi->cop->avg_pool ? "out_v += v" : "out_v = max( out_v, v )" } );
      oi->template_var_values.push_back( {"op_post", oi->cop->avg_pool ? "out_v /= (float)("+str(oi->kern_sz*oi->kern_sz)+")" : "" } );
      gen_call( "pool", oi, {oi->ni->name,oi->no->name} );
    } else if( oi->is_conv ) {
      gen_conv_filts( oi );
      // FIXME: decls dup'd with gen_conv_filts()
      string const filts_id = oi->cop->tag + "_filts";
      string const filtsxp_id = filts_id + "_xposed";
      string const biases_id = oi->cop->tag + "_biases";
      string const in_id = cop->bots[0];
      dims_t const & in_dims = rtc->get_var_dims_floats( in_id );

      vect_string in_arg_ids{ filtsxp_id, biases_id };

      // printf( "rf->name=%s oi->single_k1conv_output=%s poi->single_k1conv_output=%s oi->is_k1conv=%s\n", str(rf->name).c_str(), str(oi->single_k1conv_output).c_str(), poi ? str(poi->single_k1conv_output).c_str() : "<null>", str(oi->is_k1conv).c_str() );

      if( force_zero_bias ) { force_zero_names.insert( biases_id ); }
      if( oi->is_tconv ) {
	string const xp_fn = gen_func( rtc_func_sig_t{ "in_tile_xpose", {in_dims,oi->in_dims,oi->no->cio.dims(num_imgs),oi->work}, // 'normal' out dims are for ref
	    {{"stride",str(oi->stride)},{"kern_sz",str(oi->kern_sz)},{"in_pad",str(oi->in_pad)}} } );
	in_arg_ids.push_back( gen_apply_func_to_var( in_id, oi->in_dims, xp_fn ) );
      } else if( oi->is_k1conv ) {
	if( in_dims != oi->in_dims ) { // if dims not exactly right, assume they are 'normal' dims and convert. FIXME: fails if unexpected format.
	  string const xp_fn = gen_func( rtc_func_sig_t{ "xpose_in", {in_dims,oi->in_dims}, {} } );
	  in_arg_ids.push_back( gen_apply_func_to_var( in_id, oi->in_dims, xp_fn ) );
	} else { in_arg_ids.push_back( in_id ); }	
      } else {
	in_arg_ids.push_back( in_id );
      }

      if( oi->is_k1conv ) { 
	p_op_info_t noi;
	if( oi->no->in_place_ops.empty() && (oi->no->bot_for.size() == 1) ) { // if output feeds single non-in-place operation
	  noi = must_find( *op_infos, oi->no->bot_for[0] ); // next operation
	  if( noi->is_k1conv ) { oi->single_k1conv_output = enable_write_xpose; }
	}
	bool const write_xposed = oi->single_k1conv_output;
	dims_t out_ref_dims = oi->no_dims;
	if( write_xposed ) {
	  oi->no_dims = dims_t{ vect_uint32_t{noi->work.dsz("pels_blk"), oi->in_dims.dsz("blk_iter"),oi->in_dims.dsz("blk_iter_chan"),
					      noi->work.dsz("pels_tile")*noi->work.dsz("pels") }, { "blk", "blk_iter", "blk_iter_chan", "blk_pel"}, 1 };
	}
	rtc->create_var_with_dims_floats( oi->no->name, oi->no_dims );
	in_arg_ids.push_back( oi->no->name );
	gen_call( "k1conv", oi, in_arg_ids, {oi->work,out_ref_dims} );
      } else if( oi->is_s1conv ) { 
	rtc->create_var_with_dims_floats( oi->no->name, oi->no_dims ); // note: using default output dims here (okay for s1conv)
	in_arg_ids.push_back( oi->no->name );
	gen_call( "s1conv", oi, in_arg_ids, {oi->work} );
      } else if( oi->is_tconv ) { 
	rtc->create_var_with_dims_floats( oi->no->name, oi->no_dims ); // note: using default output dims here (okay for reg. tconv)
	in_arg_ids.push_back( oi->no->name );
	gen_call( "tconv", oi, in_arg_ids, {oi->work,in_dims} ); // original in_dims for reference (not xformed)
      } else { 
	rtc->create_var_with_dims_floats( oi->no->name, oi->no_dims ); // note: using default output dims here (okay for reg. conv)
	in_arg_ids.push_back( oi->no->name );
	gen_call( "conv", oi, in_arg_ids, {oi->work} );
      }
      assert_st( oi->no->cio.chans == cop->out_chans );
    } else if( cop->is( ReLU_coi ) ) {
      assert_st( oi->ni->name == oi->no->name ); // check that this is a single in-out in-place operation
      gen_call( "relu", oi, { oi->no->name } );
    } else if( cop->is( LRN_coi ) ) {
      oi->template_var_values = {{"local_size",str(cop->lrn_local_size)},{"alpha",str(cop->lrn_alpha)},{"beta",str(cop->lrn_beta)},{"k",str(cop->lrn_k)}};
      assert_st( oi->ni->cio.sz == oi->no->cio.sz );
      assert_st( oi->ni->cio.chans == oi->no->cio.chans );
      gen_call( "lrn", oi, { oi->ni->name, oi->no->name } );
    } else if( cop->is( Dropout_coi ) ) {
      // check that this is a single in-out in-place operation
      assert_st( oi->ni->name == oi->no->name );
      // ignore for fwd
    } else if( cop->is( Softmax_coi ) ) {
      gen_call( "softmax", oi, { oi->ni->name, oi->no->name } );
    } else if( cop->is( SoftmaxWithLoss_coi ) ) {
      string const prob_node_name = cop->tag + "_prob";
      gen_node_var( prob_node_name, cop->bots[0] );
      gen_call( "softmax", oi, { cop->bots[0], prob_node_name } );
      string const loss_per_pel = cop->tops[1] + "_per_pel"; // same size as label
      gen_node_var( loss_per_pel, cop->bots[1] );
      gen_call( "sm_grad_and_loss", oi, { prob_node_name, cop->bots[1], cop->tops[0], loss_per_pel} );
      gen_call( "sum_loss_over_imgs", oi, { loss_per_pel, cop->tops[1] } );
    } else if( cop->is( Spreading_coi ) ) {
      oi->template_var_values = {{"kern_sz",str(oi->kern_sz)},{"stride",str(oi->stride)},{"avg_pool",str(oi->cop->avg_pool)},{"in_pad",str(oi->in_pad)}};
      gen_call( "spreading", oi, { cop->bots[0], cop->bots[1], cop->bots[2], cop->tops[0] } );
    } else if( cop->is( ZeroIfNeg_coi ) ) {
      gen_call( cop->type, oi, { cop->bots[0], cop->bots[1], cop->tops[0] } );
    } else if( cop->is( BckConv_coi ) ) {
      // FIXME_EFB: { in, filts, biases, out_grad_loss } --> { in_grad_loss, filts_grad_loss, biases_grad_loss }
      // current: { in, out_grad_loss } --> { in_grad_loss }
      // FIXME_EFB: since filts/biases are implict inputs/outputs, we need to fabricate/recreate thier var names here
      string fwd_tag = cop->tag;
      bool const did_strip = maybe_strip_suffix( fwd_tag, "_bck" );
      assert_st( did_strip );
      string const filts_id = fwd_tag + "_filts";
      //string const biases_id = fwd_tag + "_biases";
      oi->template_var_values = {{"stride",str(oi->stride)},{"in_pad",str(oi->in_pad)}}; 
      gen_call( "BckConv_in_grad_loss", oi, { cop->bots[0], filts_id, /*biases_id,*/ cop->bots[1], cop->tops[0] } );
    } else { rt_err( "gen_op: unhandled op of type: " + cop->type ); }
  }

  struct arg_decl_t {
    string vn;
    string io_type;
    vect_dims_t ok_dims;
  };
  inline std::ostream & operator << ( std::ostream & out, arg_decl_t const & o ) {
    for( vect_dims_t::const_iterator i = o.ok_dims.begin(); i != o.ok_dims.end(); ++i ) {
      out << strprintf( "%s --- vn=%s io_type=%s ", str(*i).c_str(), str(o.vn).c_str(), str(o.io_type).c_str() );
    }
    return out;
  }
  typedef vector< arg_decl_t > vect_arg_decl_t; 
  
  struct ix_decl_t {
    string ix_vn;
    string arg_vn;
    vect_string use_dims;
  };
  typedef vector< ix_decl_t > vect_ix_decl_t; 

  void insert_nda_dims_sz( vect_pair_str_str & mss, string const & nda_vn, dims_t const & dims ) {
    for( uint32_t i = 0; i != dims.sz(); ++i ) {
      assert_st( dims[i].has_sz_and_stride_and_name() );
      mss.push_back( make_pair( nda_vn+"_"+dims[i].name+"_dim", str(dims[i].sz) ) );
      mss.push_back( make_pair( nda_vn+"_"+dims[i].name+"_sz", str(dims[i].stride) ) );
    }
    mss.push_back( make_pair( nda_vn+"_dims_prod", str(dims.dims_prod()) ) ); // also emit dim(0)*stride(0)?
  }

  void insert_nda_ix_exprs( vect_pair_str_str & mss, string const & ix_vn, dims_t const & dims, string ix_expr = string() ) {
    if( ix_expr.empty() ) { ix_expr = ix_vn; } // if no custom ix expression, use ix_vn directly as ix expr
    for( uint32_t i = 0; i != dims.sz(); ++i ) {
      assert_st( dims[i].has_sz_and_stride_and_name() );
      string v = (dims[i].stride > 1) ? "("+ix_expr+"/"+str(dims[i].stride)+")" : ix_expr;
      mss.push_back( make_pair( ix_vn+"_"+dims[i].name+"_nomod", v ) );      
      if( i ) { v = "("+v+"%%"+str(dims[i].sz)+")"; }
      mss.push_back( make_pair( ix_vn+"_"+dims[i].name, v ) );
    }
    mss.push_back( make_pair( ix_vn+"_dims_prod", str(dims.dims_prod()) ) ); // also emit dim(0)*stride(0)?
  }
  
  typedef map< string, dims_t > map_str_dims_t;

  struct rtc_call_gen_t {
    rtc_func_sig_t rfs;
    p_string rtc_func_template; // read from file
    vect_arg_decl_t arg_decls; // parsed out of rtc_func_template
    vect_ix_decl_t ix_decls; // parsed out of rtc_func_template
    vect_pair_str_str tf_exprs;
    rtc_func_t * rf;
    map_str_dims_t all_ix_dims;
    map_str_str cgs; // code-gen sections
    void cg_add_line( string const & sn, string const & line ) { 
      string & cg = cgs[sn];
      if(cg.empty()) { cg += "// begin "+sn+"\n"; } // adding first line in (new) section, add header
      cg += "   " + line + "\n"; 
    }
    dims_t const & get_arg_dims_by_name( string const & arg_vn ) {
      for( uint32_t i = 0; i != arg_decls.size(); ++i ) { if( arg_decls[i].vn == arg_vn ) { return rfs.args.at(i); } }
      rt_err( strprintf( "referenced arg '%s' not declared\n", str(arg_vn).c_str() ) );
    }
    
    void init( rtc_func_sig_t const & rfs_, string const & gen_fn, rtc_funcs_t & rtc_funcs, string & rtc_prog_str ) {
      rf = 0;
      rfs = rfs_;
      tf_exprs.push_back( make_pair( "rtc_func_name", gen_fn ) );
      for( vect_rtc_func_param_info_t::const_iterator i = rfs.template_var_values.begin(); i != rfs.template_var_values.end(); ++i ) {
	tf_exprs.push_back( make_pair( i->name, i->val ) );
      }
      rtc_func_template = read_whole_fn( (path(py_boda_test_dir()) / "rtc" / (rfs.fn+".cucl")).string() );
      parse_template();
      check_args();
      uint32_t num_nonref_args = 0;
      for( uint32_t i = 0; i != rfs.args.size(); ++i ) { 
	if( rfs.args[i].has_sz_and_stride_and_name() ) { insert_nda_dims_sz( tf_exprs, arg_decls[i].vn, rfs.args[i] ); }
	if( arg_decls[i].io_type != "REF" ) { ++num_nonref_args; }
      }

      rf = &rtc_funcs.insert( make_pair( gen_fn, rtc_func_t{gen_fn,0,0} ) ).first->second;
      assert_st( !rf->finalized );
      rf->arg_sizes.resize( num_nonref_args ); // values unused, but size must match # args at call time
      // FIXME: gen_call() doesn't handle num_nonref_args != arg_decls.size() case yet
      rf->tpb = 0;
      rf->blks = 0;
      uint32_t const default_tpb = 256; // FIXME: make 'reasonable default' tpb configurable/heurisitic
      for( vect_ix_decl_t::const_iterator i = ix_decls.begin(); i != ix_decls.end(); ++i ) {
	dims_t const & ix_arg_dims = get_arg_dims_by_name( i->arg_vn );
	if( !ix_arg_dims.has_sz_and_stride_and_name() ) { rt_err( "NEVER_SAY_NEVER, but can't create CUCL IX for dynamically-sized var" ); }
	dims_t ix_dims;
	if( i->use_dims.empty() ) { ix_dims = ix_arg_dims; } 
	else {
	  for( vect_string::const_iterator j = i->use_dims.begin(); j != i->use_dims.end(); ++j ) {
	    dim_t const * use_dim = ix_arg_dims.get_dim_by_name( *j );
	    if( !use_dim ) { rt_err( "specified use_dim '"+*j+"' not found in target arg's dims" ); }
	    ix_dims.push_back( *use_dim );
	  }
	}
	ix_dims.calc_strides(); // note: stride are garbage prior to this call (which is okay)
	must_insert( all_ix_dims, i->ix_vn, ix_dims );
	insert_nda_ix_exprs( tf_exprs, i->ix_vn, ix_dims );
	// special cases for index var names
	if( i->ix_vn == "GLOB_ID_1D" ) { 
	  // if GLOB_ID_1D is an index for some arg, assume we want 1 thread per element of that arg, and assume block
	  // size doesn't matter, so use a reasonable default.
	  if( rf->tpb || rf->blks ) { rt_err( "CUCL error: GLOB_ID_1D IX encoutered after setting tpb or blks (some other way(s))" ); }
	  rf->tpb = default_tpb;
	  rf->blks = u32_ceil_div( ix_dims.dims_prod(), rf->tpb );
	} else if( i->ix_vn == "GRP_ID_1D" ) {
	  if( rf->blks ) { rt_err( "CUCL error: GRP_ID_1D IX encoutered after setting blks (some other way)" ); }
	  rf->blks = ix_dims.dims_prod();
	} else if( i->ix_vn == "LOC_ID_1D" ) {
	  if( rf->tpb ) { rt_err( "CUCL error: LOC_ID_1D IX encoutered after setting tpb (some other way)" ); }
	  rf->tpb = ix_dims.dims_prod();
	}
      }
      if( !rf->tpb ) { rf->tpb = default_tpb; } // if not set, use a reasonable default
      // assert_st( rf->blks ); // too strong. if not set, dynamic # blks case

      // *** custom codegen hooks ***
      if( rfs.fn == "conv" ) { gen_op_conv(); } 
      else if( rfs.fn == "s1conv" ) { gen_op_s1conv(); } 
      else if( rfs.fn == "k1conv" ) { gen_op_k1conv(); } 
      else if( rfs.fn == "tconv" ) { gen_op_tconv(); } 

      // make these always availible as template vars, since why not?
      tf_exprs.push_back( make_pair( "tpb", str(rf->tpb) ) ); // should always be fixed/constant/valid (i.e. gen'd kernels never have dynamic tpb)
      tf_exprs.push_back( make_pair( "blks", str(rf->blks) ) ); // may be 0 if # of blocks is dynamic
      tf_exprs.push_back( make_pair( "warp_sz", str("UNKNOWN") ) ); // yeah, not the best, but probably not exactly wrong. don't use it for real

      rf->finalized = 1;
      instantiate_template( rtc_prog_str );
    }

    void check_args( void ) {
      string arg_check_error;
      if( rfs.args.size() != arg_decls.size() ) { arg_check_error = "declared/called argument count mismatch"; }
      else {
	for( uint32_t i = 0; i != rfs.args.size(); ++i ) {
	  if( !rfs.args[i].has_name() ) { arg_check_error += "call arg "+str(i)+ " must have names for all dims; "; }
	  if( rfs.args[i].has_sz_and_stride_and_name() && rfs.args[i].has_padding() ) { 
	    arg_check_error += "call args "+str(i)+ " must not have padding; "; } // FIXME: maybe too strong
	  bool matches_decl = 0;
	  for( uint32_t j = 0; j != arg_decls[i].ok_dims.size(); ++j ) {
	    if( rfs.args[i].matches_template( arg_decls[i].ok_dims[j] ) ) { matches_decl = 1; }
	  }
	  if( !matches_decl ) { arg_check_error += "call arg "+str(i)+" incompatible with decl arg "
	      "(dim count mismatch or dim (non-zero and thus req'd) name/size/stride mismatch; "; }
	}
      }
      if( !arg_check_error.empty() ) {
	string arg_err = "RTC call argument error: " + rfs.fn + ": " + arg_check_error + "\n";
	for( uint32_t i = 0; i != std::max(rfs.args.size(),arg_decls.size()); ++i ) {
	  arg_err += strprintf( "ARG[%s]:\n", str(i).c_str() );
	  if( i < arg_decls.size() ) { arg_err += strprintf( "  DECL: %s\n", str(arg_decls[i]).c_str() ); }
	  if( i < rfs.args.size() ) {  arg_err += strprintf( "  CALL: %s\n", str(rfs.args[i]).c_str() ); }
	}
	rt_err( arg_err );
      }
    }

    void instantiate_template( string & rtc_prog_str ) {
      rtc_prog_str += "// -- codegen begins for '"+rfs.fn+"'; template substituion table used (bulk sections ommited): --\n";
      for( vect_pair_str_str::const_iterator i = tf_exprs.begin(); i != tf_exprs.end(); ++i ) {
	rtc_prog_str += strprintf( "/* %s = %s */\n", str(i->first).c_str(), str(i->second).c_str() );
      }
      for( map_str_str::iterator i = cgs.begin(); i != cgs.end(); ++i ) { // terminate and emit bulk cgs
	i->second += "    // end "+i->first;
	tf_exprs.push_back( make_pair( i->first, i->second ) ); 
      } 
      lexp_name_val_map_t tf_nvm{ p_lexp_t() };
      tf_nvm.insert_leafs_from( tf_exprs );
      string rtc_func_str;
      str_format_from_nvm( rtc_func_str, *rtc_func_template, tf_nvm );
      rtc_prog_str += rtc_func_str;
      //printf( "rtc_func_name=%s rf->tpb=%s rf->blks=%s\n", str(rtc_func_name).c_str(), str(rf->tpb).c_str(), str(rf->blks).c_str()); 
      //rf->finalized = 1;
    }

    static dims_t dims_from_nda_spec( string const & nda_spec ) {
      vect_string const dim_names = split( nda_spec, ':' );
      dims_t arg_dims;
      // for now, assume template: (1) can handle any size for all nda dims (to relax, could allow spec of size,
      // then use that instead of 0/wild for sz below) (2) all dims are static (to relax: unclear what
      // syntax/encoding would be)
      for( vect_string::const_iterator i = dim_names.begin(); i != dim_names.end(); ++i ) { arg_dims.add_dims( *i, 0 ); }
      return arg_dims;
    }

    void parse_template( void ) {
      vect_string lines = split( *rtc_func_template, '\n' );
      for( vect_string::const_iterator i = lines.begin(); i != lines.end(); ++i ) {
	// find magic CUCL comment (if any) and process it
	string const mmc = get_part_after( *i, "//" );
	vect_string mmc_parts = split_ws( strip_ws( mmc ) );
	if( (mmc_parts.size() > 0) && (mmc_parts[0] == "CUCL" ) ) {
	  if( mmc_parts.size() < 2 ) { rt_err( "invalid CUCL magic comment. missing directive after CUCL. saw: " + *i ); }
	  string const & cd = mmc_parts[1];
	  if( (cd == "IN") || (cd == "INOUT") || (cd == "OUT") || (cd == "REF") || (cd == "IX") ) { 
	    if( cd == "IX" ) {
	      if( mmc_parts.size() < 4 ) { rt_err( "invalid CUCL IX decl; missing ix_name and/or arg_name: " + *i ); }
	      string const ix_name = mmc_parts[2];
	      string const arg_name = mmc_parts[3];
	      ix_decls.push_back( ix_decl_t{ ix_name, arg_name } );
	      for( uint32_t i = 4; i != mmc_parts.size(); ++i ) {	
		vect_string const opt_parts = split( mmc_parts[i], '=' );
		if( opt_parts.size() != 2 ) { rt_err( "invalid CUCL IX decl option '"+mmc_parts[i]+"', should have exactly 2 '=' seperated parts" ); }
		else if( opt_parts[0] == "use_dims" ) { ix_decls.back().use_dims = split( opt_parts[1], ':' ); }
		else { rt_err( "invalid CUCL IX decl option '"+opt_parts[0]+"'. known opts: use_dims" ); }
	      }
	    } else {
	      if( mmc_parts.size() < 3 ) { rt_err( "invalid CUCL IN/INOUT/OUT annotation; missing dims spec: " + *i ); }
	      // get var name 
	      vect_string const arg_decl = split_ws( strip_ws( replace_chars_with_char( get_part_before( *i, "//" ), ",);{*/", ' ' ) ) );
	      string const arg_name = arg_decl.back();
	      vect_dims_t ok_dims;
	      for( uint32_t i = 2; i != mmc_parts.size(); ++i ) { ok_dims.push_back( dims_from_nda_spec( mmc_parts[i] ) ); }
	      arg_decls.push_back( arg_decl_t{ arg_name, cd, ok_dims } );
	    }
	  } else { rt_err( "invalid CUCL directive '"+cd+"'. saw:" + *i ); }
	}
      }
    }
    string add_bias_then_maybe_relu( dims_t const & work, uint32_t const & tx, uint32_t const ty ) { 
      string const ve = strprintf( "(out_tile[%s] + filts_strip[%s])", str((ty*work.dsz("out_chan")+tx)).c_str(), str(tx).c_str() );
      return rfs.get_u32_tvv("conv_has_relu") ? ( "max(0.0f,"+ve+")" ) : ve;
    }    
    void gen_op_conv( void ) {
      dims_t const & work = get_arg_dims_by_name( "work" ); // note: usage of t_tile_sz removed in favor of sizes of work.pels/work.out_chan
      // check that we have enough threads per block to load smem using one-elem-per-thread. FIXME: allow for cases when this does not hold.
      assert_st( rf->tpb >= (work.dsz( "out_chan" ) * work.dsz("out_chan_tile") ) ); 
      uint32_t const pel_smem_load_iter = u32_ceil_div( (work.dsz( "pels" ) * work.dsz( "pels_tile" )), rf->tpb );
      tf_exprs.push_back( std::make_pair( "pel_smem_load_iter", str(pel_smem_load_iter) ) );
      tf_exprs.push_back( std::make_pair( "out_chan_tile", 
					  "(%(LOC_ID_1D_out_chan_tile)+%(GRP_ID_1D_out_chan_blk)*%(work_out_chan_tile_dim))"));
      tf_exprs.push_back( std::make_pair( "pel_tile",
					  "(%(LOC_ID_1D_pels_tile)+%(GRP_ID_1D_pels_blk)*%(work_pels_tile_dim))"));
      tf_exprs.push_back( std::make_pair( "out_chan_ix","(%(out_chan_tile)*%(work_out_chan_dim))" ) );
      for( uint32_t i = 0; i != work.dsz( "pels" ); ++i ) {
	insert_nda_ix_exprs( tf_exprs, "pel_ix_" + str(i), must_find(all_ix_dims,"out_pel_ix"),
			     strprintf( "(%%(pel_tile)*%%(work_pels_dim)+%s)", str(i).c_str() ) );
      }
      string const get_in = strprintf( 
	"float v = 0;\n"
	"      int const smem_in_ix_y = %%(out_pel_ix_y)*%%(stride)+%%(filts_ix_out_chan_elem_y) - %%(in_pad);\n"
	"      int const smem_in_ix_x = %%(out_pel_ix_x)*%%(stride)+%%(filts_ix_out_chan_elem_x) - %%(in_pad);\n"
	"      if(smem_in_ix_y >= 0 && smem_in_ix_x >= 0 && \n"
	"          %%(out_pel_ix_img) < %%(in_img_dim) && \n"
	"         smem_in_ix_x < %%(in_x_dim) && smem_in_ix_y < %%(in_y_dim) ) {\n"
	"        v = in[%%(out_pel_ix_img)*%%(in_img_sz) +\n"
	"          %%(filts_ix_out_chan_elem_in_chan)*%%(in_chan_sz) +\n"
	"          smem_in_ix_y*%%(in_y_sz) +\n"
	"          smem_in_ix_x*%%(in_x_sz)];\n" 
	"      }"
				       );
      tf_exprs.push_back( std::make_pair( "get_in", get_in ) );
      for( uint32_t tx = 0; tx != work.dsz( "out_chan" ); ++tx ) {
	cg_add_line( "dummy_loads", strprintf( "filts_strip[%s] = filts_smem[(LOC_ID_1D %%%% 32) + %s];", str(tx).c_str(), str(tx).c_str() ) );
	cg_add_line( "loads", strprintf( "filts_strip[%s] = filts_smem[%%(LOC_ID_1D_out_chan_tile)+%s*%%(work_out_chan_tile_dim)];",
					 str(tx).c_str(), str(tx).c_str() ) );
      }
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) { // note: could merge with above loop, but we want to use ty for consistency
	cg_add_line( "dummy_loads", strprintf( "in_strip[%s] = in_smem[(LOC_ID_1D %%%% 32) + %s];", str(ty).c_str(), str(ty).c_str() ));
	cg_add_line( "loads", strprintf( "in_strip[%s] = in_smem[%%(LOC_ID_1D_pels_tile)*%%(work_pels_dim)+%s];",
					 str(ty).c_str(), str(ty).c_str() ) );
      }
      cg_add_line( "stores", "int32_t tpix[%(work_pels_dim)];");
      cg_add_line( "stores", "int32_t tcix[%(work_out_chan_dim)];");
      // FIXME: should somehow assert that both out_ix and pel_ix_N have the same dims here
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) { 
	cg_add_line( "stores", 
		     strprintf( "tpix[%s] = %%(pel_ix_%s_img)*%%(out_img_sz) + "
				"( %%(pel_ix_%s_x_nomod) %%%% (%%(out_y_dim)*%%(out_x_dim)) ); // cache out pel ixs ", // note: y:x adj-dim opt.
				str(ty).c_str(), str(ty).c_str(), str(ty).c_str() ) );
      }
      for( uint32_t ty = 0; ty != work.dsz( "out_chan" ); ++ty ) { 
	cg_add_line( "stores", strprintf( "  tcix[%s] = (%%(out_chan_ix)+%s)*%%(out_chan_sz); // cache out chan ixs",
					  str(ty).c_str(), str(ty).c_str() ) );
      }	
      cg_add_line( "dummy_stores", "out[0] = 0.0f");
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) {
	cg_add_line( "stores", "if( %(pel_ix_"+str(ty)+"_x_nomod) >= %(pel_ix_0_dims_prod) ) { return; } "
		     "// this pel and the following are off-the-end pels, so don't store them." );
	for( uint32_t tx = 0; tx != work.dsz( "out_chan" ); ++tx ) {
	  cg_add_line( "fmas", strprintf( "out_tile[%s] += filts_strip[%s]*in_strip[%s];", 
					  str((ty*work.dsz( "out_chan" )+tx)).c_str(), str(tx).c_str(), str(ty).c_str() ) );
	  cg_add_line( "stores", strprintf( "if( tcix[%s] < (%%(out_chan_dim)*%%(out_chan_sz)) ) { out[ tpix[%s] + tcix[%s] ] = %s; }",
					    str(tx).c_str(), str(ty).c_str(), str(tx).c_str(), add_bias_then_maybe_relu(work,tx,ty).c_str() ) );
	  cg_add_line( "dummy_stores", " + " + add_bias_then_maybe_relu(work,tx,ty));
	}
      }
      cg_add_line( "dummy_stores", ";");
    }

    void gen_op_s1conv( void ) {
      assert_st( rfs.get_u32_tvv("stride") == 1 );
      rf->has_final_flags_arg = 1;
      dims_t const & work = get_arg_dims_by_name( "work" ); // note: usage of t_tile_sz removed in favor of sizes of work.pels/work.out_chan
      tf_exprs.push_back( std::make_pair( "line_buf_sz", "(%(in_pad)+%(in_x_dim)+%(in_pad))"));
      uint32_t const blk_out_chans = work.dsz("out_chan_tile")*work.dsz("out_chan");
      uint32_t const out_chan_bias_smem_load_iter = u32_ceil_div( blk_out_chans, rf->tpb );
      tf_exprs.push_back( std::make_pair( "out_chan_bias_smem_load_iter", str(out_chan_bias_smem_load_iter) ) );
      dims_t const & filts = get_arg_dims_by_name( "filts" );
      assert_st( filts.dsz("out_chan_reg")*filts.dsz("out_chan_tile") == blk_out_chans ); // also aka %(filts_x_sz)
      dims_t const & in_dims = get_arg_dims_by_name( "in" );
      // generate filter smem loads
      uint32_t const out_chan_smem_load_iter = u32_ceil_div( blk_out_chans * filts.dsz("x"), rf->tpb );    
      if( rf->tpb == blk_out_chans ) {
	assert_st( out_chan_smem_load_iter * rf->tpb == blk_out_chans * filts.dsz("x") );
	tf_exprs.push_back( std::make_pair( "filts_off_adj", "LOC_ID_1D" ));
	for( uint32_t i = 0; i != out_chan_smem_load_iter; ++i ) {
	  cg_add_line( "filts_smem_loads", strprintf( "filts_smem[LOC_ID_1D + %%(tpb) * %s] = filts[filts_off+(%s*%%(filts_x_sz))];",
						      str(i).c_str(), str(i).c_str() ) );
	} 
      } else {
	tf_exprs.push_back( std::make_pair( "filts_off_adj", "0" ));
	for( uint32_t i = 0; i != out_chan_smem_load_iter; ++i ) {
	  string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	  string eif;
	  if( (i+1) == out_chan_smem_load_iter ) { 
	    cg_add_line( "filts_smem_loads", "if( "+ixe+" < "+str(blk_out_chans*filts.dsz("x"))+") { " ); eif = "}"; 
	  }
	  cg_add_line( "filts_smem_loads", strprintf("filts_smem[%s] = filts[filts_off+((%s/%s)*%%(filts_x_sz))"
						     "+(%s %%%% %s)];%s",ixe.c_str(),ixe.c_str(),str(blk_out_chans).c_str(),
						     ixe.c_str(),str(blk_out_chans).c_str(),eif.c_str()) );
	}
      }
      assert_st( in_dims.dsz("x")*work.dsz("line") <= rf->tpb ); // can load in_smem with one-load-per-thread FIXME: too strong?
      assert_st( (2*rfs.get_u32_tvv("in_pad")*work.dsz("line")) <= rf->tpb ); // can zero-init padding with one-store-per-thread FIXME: too strong? other bad things probably happen with large padding?
      tf_exprs.push_back( std::make_pair( "out_chan_tile", 
					  "(%(LOC_ID_1D_out_chan_tile)+%(GRP_ID_1D_out_chan_blk)*%(work_out_chan_tile_dim))"));
      tf_exprs.push_back( std::make_pair( "out_chan_ix","(%(out_chan_tile)*%(work_out_chan_dim))" ) );
      
      for( uint32_t i = 0; i != work.dsz("pels"); ++i ) {
	tf_exprs.push_back( std::make_pair( "line_x_" + str(i), 
					    strprintf( "(%%(LOC_ID_1D_line_x_tile)*%%(work_pel_dim)+%s)", str(i).c_str() ) ) );
      }
      for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	cg_add_line( "filt_loads", strprintf( "filts_strip[%s] = filts_smem[filts_smem_off+%%(LOC_ID_1D_out_chan_tile)+%s*%%(work_out_chan_tile_dim)];", str(tx).c_str(), str(tx).c_str() ) );
      }
      for( uint32_t ty = 0; ty != work.dsz("pels") + filts.dsz("x") - 1; ++ty ) { 
	cg_add_line( "in_loads", strprintf( "in_strip[%s] = in_smem[%%(line_buf_sz)*%%(LOC_ID_1D_line)+"
					    " %%(work_pels_dim)*%%(LOC_ID_1D_line_x_tile)+%s];",
					    str(ty).c_str(), str(ty).c_str() ) );
      }
      cg_add_line( "stores", "  int32_t tpix[%(work_pels_dim)];" );
      cg_add_line( "stores", "  int32_t tcix[%(work_out_chan_dim)];" );
      cg_add_line( "stores", "  if( %(out_line_img) >= %(out_img_dim) ) { return; } " );
      // FIXME: should somehow assert that both out_ix and pel_ix_N have the same dims here
      for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) { 
	cg_add_line( "stores", strprintf( "tpix[%s] = %%(out_line_img)*%%(out_img_sz) + \n"
					  "           %%(out_line_y)*%%(out_y_sz) + \n"
					  " (%%(work_pels_dim)*%%(LOC_ID_1D_line_x_tile)+%s)*%%(out_x_sz); // cache out pel ixs",
					  str(ty).c_str(), str(ty).c_str() ) );
      }
      for( uint32_t ty = 0; ty != work.dsz("out_chan"); ++ty ) { 
	cg_add_line( "stores", strprintf( "tcix[%s] = (%%(out_chan_ix)+%s)*%%(out_chan_sz); // cache out chan ixs",
					  str(ty).c_str(), str(ty).c_str() ) );
      }
      for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	cg_add_line( "stores", "  if( (%(work_pels_dim)*%(LOC_ID_1D_line_x_tile)+"+str(ty)+") >= %(out_x_dim) ) { return; } "
		     "// this pel and the following are off-the-end pels, so don't store them." );
	for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	  cg_add_line( "stores", strprintf( "if( tcix[%s] < (%%(out_chan_dim)*%%(out_chan_sz)) ) { out[ tpix[%s] + tcix[%s] ] = %s; }",
					    str(tx).c_str(), str(ty).c_str(), str(tx).c_str(), add_bias_then_maybe_relu(work,tx,ty).c_str() ) );
	}
      }
      cg_add_line( "inner_loop_body", "filts_smem_off = 0;" );
      cg_add_line( "inner_loop_body", cgs["in_loads"] + ";" );
      for( uint32_t kx = 0; kx != filts.dsz("x"); ++kx ) {
	cg_add_line( "inner_loop_body", cgs["filt_loads"] + ";" );
	cg_add_line( "inner_loop_body", "    filts_smem_off += %(filts_x_sz);" );
	for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	  for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	    cg_add_line( "inner_loop_body", strprintf( "    out_tile[%s] += filts_strip[%s]*in_strip[%s];", 
						       str((ty*work.dsz("out_chan")+tx)).c_str(), str(tx).c_str(), str(ty+kx).c_str() ) );
	  }
	}
      }
    }

    void gen_op_k1conv( void ) {
      assert_st( rfs.get_u32_tvv("in_pad") == 0 );
      assert_st( rfs.get_u32_tvv("stride") == 1 );
      dims_t const & work = get_arg_dims_by_name( "work" ); // note: usage of t_tile_sz removed in favor of sizes of work.pels/work.out_chan
      dims_t const & filts = get_arg_dims_by_name( "filts" );
      assert_st( filts.dsz("x") == 1 ); assert_st( filts.dsz("y") == 1 );
      dims_t const & in = get_arg_dims_by_name( "in" );
      dims_t const & out = get_arg_dims_by_name( "out" );
      // calculate needed smem sizes (and total kernel needed smem size)
      // note: filts and in smem are used concurrently, then just all of all_smem as an output buffer
      uint32_t const filts_smem_sz = filts.dstride("in_chan")*in.dsz("blk_iter_chan");
      tf_exprs.push_back( std::make_pair( "filts_smem_sz", str(filts_smem_sz) ));
      uint32_t const out_smem_sz = work.dsz("pels_tile")*work.dsz("out_chan_tile")*work.dsz("pels"); // note: == oi->tpb*t_tile_sz
      tf_exprs.push_back( std::make_pair( "out_smem_sz", str(out_smem_sz) )); // note: unused, but assumed that all_smem_sz >= out_smem_sz
      uint32_t const all_smem_sz = std::max( out_smem_sz, filts_smem_sz+in.dstride("blk_iter") ); // note: %(in_blk_iter_sz) == in_smem_sz
      tf_exprs.push_back( std::make_pair( "all_smem_sz", str(all_smem_sz) ));

      uint32_t const out_chan_bias_smem_load_iter = u32_ceil_div( filts.dstride("x"), rf->tpb );
      tf_exprs.push_back( std::make_pair( "out_chan_bias_smem_load_iter", str(out_chan_bias_smem_load_iter) ) );

      // generate smem loads
      uint32_t const out_chan_smem_load_iter = u32_ceil_div( filts_smem_sz, rf->tpb );    
      for( uint32_t i = 0; i != out_chan_smem_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string eif;
	if( (i+1)*rf->tpb > filts_smem_sz ) { cg_add_line( "smem_loads", "if( "+ixe+" < %(filts_smem_sz) ) {" );eif = "}";}
	// note: load is (always) contiguous
	cg_add_line( "smem_loads", strprintf("filts_smem[%s] = filts[filts_off+(%%(tpb)*%s)];%s",ixe.c_str(),str(i).c_str(),eif.c_str()) );
      }
      uint32_t const in_smem_load_iter = u32_ceil_div( in.dstride("blk_iter"), rf->tpb );    
      for( uint32_t i = 0; i != in_smem_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string eif;
	if( (i+1)*rf->tpb > in.dstride("blk_iter") ) { cg_add_line( "smem_loads", "if( "+ixe+" < %(in_blk_iter_sz)) { ");eif = "}";}
	cg_add_line( "smem_loads", strprintf("    in_smem[%s] = in[ blk_in_ix_base + (%%(tpb)*%s) ];%s\n",
					     ixe.c_str(),str(i).c_str(),eif.c_str()) );
      }
      tf_exprs.push_back( std::make_pair( "out_chan_tile", "(%(GRP_ID_1D_out_chan_blk)*%(work_out_chan_tile_dim)+%(LOC_ID_1D_out_chan_tile))"));
      tf_exprs.push_back( std::make_pair( "out_chan_ix","(%(out_chan_tile)*%(work_out_chan_dim))" ) );

      // cg_add_line( "stores", "  if( %(out_line_img) >= %(out_ix_img_dim) ) { return; } "; // not possible due to no-partial-imgs-per-block
      // FIXME: should somehow assert that both out_ix and pel_ix_N have the same dims here
      // FIXME: out_pel must be per-tpix (again)
      if( out.get_dim_by_name("blk") ) {
	// padded # of in chans of next layer  == out.dsz("blk_iter")*out.dsz("blk_iter_chan")
	// padded # of out chans of this layer == work.dsz("out_chan_blk")*work.dsz("out_chan_tile")*work.dsz("out_chan")
	// if these are ==, we don't have to worry about bounds-checking our writes to out in the chan dim
	assert_st( work.dsz("out_chan_blk")*work.dsz("out_chan_tile")*work.dsz("out_chan") == out.dsz("blk_iter")*out.dsz("blk_iter_chan") );
	// padded # of in pels of next layer:  == out.dsz("blk")*out.dsz("blk_pel")
	// padded # of out pels of this layer: == work.dsz("pels_blk")*work.dsz("pels_tile")*work.dsz("pels")
	// if these are ==, we don't have to worry about bounds-checking our writes to out in the pel dim
	assert_st( work.dsz("pels_blk")*work.dsz("pels_tile")*work.dsz("pels") == out.dsz("blk")*out.dsz("blk_pel") );

	// we assume out_blk_pel_dim (== noi->tix_pels_tile_sz*t_tile_sz) is divisible by t_tile_sz. but let's check it explicitly:
	// FIXME_WXP: i don't see where we assume this, and hence i dunno what t_tile_sz refers to below. poop. assert is removed for now:
	// assert_st( (out.dsz("blk_pel") % t_tile_sz) == 0 );
	// we assume the out chans are a single (span of) dims in out. FIXME: check this?. FIXME_WXP: what does this even mean?

	//cg_add_line( "stores", "  int32_t xpbuf[%(t_tile_sz)];\n";
	// FIXME: assumes (for GRP_ID_1D_pels_blk*... term) that input and output block have same # of pels ... too strong?
	assert_st( out.dsz("blk_pel") == in.dsz("blk_pel") );
	cg_add_line( "stores", "int32_t const out_ix = (%(GRP_ID_1D_out_chan_blk)*%(work_out_chan_tile_dim)*%(work_out_chan_dim))*%(out_blk_iter_chan_sz) + %(GRP_ID_1D_pels_blk)*%(out_blk_sz);" ); 
	cg_add_line( "stores", "int32_t xpbuf_rd_pel;" );
	cg_add_line( "stores", "int32_t xpbuf_rd_chan;" );

	for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	  // transpose each thread's tx'th out_chan (= t_tile_sz out chans across all threads) into xpbuf (again across all threads)
	  // such that we can do (mostly) sequential writes to global memory for this set of t_tile_sz out chans
	  cg_add_line( "stores", "  BARRIER_SYNC;" );
	  for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) { // out_tile[] (registers) -> all_smem[]
	    cg_add_line( "stores", strprintf( "out_smem_off[%%(tpb)*%s] = %s;", str(ty).c_str(), add_bias_then_maybe_relu(work,tx,ty).c_str() ) );
	  }
	  cg_add_line( "stores", "  BARRIER_SYNC;" );
	  for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) { // all_smem[] -> [xpbuf[] (registers)] -> out[] (global)
	    // here, we reshape the threads so that the total threads across iterations (%(tbp)*work.dsz("pels")) covers
	    // the space of the data in out_smem as a (simple) 2D array ordered as chan:pel. thus, for each thread, we
	    // have a single chan and pel index that we must read from smem and write to global memory. this mapping is
	    // such that the writes to global memory are somewhat sequential (with jumps at chan boundaries). however,
	    // for the reads from smem we just calculate the correct index and hope for the best. note that the actual
	    // output chan indexes read/written to here are strided by %(work_out_chan_dim) and offset by tx.
	    string const obe = "(LOC_ID_1D + %(tpb)*"+str(ty)+")";
	    cg_add_line( "stores", "  xpbuf_rd_pel = "+obe+" %% %(out_blk_pel_dim) ;" );
	    cg_add_line( "stores", "  xpbuf_rd_chan = "+obe+" / %(out_blk_pel_dim) ;" );
	    cg_add_line( "stores", strprintf( "out[out_ix + xpbuf_rd_pel + (xpbuf_rd_chan*%%(work_out_chan_dim)+%s)*%%(out_blk_iter_chan_sz)] = "
					      "all_smem[xpbuf_rd_chan+(xpbuf_rd_pel %%%% %%(work_pels_dim))*%%(tpb)"
					      "+ (xpbuf_rd_pel / %%(work_pels_dim))*%%(work_out_chan_tile_dim) ];",
					      str(tx).c_str() ) );
	  }
	  for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) { // xpbuf[] registers -> out[] (global)
	    // TODO/UNUSED?
	  }	
	}
      } else {
	cg_add_line( "stores", "  int32_t tpix[%(work_pels_dim)];" );
	cg_add_line( "stores", "  int32_t tcix[%(work_out_chan_dim)];" );
	for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) { 
	  insert_nda_ix_exprs( tf_exprs, "out_pel_" + str(ty), must_find(all_ix_dims,"out_ref_pel"),
			       "( (%(GRP_ID_1D_pels_blk)*%(work_pels_tile_dim) + %(LOC_ID_1D_pels_tile))*%(work_pels_dim) + "+str(ty)+" )" );
	  cg_add_line( "stores", strprintf( "  tpix[%s] = %%(out_pel_%s_img)*%%(out_img_sz) + "
					    " %%(out_pel_%s_x)*%%(out_x_sz) + %%(out_pel_%s_y)*%%(out_y_sz) " // FIXME_WXP:restore: y:x adj-dim opt?
					    "  ; // cache out pel ixs",
					    str(ty).c_str(), str(ty).c_str(), str(ty).c_str(), str(ty).c_str() ) );
	}
	for( uint32_t ty = 0; ty != work.dsz("out_chan"); ++ty ) { 
	  cg_add_line( "stores", strprintf( "  tcix[%s] = (%%(out_chan_ix)+%s)*%%(out_chan_sz); // cache out chan ixs",
					    str(ty).c_str(), str(ty).c_str() ) );
	}
	for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	  cg_add_line( "stores", "  if( %(out_pel_"+str(ty)+"_img) >= %(out_img_dim) ) { return; } "
		       "// this pel and the following are off-the-end pels, so don't store them." );
	  for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	    cg_add_line( "stores", strprintf( "if( tcix[%s] < (%%(out_chan_dim)*%%(out_chan_sz)) ) { out[ tpix[%s] + tcix[%s] ] = %s; }",
					      str(tx).c_str(), str(ty).c_str(), str(tx).c_str(), add_bias_then_maybe_relu(work,tx,ty).c_str() ) );
	  }
	}
      }
      for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	  cg_add_line( "dummy_stores", strprintf( "out_off[%s] = %s;", 
						  str((ty*work.dsz("out_chan")+tx)*rf->tpb).c_str(), add_bias_then_maybe_relu(work,tx,ty).c_str() ) );
	}
      }
      for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	cg_add_line( "bias_loads", strprintf( "filts_strip[%s] = filts_smem_off[%s*%%(work_out_chan_tile_dim)];", 
					      str(tx).c_str(), str(tx).c_str() ) );
      }
      assert_st( in.dsz("blk_pel") == work.dsz("pels_tile")*work.dsz("pels") ); // by input xform design
      for( uint32_t ict = 0; ict != in.dsz("blk_iter_chan"); ++ict ) {
	for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	  cg_add_line( "inner_loop_body", strprintf( "filts_strip[%s] = filts_smem_off[(%s*%%(filts_in_chan_sz))+%s*%%(work_out_chan_tile_dim)];", 
						     str(tx).c_str(), str(ict).c_str(), str(tx).c_str() ) );
	}
	for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) { 
	  cg_add_line( "inner_loop_body", strprintf( "in_strip[%s] = in_smem_off[(%s*%%(in_blk_pel_dim)+%s)];",
						     str(ty).c_str(), str(ict).c_str(), str(ty).c_str() ) );
	}
	for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	  for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	    cg_add_line( "inner_loop_body", strprintf( "out_tile[%s] += filts_strip[%s]*in_strip[%s];", 
						       str((ty*work.dsz("out_chan")+tx)).c_str(), str(tx).c_str(), str(ty).c_str() ) );
	  }
	}
      }
      rf->has_final_flags_arg = 1;
    }

    void gen_op_tconv( void ) {
      dims_t const & work = get_arg_dims_by_name( "work" ); // note: usage of t_tile_sz removed in favor of sizes of work.pels/work.out_chan
      dims_t const & filts = get_arg_dims_by_name( "filts" );
      dims_t const & in = get_arg_dims_by_name( "in" );
      //dims_t const & out = get_arg_dims_by_name( "out" );
      // uint32_t const in_ix_blk_x_dim = oi->out_to_in(t_tile_sz); --> in_blk_x_dim / in.dsz("blk_x")
      // uint32_t const blk_filt_ix_sz = work_out_chan_tile_dim * t_tile_sz; --> filts_x_sz / filts.dsz("x")
      uint32_t const out_chan_smem_load_iter = u32_ceil_div( filts.dstride("y"), rf->tpb );  // filt smem loads
      for( uint32_t i = 0; i != out_chan_smem_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string eif; // note: load is (always) contiguous
	if( (i+1)*rf->tpb > filts.dstride("y") ) { cg_add_line( "filt_smem_loads", "if( "+ixe+" < %(filts_y_sz) ) { " );eif = "}";}
	cg_add_line( "filt_smem_loads", strprintf("filts_smem[%s] = filts[filts_off+(%%(tpb)*%s)];%s",ixe.c_str(),str(i).c_str(),eif.c_str()));
      }
      cg_add_line( "filt_smem_loads", "filts_off += %(filts_y_sz);" );

      uint32_t const in_smem_load_iter = u32_ceil_div( in.dstride("blk_in_chan"), rf->tpb );  // in smem loads
      for( uint32_t i = 0; i != in_smem_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string eif;
	if( (i+1)*rf->tpb > in.dstride("blk_in_chan") ) { cg_add_line( "in_smem_loads", "if( "+ixe+" < %(in_blk_in_chan_sz)) { " );eif = "}";}
	cg_add_line( "in_smem_loads", strprintf("in_smem[%s] = in[ blk_in_ix_base + (%%(tpb)*%s) ];%s",	ixe.c_str(),str(i).c_str(),eif.c_str()));
      }
      cg_add_line( "in_smem_loads", "blk_in_ix_base += %(in_blk_in_chan_sz);" );

      for( uint32_t i = 0; i != in.dsz("blk_x"); ++i ) {
	cg_add_line( "inner_loop_body", strprintf( "in_strip[%s] = in_smem_off[%s];", str(i).c_str(), str(i).c_str() ) );
      }
      assert_st( work.dsz("out_chan_tile") == filts.dsz("out_chan_tile") ); // also == %(filts_out_chan_reg_sz)
      for( uint32_t kx = 0; kx != filts.dsz("x"); ++kx ) {
	for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	  cg_add_line( "inner_loop_body", strprintf( "filts_strip[%s] = filts_smem_off[%s*%%(filts_x_sz)+%s*%%(filts_out_chan_reg_sz)];", 
						     str(tx).c_str(), str(kx).c_str(), str(tx).c_str() ) );
	}
	for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	  for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	    cg_add_line( "inner_loop_body", strprintf( "out_tile[%s] += filts_strip[%s]*in_strip[%s];", str((ty*work.dsz("out_chan")+tx)).c_str(), 
						       str(tx).c_str(), str(ty*rfs.get_u32_tvv("stride")+kx).c_str() ) );
	  }
	}
      }
      uint32_t const out_chan_bias_smem_load_iter = u32_ceil_div( filts.dsz("x"), rf->tpb );
      tf_exprs.push_back( std::make_pair( "out_chan_bias_smem_load_iter", str(out_chan_bias_smem_load_iter) ) );
      for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	cg_add_line( "bias_loads", strprintf( "filts_strip[%s] = filts_smem_off[%s*%%(filts_out_chan_reg_sz)];", str(tx).c_str(), str(tx).c_str() ) );
      }
      //cg_add_line( "stores", "  if( %(out_line_y) >= %(out_ix_y_sz) ) { return; }" ); // not possible
      cg_add_line( "stores", "if( %(out_line_img) >= %(out_img_dim) ) { return; }" );
      cg_add_line( "stores", "int32_t out_x = %(GRP_ID_1D_blk_bx)*%(work_pels_dim);" );
      cg_add_line( "stores", "int32_t out_chan = (%(GRP_ID_1D_out_chan_blk)*%(work_out_chan_tile_dim) + %(LOC_ID_1D_out_chan_tile))*%(work_out_chan_dim);" );
      cg_add_line( "stores", "GASQ float * out_off = out + %(out_line_img)*%(out_img_sz) + out_chan*%(out_chan_sz) + "
		   "%(out_line_y)*%(out_y_sz) + out_x*%(out_x_sz) ;" );

      for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	cg_add_line( "stores", "if( (out_x + "+str(ty)+") >= %(out_x_dim) ) { return; } "
		     "// this x value and the following are off-the-end pels, so don't store them." );
	for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
#if 1
	  string const ve = add_bias_then_maybe_relu(work,tx,ty);

#else
	  string const ve = strprintf( "(filts_strip[%s])", str(tx).c_str() );
#endif
	  cg_add_line( "stores", strprintf( "if( (out_chan + %s) < %%(out_chan_dim) ) { "
						   "out_off[ %s*%%(out_chan_sz) + %s*%%(out_x_sz) ] = %s; }",
						   str(tx).c_str(), str(tx).c_str(), str(ty).c_str(), ve.c_str() ) );
	}
      }
      rf->has_final_flags_arg = 1;
    }
  };

  // running test case for gen_call():
  // boda test_compute --model-name=nin_imagenet --wins-per-image=1 --imgs='(pil_fn=%(boda_test_dir)/pascal/head_1/%%s.txt)' --run-cnet='(in_sz=227 227,in_num_imgs=20,ptt_fn=%(models_dir)/%(model_name)/train_val.prototxt,trained_fn=%(models_dir)/%(model_name)/best.caffemodel,out_node_name=pool4_grad_loss,add_bck_ops=1)' --cf2="(mode=rtc,per_call_fn=out.py)" --max-err=1 && cat test_compute.txt 

  string conv_pipe_fwd_t::gen_func( rtc_func_sig_t const & rfs ) {
    string & gen_fn = rtc_func_sigs_map[rfs];
    if( gen_fn.empty() ) { // need to instatiate function and pick unused name
      gen_fn = rfs.gen_unused_fn( used_rtc_func_names );
      rtc_call_gen_t().init( rfs, gen_fn, rtc_funcs, rtc_prog_str );
    }    
    return gen_fn;
  }
  void conv_pipe_fwd_t::gen_call( string const & fn, p_op_info_t const & oi, vect_string const & args, 
				  vect_dims_t const & ref_dims, bool const is_init_call ) { 
    // note: we generally assume all strides are 0 (uncalculated), and assume no (non-explicit) padding. it's unclear if
    // this is the best idea. note: we assume that all arg dims are already availible
    rtc_func_sig_t rfs;
    rfs.fn = fn;
    rfs.template_var_values = oi->template_var_values;
    for( vect_string::const_iterator i = args.begin(); i != args.end(); ++i ) { rfs.args.push_back( rtc->get_var_dims_floats( *i ) ); }
    rfs.args.insert( rfs.args.end(), ref_dims.begin(), ref_dims.end() );
    // note: we assume the generated function only work for exactly these input/output sizes. if not, we'd set some dims to 0/wild
    string const & gen_fn = gen_func( rfs );
    (is_init_call ? init_calls : fwd_calls).push_back( rtc_func_call_t{ gen_fn, args, {}, {}, {}, oi->cop->tag } );
  }

  // gen_node_var() creates a var directly corresponding to a pipe node.  usually, but not always, name == node_node; in
  // that case the var is directly mirroring a pipe node
  void conv_pipe_fwd_t::gen_node_var( string const & name, string const & node_name ) { 
    rtc->create_var_with_dims_floats( name, cp->must_get_node(node_name)->cio.dims(num_imgs) );
  }

  // quantize command line example:
  // export QOPTS="keep_bits=8,quantize=(_=(name=conv1,max_val=4096),_=(name=conv2,max_val=1024),_=(name=conv3,max_val=1024),_=(name=conv4,max_val=512),_=(name=conv5,max_val=512))

  // CUDA_VISIBLE_DEVICES=0 DISABLE_CUDNN=0 time boda test_lmdb --model-name=alexnet_ng_conv --num-to-read=1000 --run-cnet="(in_sz=227 227,in_num_imgs=20,ptt_fn=%(models_dir)/%(model_name)/train_val.prototxt,trained_fn=%(models_dir)/%(model_name)/best.caffemodel,out_node_name=fc8-conv,compute_mode=1,conv_fwd=(mode=rtc,enable_stats=0,show_rtc_calls=0,${QOPTS}))"

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

  void conv_pipe_fwd_t::init( p_conv_pipe_t const & cp_, uint32_t const & num_imgs_ ) {
    num_imgs = num_imgs_;
    assert_st( num_imgs );
    cp = cp_;
    assert_st( cp );
    op_infos.reset( new map_str_p_op_info_t );
    for( map_str_p_conv_op_t::iterator i = cp->convs->begin(); i != cp->convs->end(); ++i ) { 
      p_op_info_t & oi = (*op_infos)[i->first];
      assert_st( !oi );
      oi = make_shared< op_info_t >();
      oi->init( cp, i->second, num_imgs, enable_k1conv, enable_s1conv, enable_tconv, force_enable_tconv, t_tile_sz );
    }

    rtc->init();

    for( vect_string::const_iterator i = def.begin(); i != def.end(); ++i ) { rtc_prog_str += "#define "+*i+" 1\n"; }
    cp->topo_visit_setup();
    for( set_string::const_iterator i = cp->bots.begin(); i != cp->bots.end(); ++i ) { gen_ops_rec( *i ); }

    rtc->compile( rtc_prog_str, show_compile_log, enable_lineinfo );
    for( rtc_funcs_t::iterator i = rtc_funcs.begin(); i != rtc_funcs.end(); ++i ) { rtc->check_runnable( i->first, show_func_attrs ); }

    rtc->copy_ndas_to_vars( op_param_names, *cp->op_params ); // copy op_params in (FIXME/note: implicit  on names)
    for( set_string::const_iterator i = force_zero_names.begin(); i != force_zero_names.end(); ++i ) { rtc->set_var_to_zero( *i ); }

    // transpose filters ... and do any other init-time work added after this comment was written ;)
    for( vect_rtc_func_call_t::iterator i = init_calls.begin(); i != init_calls.end(); ++i ) { run_rfc( *i ); }
    rtc->finish_and_sync();
  }

  void conv_pipe_fwd_t::run_rfc( rtc_func_call_t & rfc ) {
    rtc_func_t const & rf = must_find( rtc_funcs, rfc.rtc_func_name );
    uint32_t blks = rf.blks; // if non-zero, blks is static, and we can check arg sizes
    //printf( "rf.name=%s rf.arg_sizes=%s rfc.args.size()=%s\n", str(rf.name).c_str(), str(rf.arg_sizes).c_str(), str(rfc.args.size()).c_str() );

    // FIXME: arg size checking bot broken/removed during the
    // ocl/nvrtc split/factoring out. oops. also, rf.arg_sizes (or
    // whatever it becomes) probably needs to be split into
    // in/inout/out parts or something.
    if( blks ) { assert( rf.arg_sizes.size() == (rfc.in_args.size()+rfc.inout_args.size()+rfc.out_args.size()) ); }
    // FIXME: check that we're passing the correct # of args here somehow.
    if( !blks ) { // handle dynamic # of blks case
      // FIXME: pretty limited / special cased here
      assert_st( rfc.u32_args.size() > 0 );
      blks = u32_ceil_div( rfc.u32_args[0], rf.tpb );
    }
    if( show_rtc_calls ) { 
      printf( "%s( in{%s} inout{%s} out{%s} -- u32{%s} ) tpb=%s blks=%s\n", str(rfc.rtc_func_name).c_str(), 
	      str(rfc.in_args).c_str(), str(rfc.inout_args).c_str(), str(rfc.out_args).c_str(), str(rfc.u32_args).c_str(),
	      str(rf.tpb).c_str(), str(blks).c_str() );
    }
    rfc.tpb.v = rf.tpb;
    rfc.blks.v = blks;
    if( rf.has_final_flags_arg ) { rfc.u32_args.push_back( flags ); }
    rtc->run( rfc );
    if( rf.has_final_flags_arg ) { rfc.u32_args.pop_back(); }
  }

  void conv_pipe_fwd_t::run_fwd( p_map_str_p_nda_float_t const & fwd, vect_string const & to_get_vns ) {
    if( enable_double_run ) {
      // optional: run fwd rfc's one for testing/flushing/cache setup. note: ~*doubles* total run time ...
      for( vect_rtc_func_call_t::iterator i = fwd_calls.begin(); i != fwd_calls.end(); ++i ) { run_rfc( *i ); }
    }
    timer_t t("conv_pipe_fwd_t::run_fwd");
    if( enable_prof ) { rtc->profile_start(); }
    //printf("run_fwd() begin\n");
    rtc->copy_ndas_to_vars( vect_string{cp->bots.begin(),cp->bots.end()}, *fwd ); // copy sources in. FIXME/note: implicit  inside
    //printf("run_fwd() exec\n");
    for( vect_rtc_func_call_t::iterator i = fwd_calls.begin(); i != fwd_calls.end(); ++i ) { run_rfc( *i ); }
    rtc->finish_and_sync();
    float const compute_dur = fwd_calls.empty() ? 0.0f : rtc->get_dur( fwd_calls.front(), fwd_calls.back() );
    if( enable_prof ) { rtc->profile_stop(); }
    if( !per_call_fn.empty() ) {
      p_ofstream out = ofs_open( per_call_fn );
      (*out) << strprintf("net.args.num_imgs=%s\n", str(num_imgs).c_str() );
      (*out) << strprintf("num_img=%s\n", str(num_imgs).c_str() ); // FIXME: dup'd in flops.py, need to set both here ...
      (*out) << strprintf("net.args.runtime=%s\n", str(compute_dur/1000.0).c_str() );
      for( vect_rtc_func_call_t::iterator i = fwd_calls.begin(); i != fwd_calls.end(); ++i ) {
	rtc_func_call_t & rfc = *i;
	if( rfc.call_tag.empty() ) { continue; }
	float const rfc_dur = rtc->get_dur( rfc, rfc );
	(*out) << strprintf( "per_layer_time['%s']=per_layer_time.get('%s',0.0) + %s # %s \n", 
			     str(rfc.call_tag).c_str(), str(rfc.call_tag).c_str(), str(rfc_dur/1000.0).c_str(), rfc.rtc_func_name.c_str() );
      }
      cp->dump_ops( *out, 0 );
    }
    //printf("run_fwd() copy out\n");
    rtc->copy_vars_to_ndas( to_get_vns, *fwd ); // copy requested vars out
    update_stats();
    rtc->release_per_call_id_data();
    //printf("run_fwd() done\n");
  }
  
#include"gen/rtc_fwd.cc.nesi_gen.cc"
}
