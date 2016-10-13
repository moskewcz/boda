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
#include"cnn_op.H"

namespace boda 
{
  struct rtc_fwd_func_call_t {
    p_rcg_func_call_t rfc;
    string call_tag;
    uint32_t call_id;
    rtc_fwd_func_call_t( p_rcg_func_call_t const &rfc_, string const & call_tag_ ) : 
      rfc(rfc_), call_tag(call_tag_), call_id( uint32_t_const_max ) {}
  };
  typedef vector< rtc_fwd_func_call_t > vect_rtc_fwd_func_call_t; 

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

  typedef shared_ptr< dims_t > p_dims_t; 

  typedef set< op_base_t > set_op_base_t;

  struct conv_pipe_fwd_t : virtual public nesi, public has_conv_fwd_t // NESI(help="compute conv pipe forward using rtc",
			   // bases=["has_conv_fwd_t"], type_id="rtc" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    rtc_compile_opts_t compile_opts; // NESI(default="()",help="runtime compilation options")
    uint32_t enable_stats; //NESI(default=0,help="if 1, dump stats")
    uint32_t enable_prof; //NESI(default=1,help="if 1, enable profiling")
    uint32_t enable_double_run; //NESI(default=0,help="if 1, run ops an extra time before the timed run (doubles run time, might improve timing quality/repeatability).")
    string per_call_fn; //NESI(default="",help="if non-empty, write per-call profiling (timing via events) to given file.")
    vect_p_quantize_ops_t quantize; //NESI(help="per-layer quantize options")

    op_tune_t op_tune; //NESI(default="()",help="tuning parameters / options")

    uint32_t enable_bconv; //NESI(default=0,help="if 1, enable bconv")
    uint32_t enable_write_xpose; //NESI(default=0,help="if 1, enable experimental k1conv write xposing")
    uint32_t force_zero_bias; //NESI(default=0,help="if 1, force biases to zero")
    uint32_t flags; //NESI(default=0,help="dynamic flags to pass to kernels that request them (often to trick compiler)")

    vect_string dump_vars; // NESI(help="dump out values of these vars after forward")

    filename_t rtc_func_sigs_fn; //NESI(default="rtc_func_sigs.txt",help="file to hold all generated func signatures")
    uint32_t write_op_sigs; //NESI(default=0,help="if 1, write op sigs to op_sigs_fn")
    filename_t op_sigs_fn; //NESI(default="op_sigs_full.txt",help="file to hold unique op signatures")
    set_op_base_t all_op_sigs;
    p_dims_t dummy_dims; // NESI(help="HACK: dummy NESI var of type dims_t (otherwise unused) to force tinfo generation. see map_str_T FIXME in nesi.cc")

    p_conv_pipe_t cp;
    p_map_str_p_conv_op_t op_infos;

    vect_string op_param_names;
    set_string filts_names;
    set_string inxp_names;
    set_string force_zero_names;

    vect_string stats_names;
    map_str_float_t stats_map;

    p_rtc_compute_t rtc; //NESI(help="rtc back-end to use")

    vect_rtc_fwd_func_call_t fwd_calls;
    void add_fwd_call( p_rcg_func_call_t const & rcg, string const & call_tag ) { 
      fwd_calls.emplace_back( rcg, call_tag ); 
    }

    virtual void init( p_conv_pipe_t const & cp_, nesi_init_arg_t * const nia );
    virtual void run_fwd( vect_string const & to_set_vns, p_map_str_p_nda_float_t const & fwd, vect_string const & to_get_vns );
    vect_uint32_t dropout_cixs;
    virtual void set_det_drop_seed( uint32_t const & det_drop_seed_ ) { 
      // sigh.
      for( vect_uint32_t::const_iterator i = dropout_cixs.begin(); i != dropout_cixs.end(); ++i ) {
	assert( (*i) < fwd_calls.size() );
	rcg_func_call_t & rfc = *(fwd_calls[*i].rfc);
	assert_st( has( rfc.arg_map, "det_drop_seed" ) );
	rfc.arg_map["det_drop_seed"] = make_scalar_nda(det_drop_seed_);
      }
    }

    void update_stats( void );
    string dump_var( string const & n );
    virtual string get_info_log( void );
  protected:
    vect_string gen_op_stats( string const & top_in );
    void gen_op_quantize( string const & top_in, uint32_t const & max_val, uint32_t const & keep_bits );

    rtc_codegen_t codegen;
    void gen_call( p_conv_op_t const & oi );
    void gen_call( string const & fn, p_conv_op_t const & oi );
    string gen_apply_func_to_var( string const & in_an, string const & in_var,
                                  string const & ret_an, dims_t const & ret_dims, 
                                  string const & func_name, p_conv_op_t const & oi );
    void gen_node_var( string const & name, string const & node_name );
    void gen_op( p_conv_op_t const & cop );
    void gen_ops_rec( string const & node_name );
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
      p_nda_float_t nda = make_shared< nda_float_t >( rtc->copy_var_as_flat_nda( *i ) );
      assert_st( nda->elems_sz() == 1 );
      float v = *nda->elems_ptr();
      if( has( stats_map, *i ) ) { v = stats_reduce( *i, v, stats_map[*i] ); }
      stats_map[*i] = v;
    }
  }

  string conv_pipe_fwd_t::dump_var( string const & n ) {
    string ret;
    p_nda_float_t nda = make_shared< nda_float_t >( rtc->create_nda_from_var( n ) );
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
    uint32_t in_sz = rtc->get_var_dims( top_in ).dims_prod(); // treat input as flat
    uint32_t primary_in = 1;
    assert_st( in_sz );
    string const top_in_reshape = top_in + "_flat_reshape";
    rtc->create_var_with_dims_as_reshaped_view_of_var( top_in_reshape, dims_t{ {in_sz}, {"v"}, "float" }, top_in );
    // the var_stats template doesn't specify tpb, so we assume this will be used. we should check this for any gen'd func.
    dims_t arg_dims( {0}, {"v"}, "float" ); // all vars are single-dim with wild/any size
    vect_string cur_ins;
    for( uint32_t i = 0; i != reds.size(); ++i ) { cur_ins.push_back( top_in_reshape ); } // initial inputs are all top_in 
    while( in_sz > 1 ) {
      uint32_t const out_sz = u32_ceil_div( in_sz, rtc_call_geom_t::get_default_tpb() ); 
      op_base_t var_stats;
      var_stats.set_func_name( "var_stats" );
      var_stats.set_dims( "primary_in", make_scalar_dims_t("uint32_t") ); // bool; 0/1 indicates first-level
      for( uint32_t i = 0; i != reds.size(); ++i ) {  // input dims (const); initial inputs
        var_stats.set_dims( reds[i] + "_in",  dims_t( {in_sz}, {"v"}, "float" ) );
        var_stats.set_dims( reds[i] + "_out", dims_t( {out_sz}, {"v"}, "float" ) ); 
      } 
      var_stats.set_u32( "tpb", rtc_call_geom_t::get_default_tpb() );
      vect_string cur_outs;
      //vect_string args = cur_ins;
      vect_string out_args;
      for( uint32_t i = 0; i != reds.size(); ++i ) { 
	string cur_out = top_in + "_" + reds[i] + "_out_sz_" + str(out_sz);
	rtc->create_var_with_dims( cur_out, dims_t{ {out_sz}, {"v"}, "float" } );
	cur_outs.push_back( cur_out );
	//args.push_back( cur_out );
      }
      map_str_rtc_arg_t arg_map;
      assert_st( cur_ins.size() == reds.size() );
      for( uint32_t i = 0; i != reds.size(); ++i ) { must_insert( arg_map, reds[i]+"_in", cur_ins[i] ); }
      assert_st( cur_outs.size() == reds.size() );
      for( uint32_t i = 0; i != reds.size(); ++i ) { must_insert( arg_map, reds[i]+"_out", cur_outs[i] ); }
      must_insert( arg_map, "primary_in", make_scalar_nda(primary_in) );

      p_rcg_func_call_t rfc = codegen.gen_func( var_stats, arg_map );
      assert_st( rfc->rcg->rtc_call_geom.tpb == rtc_call_geom_t::get_default_tpb() );
      add_fwd_call( rfc, "var_stats__" + top_in );

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
    op_base_t quantize_op;
    quantize_op.set_func_name("quantize");
    quantize_op.set_dims("out",rtc->get_var_dims(top_in));
    quantize_op.set_dims("max_val",make_scalar_dims_t("uint32_t"));
    quantize_op.set_dims("drop_mask",make_scalar_dims_t("uint32_t"));
    p_rcg_func_call_t rtc = codegen.gen_func( quantize_op, map_str_rtc_arg_t{{"out",top_in}, 
        {"max_val",make_scalar_nda(max_val)},{"drop_mask",make_scalar_nda(drop_mask)}} );
    add_fwd_call( rtc , "quantize__"+top_in );
  }

  // this assumes that in_var is valid/calculated, and returns ret_var=func(in_var). it assumes that func is a stateless
  // unary operator (with two args: {in,out}), so that only one unique ret_var need only be generated per unique
  // in_var/func<ret_dims> pair. ret_var is named in_var+"__"+func_name+ret_dims_as_string.
  string conv_pipe_fwd_t::gen_apply_func_to_var( string const & in_an, string const & in_var, 
						 string const & ret_an, dims_t const & ret_dims, 
                                                 string const & func_name, p_conv_op_t const & oi )
  {
    p_rcg_func_call_t rfc = codegen.gen_func_override_func_name( func_name, *oi, map_str_rtc_arg_t() );
    string const ret_var = in_var + "__" + rfc->rcg->gen_fn;
    bool const did_ins = inxp_names.insert( ret_var ).second;
    if( did_ins ) { // newly-seen/used ret_var, so create and calc it here
      must_insert( rfc->arg_map, in_an,  in_var  );
      must_insert( rfc->arg_map, ret_an, ret_var );
      add_fwd_call( rfc, in_var + "__inxp" );
      rtc->create_var_with_dims( ret_var, ret_dims );
    }
    return ret_var;
  }

  // FIXME: mostly dup'd with similar code in rtc_func_gen.cc for generated function signatures
  typedef shared_ptr< op_base_t > p_op_base_t; 
  p_op_base_t make_p_op_base_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );
  void write_sigs( set_op_base_t & all_op_sigs, filename_t const & op_sigs_fn ) {
    if( boost::filesystem::is_regular_file( op_sigs_fn.exp ) ) {  // read in existing contents of file if it exists
      p_vect_string in_lines = readlines_fn( op_sigs_fn );
      for( vect_string::const_iterator i = in_lines->begin(); i != in_lines->end(); ++i ) {
	p_op_base_t v = make_p_op_base_t_init_and_check_unused_from_lexp( parse_lexp( *i ), 0 );
	all_op_sigs.insert( *v );
      }
    }
    // write set back out
    p_ostream out = ofs_open( op_sigs_fn );
    for( set_op_base_t::const_iterator i = all_op_sigs.begin(); i != all_op_sigs.end(); ++i ) { (*out) << str( *i ) << "\n"; }
  }

  void set_rtc_arg( p_conv_op_t const & oi, p_rtc_compute_t const & rtc, string const & an, string const & vn ) {
    oi->set_arg( rtc->get_var_dims(vn), an, vn );
  }
  
  void conv_pipe_fwd_t::gen_op( p_conv_op_t const & cop ) {
    if( write_op_sigs ) { all_op_sigs.insert( *cop ); } // unique ops if requested
    p_conv_op_t const & oi = must_find( *op_infos, cop->tag );
    if( oi->has( "fused" ) ) { return; } // operation was fused into another, so do nothing here for it
    if( oi->is( Concat_coi ) ) {      
      uint32_t chans_out_done = 0;
      for( uint32_t bi = 0; bi != oi->get_u32("ins_num"); ++bi ) {
	dims_t const & dims_in = oi->get_dims( oi->coi->bot_an(bi) );
	assert_st( get_xy_dims( dims_in ) == get_xy_dims( oi->get_dims("out") ) );
	assert_st( chans_out_done+dims_in.dsz("chan") <= oi->get_dims("out").dsz("chan") );
        oi->set_u32( "ocix", chans_out_done );
	set_rtc_arg( oi, rtc, "in", oi->get_arg( oi->coi->bot_an(bi) ) );
	gen_call( oi );
	chans_out_done += dims_in.dsz("chan");
	oi->erase_arg( "in" );
	oi->erase( "ocix" );
      }
      assert_st( chans_out_done == oi->get_dims("out").dsz("chan") );
    } else if( oi->is( Split_coi ) ) { // FIXME: pretty dup'd with Concat above ... generalize/merge/share?
      uint32_t chans_in_done = 0;
      for( uint32_t ti = 0; ti != oi->get_u32("outs_num"); ++ti ) {
	dims_t const & dims_out = oi->get_dims( oi->coi->top_an(ti) );
	assert_st( get_xy_dims( dims_out ) == get_xy_dims( oi->get_dims("in") ) );
	assert_st( chans_in_done+dims_out.dsz("chan") <= oi->get_dims("in").dsz("chan") );
        oi->set_u32( "icix", chans_in_done );
	set_rtc_arg( oi, rtc, "out", oi->get_arg( oi->coi->top_an(ti) ) );
	gen_call( oi );
	chans_in_done += dims_out.dsz("chan");
	oi->erase_arg( "out" );
	oi->erase( "icix" );
      }
      assert_st( chans_in_done == oi->get_dims("in").dsz("chan") );
    } else if( oi->is( Pooling_coi ) ) {
      if( oi->get_u32("emit_out_in_yx") == 1 ) {
	string const out_in_yx = oi->get_arg("out") + "_in_yx"; 
	rtc->create_var_with_dims( out_in_yx, oi->get_dims("out") ); // same size as out
	set_rtc_arg( oi, rtc, "out_in_yx", out_in_yx );
      } else {
	assert_st( oi->get_u32("emit_out_in_yx") == 0 );
	oi->set_null_arg_dims( "out_in_yx", oi->get_dims("out") ); // proper dims, but no var will be passed at call time
      }
      gen_call( oi );
    } else if( oi->is( Convolution_coi ) ) {
      op_param_names.push_back( oi->get_arg("filts") );
      op_param_names.push_back( oi->get_arg("biases") );
      if( force_zero_bias ) { force_zero_names.insert( oi->get_arg("biases") ); }
      string const filts_id = oi->get_arg("filts");
      if( oi->get_dims("filts") != rtc->get_var_dims( filts_id ) ) { // ipconv uses untransformed filts, otherwise:
	oi->reset_arg( "filts", gen_apply_func_to_var( "filts_ref", oi->get_arg("filts"), "filts", oi->get_dims("filts"), 
						       "xpose_filts", oi ) );
      }
      string const in_id = oi->get_arg("in");
      // note: as this point: oi->get_dims("in") may not == rtc->get_var_dims( in_id ); see comment in init()
      if( oi->get_func_name() == tconv_str ) {
	// assume input needs the below xform and apply it. FIXME(?): fails if vars are in unexpected formats.
	oi->reset_arg( "in", gen_apply_func_to_var( "in_ref", oi->get_arg("in"), "in", oi->get_dims("in"),
                                                    "tconv_xpose_in", oi ) );
      } else if( oi->get_func_name() == k1conv_str ) {
	if( oi->get_dims("in") != rtc->get_var_dims( in_id ) ) {
	  // if dims not exactly right, assume they are 'normal' dims and convert. FIXME(?): fails if vars are in unexpected formats.
	  oi->reset_arg( "in", gen_apply_func_to_var( "in_ref", oi->get_arg("in"), "in", oi->get_dims("in"), 
                                                      "k1conv_xpose_in", oi ) );
	} 	
      } 
      // FIXME: perhaps all ops should create outputs. but for now, only conv can have non-reference output dims ...
      rtc->create_var_with_dims( oi->get_arg("out"), oi->get_dims("out") ); 
      gen_call( oi );
    } else if( oi->is( ReLU_coi ) ) {
      assert_st( oi->get_arg("in") == oi->get_arg("out") ); // check that this is a single in-out in-place operation
      set_rtc_arg( oi, rtc, "inout", oi->get_arg("in") );
      gen_call( oi );
    } else if( oi->is( LRN_coi ) ) {
      assert_st( oi->get_dims("in") == oi->get_dims("out") ); // FIXME: better place/way for this check?
      if( oi->get_u32("emit_out_scale_base") == 1 ) {
	string const out_scale_base = oi->get_arg("out") + "_scale_base"; 
	rtc->create_var_with_dims( out_scale_base, oi->get_dims("out") ); // same size as out
	set_rtc_arg( oi, rtc, "out_scale_base", out_scale_base );
      } else {
	assert_st( oi->get_u32("emit_out_scale_base") == 0 );
	oi->set_null_arg_dims( "out_scale_base", oi->get_dims("out") );
      }
      gen_call( oi );
    } else if( oi->is( BckLRN_coi ) ) {
      set_rtc_arg( oi, rtc, "out_scale_base", oi->get_arg("out") + "_scale_base" ); // generated by matching LRN op
      gen_call( oi );
    } else if( oi->is( Dropout_coi ) ) {
      assert_st( oi->get_arg("in") == oi->get_arg("out") ); // check that this is a single in-out in-place operation
      set_rtc_arg( oi, rtc, "inout", oi->get_arg("in") );
      gen_call( oi );
      // FIXME: move this check (and others like it) to conv_util.cc or similar?
      float const dropout_ratio = SNE<float>( *oi->get("dropout_ratio") );
      assert_st( dropout_ratio > 0.0 );
      assert_st( dropout_ratio < 1.0 );
      // for below dep_drop_seed handling: see update code elsewhere. yeah, not the cleanest approach.
      must_insert( fwd_calls.back().rfc->arg_map, "det_drop_seed", rtc_arg_t() );
      dropout_cixs.push_back( fwd_calls.size() - 1 );
    } else if( oi->is( BckDropout_coi ) ) {
      assert_st( oi->get_arg("in") == oi->get_arg("out") ); // check that this is a single in-out in-place operation
      set_rtc_arg( oi, rtc, "inout", oi->get_arg("in") );
      gen_call( oi ); // Backwards of dropout is dropout
      must_insert( fwd_calls.back().rfc->arg_map, "det_drop_seed", rtc_arg_t() );
      dropout_cixs.push_back( fwd_calls.size() - 1 );
    } else if( oi->is( SoftmaxWithLoss_coi ) ) {
      string const prob_node_name = oi->tag + "_prob";
      gen_node_var( prob_node_name, oi->get_arg("in") );
      set_rtc_arg( oi, rtc, "prob", prob_node_name );
      string const loss_per_pel = oi->get_arg("loss") + "_per_pel"; // same size as label
      gen_node_var( loss_per_pel, oi->get_arg("label") );
      set_rtc_arg( oi, rtc, "loss_per_pel", loss_per_pel );
      gen_call( "softmax", oi );
      gen_call( "sm_grad_and_loss", oi  );
      gen_call( "sum_loss_over_imgs", oi );
    } else if( oi->is( Spreading_coi ) ) {
      set_rtc_arg( oi, rtc, "out_in_yx", oi->get_arg("out") + "_in_yx" ); // generated by matching Pooling op
      gen_call( oi );
    } else if( oi->is( BckConv_coi ) ) { 
      // { in, filts, biases, out_grad_loss } --> { in_grad_loss, filts_grad_loss, biases_grad_loss }
      string ogl_vn = oi->get_arg("out_grad_loss");
      string ogl_fn = "BckConv_in_grad_loss";
      string fgl_fn = "BckConv_filts_grad_loss";
      // FIXME the following assert used to hold, but only due to mis-setting func_name() (in those days, cts) in too
      // many cases (pool, BckConv, others). still, this assert is sort-of right, since there are no BckConv variants
      // yet, but also wrong since, like the conv case we use the 'gen_call()' override-with-no-func-name-set flow
      // below. so ... commented out for now:
      // assert_st( oi->get_func_name() == conv_str );
      if( enable_bconv ) {
#if 0
	dims_t const & ogl_dims = rtc->get_var_dims( ogl_vn );
	dims_t const & ogl_xp_dims = ogl_dims; // oi->dims_vals["out_grad_loss"];
	string ogl_xp_fn = gen_func( op_base_t{ "btconv_ogl_xpose", {ogl_dims,ogl_xp_dims}, 
	      oi->dims_vals, oi->str_vals } );
	ogl_vn = gen_apply_func_to_var( ogl_vn, ogl_xp_dims, ogl_xp_fn );
#endif
	ogl_fn = "bconv";
	fgl_fn = "bconv_fb";
      }
      gen_call( ogl_fn, oi );
      gen_call( "BckConv_biases_grad_loss", oi );
      gen_call( fgl_fn, oi );
    } else if( oi->is( ZeroIfNonPos_coi ) || oi->is( Softmax_coi ) || oi->is( Reduce_coi ) ) { 
      gen_call( oi ); // 'generic' cases (yes, there's only a few currently, but ya gotta dream, right?)
    } else { rt_err( "gen_op: unhandled op of type: " + oi->get_type() ); }
  }

  // generate call for given oi, using oi->get_func_name() to choose template/function
  void conv_pipe_fwd_t::gen_call( p_conv_op_t const & oi ) { 
    assert_st( oi->has_func_name() );
    p_rcg_func_call_t rfc = codegen.gen_func( *oi, oi->arg_map );
    if( oi->is( Convolution_coi ) && ( (oi->get_func_name() == tconv_str) || (oi->get_func_name() == k1conv_str) ) ) { 
      must_insert( rfc->arg_map, "flags", make_scalar_nda(flags) ); } // FIXME: not the place for this.
    add_fwd_call( rfc, oi->tag );
  }

  // used in cases where no single func_name()/template/function applies to operation. we override func_name(), generate
  // a call, and clear it back out. we could also equivalently create a copy of oi, fill in func_name, and discard it.
  void conv_pipe_fwd_t::gen_call( string const & fn, p_conv_op_t const & oi ) { 
    assert_st( !oi->has_func_name() );
    oi->set_func_name( fn ); 
    gen_call( oi ); 
    oi->erase_func_name();
  }

  // gen_node_var() creates a var directly corresponding to a pipe node.  usually, but not always, name == node_node; in
  // that case the var is directly mirroring a pipe node
  void conv_pipe_fwd_t::gen_node_var( string const & name, string const & node_name ) { 
    rtc->create_var_with_dims( name, cp->must_get_node(node_name)->dims );
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

  p_rtc_compute_t make_p_rtc_compute_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );

  void conv_pipe_fwd_t::init( p_conv_pipe_t const & cp_, nesi_init_arg_t * const nia ) {
    cp = cp_;
    assert_st( cp );

    op_infos.reset( new map_str_p_conv_op_t ); // maybe we should have our own copy of cp, but instead we only copy convs
    for( map_str_p_conv_op_t::iterator i = cp->convs->begin(); i != cp->convs->end(); ++i ) { 
      must_insert( *op_infos, i->first, make_shared< conv_op_t >( *i->second ) );
    }
    for( map_str_p_conv_op_t::iterator i = cp->convs->begin(); i != cp->convs->end(); ++i ) { 
      p_conv_op_t const & oi = must_find( *op_infos, i->first );
      add_cnn_codegen_annotations( oi.get(), op_tune, 0 );
    }

    // these parts might go in init, but they need to know about the overall graph of operations. so we'll call these a
    // set of post-init() but pre-codegen() graph operations on the set of conv_op_t's. both the whole graphs and all
    // individual operations should be valid for codegen and correct both before and after this pass (i.e. it is stricty
    // an optimzation pass).
    for( map_str_p_conv_op_t::iterator i = cp->convs->begin(); i != cp->convs->end(); ++i ) { 
      p_conv_op_t const & oi = must_find( *op_infos, i->first );
      if( oi->is( Convolution_coi ) ) {
	p_conv_node_t no = cp->must_get_node( oi->get_arg("out") ); // aka oi->coi->top_an(0) ...
	bool const conv_has_relu = (no->in_place_ops.size() > 0) && (no->in_place_ops[0]->is(ReLU_coi));
	// mark relu as fused-away; mark conv as having fused-on relu // NOTE/FIXME(?): relu may be not-init()-yet here ...
	if( conv_has_relu ) { must_find( *op_infos, no->in_place_ops[0]->tag )->set_u32( "fused", 1 ); } 
	oi->set_u32( "conv_has_relu", conv_has_relu );

	if( oi->get_func_name() == k1conv_str ) { 
	  if( ( no->in_place_ops.size() == conv_has_relu ) && ( no->bot_for.size() == 1) ) { // if output feeds single non-in-place operation
	    p_conv_op_t const & noi = must_find( *op_infos, no->bot_for[0] ); // next operation
	    if( enable_write_xpose && noi->is( Convolution_coi ) && (noi->get_func_name() == k1conv_str) ) { 
	      // modify output argument dims to match user's input dims. codegen will notice this and write out correctly.
	      oi->reset_arg_dims( "out", noi->get_dims( "in" ) );
	    }
	  }
	}

      }
    }
    if( !rtc ) { rtc = make_p_rtc_compute_t_init_and_check_unused_from_lexp( parse_lexp( "(be=nvrtc)" ), nia ); }
    rtc->init(); codegen.init( rtc, make_cnn_custom_codegen_t(), compile_opts );
    cp->topo_visit_setup();
    for( set_string::const_iterator i = cp->bots.begin(); i != cp->bots.end(); ++i ) { gen_ops_rec( *i ); }
    //codegen.write_rtc_func_sigs( rtc_func_sigs_fn );
    if( write_op_sigs ) { write_sigs( all_op_sigs, op_sigs_fn ); }
    rtc->copy_ndas_to_vars( op_param_names, *cp->op_params ); // copy op_params in (FIXME/note: implicit  on names)
    for( set_string::const_iterator i = force_zero_names.begin(); i != force_zero_names.end(); ++i ) { rtc->set_var_to_zero( *i ); }
    rtc->finish_and_sync();
  }

  void conv_pipe_fwd_t::run_fwd( vect_string const & to_set_vns, p_map_str_p_nda_float_t const & fwd, vect_string const & to_get_vns ) {
    if( enable_double_run ) {
      // optional: run fwd rfc's one for testing/flushing/cache setup. note: ~*doubles* total run time ...
      for( vect_rtc_fwd_func_call_t::iterator i = fwd_calls.begin(); i != fwd_calls.end(); ++i ) { 
        codegen.run_func( *i->rfc ); 
      }
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
      for( vect_rtc_fwd_func_call_t::iterator i = fwd_calls.begin(); i != fwd_calls.end(); ++i ) { 
        i->call_id = codegen.run_func( *i->rfc ); 
      }
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
      p_ostream out = ofs_open( per_call_fn );
      (*out) << strprintf("net.args.runtime=%s\n", str(compute_dur/1000.0).c_str() );
      for( vect_rtc_fwd_func_call_t::iterator i = fwd_calls.begin(); i != fwd_calls.end(); ++i ) {
	rcg_func_call_t & rfc = *i->rfc;
	if( i->call_tag.empty() ) { continue; }
	float const rfc_dur = rtc->get_dur( i->call_id, i->call_id );
	(*out) << strprintf( "per_layer_time['%s']=per_layer_time.get('%s',0.0) + %s # %s \n", 
			     str(i->call_tag).c_str(), str(i->call_tag).c_str(), str(rfc_dur/1000.0).c_str(), 
                             rfc.rcg->gen_fn.c_str() );
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
