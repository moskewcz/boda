// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"build_info.H"
#include"timers.H"
#include<boost/filesystem.hpp>
#include<boost/lexical_cast.hpp>
#include"lexp.H"
#include"has_main.H"
#include"rtc_func_gen.H"
#include"rtc_compute.H"
#include"cnn_op.H"
#include"conv_util.H"
#include"comp_util.H"

namespace boda 
{
  typedef shared_ptr< dims_t > p_dims_t; 

  struct rtc_prof_t : virtual public nesi, public has_main_t // NESI(help="profile set of rtc functions",
		      // bases=["has_main_t"], type_id="rtc_prof" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    rtc_compile_opts_t compile_opts; // NESI(default="()",help="runtime compilation options")
    uint32_t eat_megs; //NESI(default=0,help="if non-zero, allocate unused var of size eat_mega Mfloats via rtc")
    filename_t rtc_func_sigs_fn; //NESI(default="%(boda_test_dir)/rtc_func_sigs_tiny.txt",help="file to hold all generated func signatures")
    p_dims_t dummy_dims; // NESI(help="HACK: dummy NESI var of type dims_t (otherwise unused) to force tinfo generation. see map_str_T FIXME in nesi.cc")

    p_rtc_compute_t rtc; //NESI(default="(be=ocl)",help="rtc back-end to use")
    filename_t per_call_fn; //NESI(default="%(boda_output_dir)/rtc_prof.py",help="if non-empty, write per-call profiling (timing via events) to given file.")

    vect_rcg_func_call_t calls;
    // rtc->create_var_with_dims_floats( name, cp->must_get_node(node_name)->dims );
    // calls.push_back( rcg_func_call_t{ gen_fn, oi->tag, oi->arg_map } );
    
    p_ostream out;
    rtc_codegen_t codegen;

    virtual void main( nesi_init_arg_t * nia );

    double run_call( string const & func_name, p_rtc_call_gen_t const & rcg );
    void run_calls( void );
  };

  // semi-dupe'd with rtc_fwd gen_apply_func_to_var(). working toward convergence. note that in this use model, the
  // input and output variable names and arg names happen to be the same, hence the 'an_and_vn' arguments to this func.
  void run_xpose( p_op_base_t const & anno_op, rtc_codegen_t & codegen, string const & xpose_func_name, 
                  string const &out_an_and_vn, string const &in_an_and_vn )  {
    p_rcg_func_call_t rfc = codegen.gen_func_override_func_name( xpose_func_name, *anno_op, 
                           map_str_rtc_arg_t{{out_an_and_vn,out_an_and_vn},{in_an_and_vn,in_an_and_vn}});
    codegen.run_func( *rfc );
  }
  
  double profile_rcg_call( p_op_base_t const & anno_op, rtc_codegen_t & codegen,
			   p_op_base_t const & in_gen_op_orig, map_str_p_nda_t * const outs,
                           uint32_t const & run_iter ) 
  {
    timer_t t("profile_rcg_call");
    string const anno_op_func_name = anno_op->get_func_name();
    p_rcg_func_call_t rfc = codegen.gen_func( *anno_op, map_str_rtc_arg_t() ); // FIXME: not passing in args here. yet?
    p_rtc_call_gen_t const & rcg = rfc->rcg;
    map_str_rtc_arg_t & arg_map = rfc->arg_map;
    for( vect_arg_decl_t::multi_iter i = rcg->rtc_func_template->arg_decls.multi_begin( &rcg->op ); !i.at_end(); ++i ) {
      if( i.ad().io_type == "REF" ) { continue; }
      if( i.vn() == "cucl_arg_info" ) { continue; } // FIXME: not-too-nice special case for cucl_arg_info argument 
      if( i.ad().loi.v == 0 ) { // FIXME: not-too-nice special case for flags
        if( i.vn() == "flags" ) { must_insert( arg_map, "flags", make_scalar_nda(uint32_t(0)) ); continue; }
      }
      dims_t const & func_dims = rcg->get_arg_dims_by_name( i.vn() );
      if( func_dims == make_null_dims_t() ) { continue; } // NULL case -- ignore
      // FIXME: overwrite dims. yeah, this doesn't feel too right ... hmm. see comments in gen_func()
      arg_map[ i.vn() ] = i.vn();
    }
    // FIXME: horrible: some kernels take a scalar uint32_t flags, and we know 0 is 'normal-mode'. so we set it here,
    // for all ops, and hope that's okay.

    //printf( "run: i->rtc_func_name=%s\n", str(rcg->gen_fn).c_str() );
    for( map_str_rtc_arg_t::const_iterator j = arg_map.begin(); j != arg_map.end(); ++j ) {
      if( j->second.is_var() ) { 
        codegen.rtc->create_var_with_dims( j->second.n, anno_op->get_dims( j->first ) ); 
      }
    }
    vect_string xpose_vars_to_release;
    if( in_gen_op_orig ) { 
      for( vect_arg_decl_t::multi_iter i = rcg->rtc_func_template->arg_decls.multi_begin( &rcg->op ); !i.at_end(); ++i ) {
        p_op_base_t in_gen_op = make_shared<op_base_t>( *in_gen_op_orig );
	if( i.ad().io_type != "IN" ) { continue; }
        if( i.vn() == "cucl_arg_info" ) { continue; } // FIXME: not-too-nice special case for cucl_arg_info argument 
        if( i.ad().loi.v == 0 ) { continue; } // FIXME: not-too-nice special case for scalars ... better be const.
        // note: gen_data variant choice based on gen type and op type (*not* op func_name)
	in_gen_op->set_func_name( in_gen_op->get_type()+"_"+anno_op->get_type()+"_"+i.vn() ); 
        dims_t const & in_dims = anno_op->get_dims( i.vn() );
        string const ref_in_dims_name = i.vn()+"_ref";
        dims_t const & ref_in_dims = anno_op->has(ref_in_dims_name)?anno_op->get_dims(ref_in_dims_name):in_dims;
	in_gen_op->set_dims( i.vn(), ref_in_dims );
        string gen_vn = i.vn();
        if( in_dims != ref_in_dims ) { 
          gen_vn += "_ref"; 
          codegen.rtc->create_var_with_dims( gen_vn, ref_in_dims ); 
          xpose_vars_to_release.push_back( gen_vn );
        }
	p_rcg_func_call_t rfc_in_gen = codegen.gen_func( *in_gen_op, map_str_rtc_arg_t{{i.vn(),gen_vn}} );
	codegen.run_func( *rfc_in_gen );
        // check if xpose needed:
        if( gen_vn != i.vn() ) {
          // FIXME: some ugly, cut-n-paste, brittle stuff here ... but it's pending more global cleanup.
          string xpose_op = anno_op_func_name+"_xpose_"+i.vn();
          // FIXME: sigh.
          if( ( i.vn() == "filts" ) && is_k1_or_t_or_reg_conv(anno_op->get_func_name())) { xpose_op = "xpose_filts"; }
          run_xpose( anno_op, codegen, xpose_op, gen_vn, i.vn() );
        }
	//if( outs ) { must_insert( *outs, i.vn(), p_nda_float_t() ); } // include inputs in 'outputs'
      }
    }

    uint32_t call_id = uint32_t_const_max;
    for( uint32_t i = 0; i != run_iter; ++i ) { call_id = codegen.run_func( *rfc ); }

    // FIXME: xpose of OUTs is semi-dup'd with "IN"/gen_data handling above
    for( vect_arg_decl_t::multi_iter i = rcg->rtc_func_template->arg_decls.multi_begin( &rcg->op ); !i.at_end(); ++i ) {
      if( !endswith( i.ad().io_type, "OUT" ) ) { continue; }
      dims_t const & out_dims = anno_op->get_dims( i.vn() );
      string const ref_out_dims_name = i.vn()+"_ref";
      dims_t const & ref_out_dims = anno_op->has(ref_out_dims_name)?anno_op->get_dims(ref_out_dims_name):out_dims;
      string gen_vn = i.vn();
      if( out_dims != ref_out_dims ) { 
        gen_vn += "_ref"; 
        codegen.rtc->create_var_with_dims( gen_vn, ref_out_dims ); 
        xpose_vars_to_release.push_back( gen_vn );
      }
      if( gen_vn != i.vn() ) { run_xpose( anno_op, codegen, anno_op_func_name+"_xpose_"+i.vn(), gen_vn, i.vn() ); }
      if( outs ) { must_insert( *outs, i.vn(), codegen.rtc->create_nda_from_var( gen_vn ) ); } 
    }
    for( map_str_rtc_arg_t::const_iterator j = arg_map.begin(); j != arg_map.end(); ++j ) {
      if( j->second.is_var() ) {
        codegen.rtc->release_var( j->second.get_var() );
      }
    }
    for( vect_string::const_iterator i = xpose_vars_to_release.begin(); i != xpose_vars_to_release.end(); ++i ) {
      codegen.rtc->release_var( *i );
    }
    // get call duration
    //if( rfc.call_tag.empty() ) { release; return; } // FIXME: possible here? 
    codegen.rtc->finish_and_sync();
    double const rfc_dur = codegen.rtc->get_dur( call_id, call_id );
    codegen.rtc->release_per_call_id_data();
    rfc.reset(); // optional. allows just-used function (which is no longer needed) to be released now if func-gc happens.
    codegen.gc_clear();
    return rfc_dur;
  }

  p_conv_op_base_t make_p_conv_op_base_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );

  void rtc_prof_t::main( nesi_init_arg_t * nia ) {
    out = ofs_open( per_call_fn );
    rtc->init(); codegen.init( rtc, make_cnn_custom_codegen_t(), compile_opts );
    bool const enable_prof = 0;
    if( enable_prof ) { rtc->profile_start(); }
    if( eat_megs ) { rtc->create_var_with_dims( "MEMEATER", dims_t{ {1024,1024,eat_megs}, {"a","b","M"}, "float" } ); }

    p_istream rtc_func_sigs = ifs_open( rtc_func_sigs_fn );
    p_op_wisdom_t op_wisdom;
    while( op_wisdom = read_next_wisdom( rtc_func_sigs_fn.exp, rtc_func_sigs ) ) {
      p_op_base_t v = op_wisdom->op;
      double const rfc_dur = profile_rcg_call( v, codegen, 0, 0, 1 );
      (*out) << strprintf( "per_layer_time['tag']=per_layer_time.get('tag',0.0) + %s\n", str(rfc_dur/1000.0).c_str() );
    }
    if( enable_prof ) { rtc->profile_stop(); }
    rtc->finish_and_sync();
  }


  
  struct ops_be_t {
    string rtcn;
    p_rtc_compute_t rtc;
    p_rtc_codegen_t codegen;
  };
  typedef shared_ptr< ops_be_t > p_ops_be_t; 
  typedef map< string, ops_be_t > map_str_ops_be_t;

  struct ops_run_t {
    p_op_base_t op;
    op_tune_t op_tune;
    p_conv_op_base_t anno_op;    
    p_map_str_p_nda_t vs;
    double dur_secs;
  };
  typedef vector< ops_run_t > vect_ops_run_t; 


  struct ops_prof_t : virtual public nesi, public has_main_t // NESI(help="profile set of operations across backends and tuning params",
			 // bases=["has_main_t"], type_id="ops-prof" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    p_filename_t out_fn; //NESI(help="output file (output goes to stdout if not specified)")

    // FIXME: we should use a map_str_p_rtc_compute_t here, but NESI doesn't support that yet. might work with manual typedef initally?
    vect_p_rtc_compute_t rtcs; //NESI(help="list of compute backends to use")
    vect_string rtcns; //NESI(help="list of names of compute backends (must have same # of elements as --rtcs option")

    rtc_compile_opts_t compile_opts; // NESI(default="()",help="runtime compilation options")
    filename_t ops_fn; //NESI(default="%(boda_test_dir)/ops-prof/ops-prof-debug.txt",help="file to read ops from")

    map_str_op_tune_t op_tunes; //NESI(default="()",help="tuning parameters / options")

    uint32_t run_iter; //NESI(default="1",help="re-run op to profile this many times (for power testing)")

    p_op_base_t gen_data; //NESI(help="test-pattern data generation parameters (if not provided, inputs will be zeros)")

    double mrd_toler; //NESI(default="2e-4",help="maximum maximum-absolute-difference over which a failure is declared")
    map_str_double var_mrd_toler; //NESI(default="()",help="per-layer custom maximum maximum-absolute-differences over which a failure is declared (overrides mrd_toler per-layer if specified")
    uint32_t max_err; //NESI(default="10",help="print at most this many differing elems")

    map_str_ops_be_t ops_bes;

    p_rtc_codegen_t & get_codegen_for_op( ops_run_t const & ops_run ) {
      assert_st( !ops_bes.empty() );
      if( ops_run.op_tune.use_be.empty() ) { return ops_bes.begin()->second.codegen; }
      return must_find( ops_bes, ops_run.op_tune.use_be ).codegen;
    }

    virtual void main( nesi_init_arg_t * nia );
  };

  p_rtc_compute_t make_p_rtc_compute_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );


  void ops_prof_t::main( nesi_init_arg_t * nia ) {
    p_ostream out = out_fn ? ofs_open( *out_fn ) : p_ostream( &std::cout, null_deleter<std::ostream>() );

    // by default, add all enabled/availible backends
    if( rtcs.size() != rtcns.size() ) { rt_err( strprintf( "must specific the same # of rtcs and rtcns, but rtcs.size()=%s and rtcns.size()=%s\n", str(rtcs.size()).c_str(), str(rtcns.size()).c_str() ) ); }
    if( rtcs.empty() ) {
      if( is_feature_enabled("nvrtc") ) {
        ops_be_t ops_be{ "nvrtc", make_p_rtc_compute_t_init_and_check_unused_from_lexp( parse_lexp( "(be=nvrtc)" ), nia ) };
        must_insert( ops_bes, ops_be.rtcn, ops_be );
      }
      if( is_feature_enabled("opencl") ) { 
        ops_be_t ops_be{ "ocl", make_p_rtc_compute_t_init_and_check_unused_from_lexp( parse_lexp( "(be=ocl)" ), nia ) };
        must_insert( ops_bes, ops_be.rtcn, ops_be );
      }
    } else { // otherwise, use exactly/only the specified backends
      for( uint32_t i = 0; i != rtcs.size(); ++i ) { must_insert( ops_bes, rtcns[i], ops_be_t{rtcns[i],rtcs[i]} ); }
    }
    // init backends
    bool const enable_prof = 0;
    for( map_str_ops_be_t::iterator i = ops_bes.begin(); i != ops_bes.end(); ++i ) {
      ops_be_t & ops_be = i->second;
      ops_be.rtc->init();
      ops_be.codegen = make_shared<rtc_codegen_t>();
      ops_be.codegen->init( ops_be.rtc, make_cnn_custom_codegen_t(), compile_opts );
      if( enable_prof ) { ops_be.rtc->profile_start(); }
    }

    uint32_t num_mad_fail = 0;
    p_istream ops = ifs_open( ops_fn );
    p_op_wisdom_t op_wisdom;
    while( op_wisdom = read_next_wisdom( ops_fn.exp, ops ) ) {
      p_op_base_t op = op_wisdom->op;
      vect_ops_run_t ops_runs;
      for( map_str_op_tune_t::const_iterator i = op_tunes.begin(); i != op_tunes.end(); ++i ) {
        ops_runs.push_back( ops_run_t{op,i->second,make_shared<conv_op_base_t>( *op ),make_shared<map_str_p_nda_t>()} );
      }

      for( vect_ops_run_t::iterator i = ops_runs.begin(); i != ops_runs.end(); ++i ) {
        ops_run_t & ops_run = *i;
        assert_st( ops_run.vs->empty() );
        
        // generate boda variant according to tuning params (just opt and t_tile_sz currently)
        add_codegen_annotations( ops_run.anno_op, ops_run.op_tune, 0 );        
        if( gen_data ) { assert_st( gen_data->get_type() == "gen_data" ); } // FIXME: remove assert after fixing existing usages
        ops_run.dur_secs = NAN;
        string err;
        p_rtc_codegen_t const & codegen = get_codegen_for_op( ops_run );
        try { ops_run.dur_secs = profile_rcg_call( ops_run.anno_op, *codegen, gen_data, ops_run.vs.get(), run_iter ) / 1000.0; }
        catch( rt_exception const & rte ) {
          if( rte.what_and_stacktrace().find( "CL_OUT_OF_HOST_MEMORY" ) != string::npos ) { 
            err = "CL_OUT_OF_HOST_MEMORY"; 
            // FIXME: we should probably handle this at the rtc_codegen_t level better. in fact, there's a good chance the
            // handling is currently broken ... so for now, we'll give up here. note we used to call:
            // codegen.clear(); 
            assert_st( "TODO: re-handle compile/run failures better in codegen/prof" );
          }
          else { throw; }
        }
        if( err.empty() ) {
          if( i != ops_runs.begin() ) {
            vect_string const vns1 = get_keys( *ops_runs.front().vs );
            vect_string const vns2 = get_keys( *ops_run.vs );
            if( vns1 != vns2 ) { rt_err( strprintf( "reg/comp out var set mismatch: vns[0]=%s vns[%s]=%s\n", 
                                                    str(vns1).c_str(), str(i - ops_runs.begin()).c_str(), str(vns2).c_str() ) ); }
            (*out) << strprintf( "vars_to_compare: %s\n", str(vns1).c_str() );
            comp_vars( out.get(), num_mad_fail, mrd_toler, &var_mrd_toler, 0, max_err, vns1, ops_runs.front().vs, ops_run.vs );
          }
        } else {
          rt_err( "profile_rcg_call() failed: " + err ); 
        }
        
      }
    }

    for( map_str_ops_be_t::iterator i = ops_bes.begin(); i != ops_bes.end(); ++i ) {
      ops_be_t & ops_be = i->second;
      if( enable_prof ) { ops_be.rtc->profile_stop(); }
      ops_be.rtc->finish_and_sync(); 
    }

    if( !num_mad_fail ) { (*out) << strprintf( "***ALL IS WELL***\n" ); }
    else { (*out) << strprintf( "***MAD FAILS*** num_mad_fail=%s\n", str(num_mad_fail).c_str() ); }

  }

  void add_to_with_prefix( vect_pair_str_str & out, vect_pair_str_str const & in, pair_str_str const & prefix ) {
    for( vect_pair_str_str::const_iterator i = in.begin(); i != in.end(); ++i ) {
      out.push_back( {prefix.first+i->first,prefix.second+i->second} );
    }
  }

  void emit_clis( p_ostream & out, vect_pair_str_str const & run_bases, vect_pair_str_str const & op_tunes ) {
    for( vect_pair_str_str::const_iterator i = run_bases.begin(); i != run_bases.end(); ++i ) {
      pair_str_str cli = *i;
      cli.second += " --op-tunes='(";
      for( vect_pair_str_str::const_iterator j = op_tunes.begin(); j != op_tunes.end(); ++j ) {
        cli.second += string((j==op_tunes.begin())?"":",") + j->first +"=("+j->second+")";
      }
      cli.second += ")'";
      (*out) << strprintf( "<li test_name=\"%s\" cli_str=\"%s\" />\n", str(cli.first).c_str(), str(cli.second).c_str() );
    }
  }

  void gen_ops_prof_tests( p_ostream & out ) {
    
    vect_pair_str_str op_tune_sgemm_bases = { {"def", ""},
                                              {"4-16-4-lm0","MNt=4:4,MNb=16:16,Kb=4,use_local_mem=0"},
                                              {"4-16-4-lm2-vw4","MNt=4:4,MNb=16:16,Kb=4,use_local_mem=2,vw=4"},
                                              {"4-16-4-lm3-vw4","MNt=4:4,MNb=16:16,Kb=4,use_local_mem=3,vw=4" } };
    // FIXME: enable_write_xpose=1 goes where? doesn't much matter for single ops? 
    // FIXME: enable_bconv=1 goes where? not part of per-op flow currently ... maybe it should be though.
    // FIXME: note: can't test per-op against caffe here -- should be enable that somehow? it's covered by test_compute_multi, of course ..
    vect_pair_str_str op_tune_conv_bases = { {"def",""},
                                             {"opt","k1conv=1,tconv=1"} };
    vect_pair_str_str op_tunes_sgemm;
    vect_pair_str_str op_tunes_conv;
    if( is_feature_enabled("opencl") ) { 
      add_to_with_prefix( op_tunes_sgemm, op_tune_sgemm_bases, {"ocl-","use_be=ocl,"} );
      add_to_with_prefix( op_tunes_conv, op_tune_conv_bases, {"ocl-","use_be=ocl,"} );
    }
    if( is_feature_enabled("nvrtc") ) { 
      add_to_with_prefix( op_tunes_sgemm, op_tune_sgemm_bases, {"nvrtc-","use_be=nvrtc,"} );
      op_tunes_sgemm.push_back( {"culibs","use_be=nvrtc,use_culibs=1"} ); 
      add_to_with_prefix( op_tunes_conv, op_tune_conv_bases, {"nvrtc-","use_be=nvrtc,"} );
      op_tunes_conv.push_back( {"culibs","use_be=nvrtc,use_culibs=1"} ); 
    }
                                 
    string const cli_base = "boda ops-prof --out-fn='%(boda_output_dir)/cnn_op_info.txt'";
    string const sgemm_ops = " --ops-fn='%(boda_test_dir)/sgemm-ops-debug.txt'";
    string const cnn_ops = " --ops-fn='%(boda_test_dir)/conv-ops-debug.txt'";
    string const gen_data_mode_600 = " --gen-data='(str_vals=(type=gen_data),nda_vals=(vi=(tn=float,v=0.0),mode=(tn=uint32_t,v=600)))'";
    string const gen_data_mode_5 = " --gen-data='(str_vals=(type=gen_data),nda_vals=(vi=(tn=float,v=0.0),mode=(tn=uint32_t,v=5)))'";
    vect_pair_str_str run_bases_sgemm;
    run_bases_sgemm.push_back( {"sgemm-gen600", cli_base + sgemm_ops + gen_data_mode_600 } );
    run_bases_sgemm.push_back( {"sgemm-gen5", cli_base + sgemm_ops + gen_data_mode_5 } );
    vect_pair_str_str run_bases_conv;
    run_bases_conv.push_back( {"conv-gen5", cli_base + cnn_ops + gen_data_mode_5} );
    
    (*out) << "<root>\n";
    emit_clis( out, run_bases_sgemm, op_tunes_sgemm );
    emit_clis( out, run_bases_conv, op_tunes_conv );
    (*out) << "</root>\n";                                
  }

  // normally, the output of this mode is generated automatically by a magic filename-based hack in
  // test_cmds_t. however, this mode is provided as a way to generate the file without going though test_cmds_t ...
  struct gen_ops_prof_tests_t : virtual public nesi, public has_main_t // NESI( help="generate list of ops-prof tests",
			// bases=["has_main_t"], type_id="gen_ops_prof_tests")
  {
    filename_t out_fn; //NESI(default="gen_ops_prof_tests.xml",help="output: xml list of tests.")
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support    
    virtual void main( nesi_init_arg_t * nia ) {
      p_ostream out = ofs_open( out_fn.exp );
      gen_ops_prof_tests( out );
    }
  };


#include"gen/rtc_prof.cc.nesi_gen.cc"

}
