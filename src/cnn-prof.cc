// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"timers.H"
#include<boost/filesystem.hpp>
#include<boost/lexical_cast.hpp>
#include"lexp.H"
#include"has_main.H"
#include"rtc_func_gen.H"
#include"rtc_compute.H"
#include"conv_util.H"
#include"comp_util.H"
#include<iostream>
#include"cnn_op.H"

namespace boda 
{

  // FIXME: mostly dup'd with similar code in rtc_func_gen.cc for generated function signatures
  typedef shared_ptr< conv_op_base_t > p_conv_op_base_t; 
  p_conv_op_base_t make_p_conv_op_base_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );

  string dims_yxc_str( dims_t const & d, bool include_img = 0 ) { 
    return strprintf( "$ %s %s \\dx %s \\dx %s $", include_img ? (str(d.dsz("img"))+" \\dx").c_str():"",
		      str(d.dsz("y")).c_str(), str(d.dsz("x")).c_str(), str(d.dsz("chan")).c_str() ); }
  string mkn_str( uint64_t const & M, uint64_t const & K, uint64_t const & N  ) { 
    if( M == K && K == N ) { return strprintf( "$ %s $", str(M).c_str() ); }
    else { return strprintf( "$ %s \\dx %s \\dx %s $", str(M).c_str(), str(K).c_str(), str(N).c_str() ); }
  }


  struct conv_op_info_to_latex_t {
    p_conv_op_base_t op;    
    dims_t din;
    dims_t dout;
    uint64_t B;
    uint64_t M,N,K;
    uint64_t forward_bytes, forward_flops;
    bool emit_bw;
    uint32_t inc_op_info_in_eff;
    // locally override the global pp_foo() function with member functions that can control formatting
    uint32_t print_format;
#define PP_FMT( t ) string pp_##t( double const v ) const { return (print_format == 0) ? boda::pp_##t( v ) : str(v); }
    PP_FMT( bytes ) PP_FMT( flops ) PP_FMT( val ) PP_FMT( secs ) PP_FMT( fps ) PP_FMT( bps )
#undef PP_FMT

    void base_info( std::ostream * const out ) {
      if( op->is( Convolution_coi ) ) {
	assert_st( op->kern_sz().dims_are_same() );
	assert_st( op->stride().dims_are_same() );
	(*out) << strprintf( "%s & %s & %s", str(op->kern_sz().d[0]).c_str(), str(op->stride().d[0]).c_str(), str(dout.dsz("chan")).c_str() );
      }
    }
    // MKN & Bytes & FLOPs & F/B
    void ai_mkn_row( std::ostream * const out ) {
      double const ai = double(forward_flops)/double(forward_bytes);
      (*out) << strprintf( " %s & %s & %s & %s ", mkn_str(M,K,N).c_str(), 
                           pp_bytes(forward_bytes).c_str(), pp_flops(forward_flops).c_str(), pp_val(ai).c_str() );
    }
    void info_row( std::ostream * const out ) {
      base_info( out );
      if( op->is( Convolution_coi ) ) {
	(*out) << strprintf( " & %s & %s & %s & ", str(B).c_str(), dims_yxc_str(din).c_str(), dims_yxc_str(dout).c_str() );
      }
      ai_mkn_row( out );
      (*out) << "\\\\ " << std::endl;
    }

    // SGEMM comp eff row
    // MKN & Bytes & FLOPs & F/B & Runtime(comp) & GF/s(comp) & Runtime(non-comp) & GF/s(non-comp) & Speedup-of-non-comp (comp/non-comp)

    // conv eff row
    // KSZ & Stride & out_chans & $dims(in)$ & variant & MxKxN & Bytes & FLOPs & F/B & Runtime & GF/s & %Peak
    void eff_row( std::ostream * const out, string const & rtc_op_type, 
                  double const & runtime_secs, double const & peak_flops,
                  double const & runtime_secs_comp ) {
      if( op->is( sgemm_coi ) ) {
        ai_mkn_row( out );
        double const fps_comp = double(forward_flops)/runtime_secs_comp;
        (*out) << strprintf( " & %s & %s ", pp_secs(runtime_secs_comp).c_str(), pp_fps(fps_comp).c_str() );
        double const fps = double(forward_flops)/runtime_secs;
        (*out) << strprintf( " & %s & %s ", pp_secs(runtime_secs).c_str(), pp_fps(fps).c_str() ); 
        (*out) << strprintf( " & %.2fx ", double(runtime_secs_comp/runtime_secs) );
      }
      else {
        base_info( out );
        (*out) << strprintf( " & %s & \\verb|%s| & ", dims_yxc_str(din,1).c_str(), rtc_op_type.c_str() );
        if( inc_op_info_in_eff ) { ai_mkn_row( out ); (*out) << " & "; }
        double const fps = double(forward_flops)/runtime_secs;
        (*out) << strprintf( " %s & %s & %s ", pp_secs(runtime_secs).c_str(), pp_fps(fps).c_str(), pp_val(fps/peak_flops*100.0).c_str() ); 

        if( emit_bw ) {
          // HACK: emit human-readable BW #s for now, breaks later flow/latex
          double const peak_bps = 20e9;
          double const bps = double(forward_bytes)/runtime_secs;
          (*out) << strprintf( " -- %s %s --", pp_bps(bps).c_str(), pp_val(bps/peak_bps*100.0).c_str() );
        }
      }
      (*out) << "\\\\ " << std::endl;
    }
    void init( p_conv_op_base_t const & op_, uint32_t const & print_format_, uint32_t const & inc_op_info_in_eff_) {
      print_format = print_format_;
      inc_op_info_in_eff = inc_op_info_in_eff_;
      op = op_;
      emit_bw = 0;
      if( op->is( Convolution_coi ) ) {
	dout = op->get_dims("out");
	din = op->get_dims("in");
	B = din.dsz( "img" );
	assert_st( B == dout.dsz("img" ) );
	// AI-related calculations
	dims_t const & filts = op->get_dims("filts");
	dims_t const & biases = op->get_dims("biases");
	M = dout.dsz("img")*dout.dsz("x")*dout.dsz("y"); // note: all-imgs M
	K = filts.dsz("in_chan")*filts.dsz("x")*filts.dsz("y");
	N = filts.dsz("out_chan");
	forward_bytes = (din.dims_prod() + dout.dims_prod() + filts.dims_prod() + biases.dims_prod()) * 4;
	forward_flops = M * N * K * 2;
      } else if( op->is( sgemm_coi ) ) {
	dout = op->get_dims("c");
	dims_t a = op->get_dims("a");
	dims_t b = op->get_dims("b");
	B = 1;
	M = a.dsz("M");
	K = a.dsz("K");
	assert_st( b.dsz("K") == K );
	N = b.dsz("N");
	assert_st( dout.dsz("M") == M );
	assert_st( dout.dsz("N") == N );
	forward_flops = M * N * K * 2;
	forward_bytes = (a.dims_prod() + b.dims_prod() + dout.dims_prod()) * 4;
      } else { rt_err( "cnn-op-info: unhandled op: " + op->type ); }
      
    }
  };


  struct cnn_op_info_t : virtual public nesi, public has_main_t // NESI(help="print info for set of CNN operations",
			 // bases=["has_main_t"], type_id="cnn_op_info" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    rtc_compile_opts_t compile_opts; // NESI(default="()",help="runtime compilation options")
    filename_t cnn_func_sigs_fn; //NESI(default="%(boda_test_dir)/cnn_func_sigs_tiny.txt",help="file to read cnn ops from")
    p_filename_t out_fn; //NESI(help="output file (output goes to stdout if not specified)")
    p_filename_t op_info_tab_fn; //NESI(help="file to write op info latex rows to")
    p_filename_t op_eff_tab_fn; //NESI(help="file to write op info latex rows to")
    uint32_t print_format; //NESI(default="0",help="0 == pretty printing; 1 == raw printing")
    uint32_t inc_op_info_in_eff; //NESI(default="0",help="if 1 includ full op info in eff lines")
    double peak_flops; //NESI(default=6600e9,help="peak flops of platform (for computing peak %s)")

    op_tune_t op_tune; //NESI(default="()",help="tuning parameters / options")
    uint32_t run_iter; //NESI(default="1",help="re-run op to profile this many times (for power testing)")
    op_tune_t op_tune_comp; //NESI(default="(use_culibs=1,MNt=4:4,MNb=8:8,Kb=1,opt=0)",help="tuning parameters / options")

    map_str_op_tune_t per_op_tune; //NESI(default="()",help="tuning parameters / options")

    p_op_base_t gen_data; //NESI(help="test-pattern data generation parameters (if not provided, inputs will be zeros)")
    p_rtc_compute_t rtc; //NESI(default="(be=ocl)",help="rtc back-end to use")

    // comparison testing related options:
    p_rtc_compute_t rtc_comp; //NESI(help="rtc back-end to use for correctness-testing comparison")
     double mad_toler; //NESI(default="1e-5",help="maximum maximum-absolute-difference over which a failure is declared")
    map_str_double var_mad_toler; //NESI(default="()",help="per-layer custom maximum maximum-absolute-differences over which a failure is declared (overrides mad_toler per-layer if specified")
    uint32_t max_err; //NESI(default="10",help="print at most this many differing elems")

    rtc_codegen_t codegen;
    rtc_codegen_t codegen_comp;

    virtual void main( nesi_init_arg_t * nia );
  };

  void add_codegen_annotations( p_conv_op_base_t const & anno_op, op_tune_t const & op_tune, 
                                map_str_op_tune_t const *per_op_tune ) {
    if( anno_op->is( Convolution_coi ) ) {
      if( op_tune.use_culibs ) { 
        anno_op->type = "cudnn_conv";
        must_insert( anno_op->str_vals, "conv_has_relu", str(1) );
      } else { 
        add_cnn_codegen_annotations( anno_op.get(), op_tune, per_op_tune ); 
        anno_op->type = must_find( anno_op->str_vals, "cts" );
        must_insert( anno_op->str_vals, "conv_has_relu", str(1) );
      }
    } else if( anno_op->is( sgemm_coi ) ) {
      if( op_tune.use_culibs ) {
        anno_op->type = "cublas_sgemm";
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
        must_insert( anno_op->dims_vals, "work", work );
        must_insert( anno_op->str_vals, "use_local_mem", str(op_tune.use_local_mem) );
        must_insert( anno_op->str_vals, "prof_variant", str(op_tune.prof_variant) );
        must_insert( anno_op->str_vals, "vw", str(op_tune.vw) );
        if( op_tune.prof_variant ) { 
          anno_op->type = "sgemm_prof";
        } else {
          if( op_tune.use_local_mem == 0 ) { anno_op->type = "sgemm_no_local"; }
          if( op_tune.use_local_mem == 2 ) { anno_op->type = "sgemm_simd"; }
          if( op_tune.use_local_mem == 3 ) { anno_op->type = "sgemm_simd_local"; }
        }
      }	  
    }
  }

  double profile_rcg_call( p_op_base_t const & anno_op, rtc_codegen_t & codegen,
			   p_op_base_t const & in_gen_op, map_str_p_nda_t * const outs, uint32_t const & run_iter );

  void cnn_op_info_t::main( nesi_init_arg_t * nia ) {
    vect_p_conv_op_t sigs;
    p_vect_string in_lines = readlines_fn( cnn_func_sigs_fn );
    p_ostream out = out_fn ? ofs_open( *out_fn ) : p_ostream( &std::cout, null_deleter<std::ostream>() );
    p_ostream oit_out = op_info_tab_fn ? ofs_open( *op_info_tab_fn ) : 0;
    p_ostream oet_out = op_eff_tab_fn ? ofs_open( *op_eff_tab_fn ) : 0;

    rtc->init(); codegen.init( rtc, compile_opts );
    if( rtc_comp ) { rtc_comp->init(); codegen_comp.init( rtc_comp, compile_opts ); }
    bool const enable_prof = 0;
    if( enable_prof ) { rtc->profile_start(); if(rtc_comp) { rtc_comp->init(); } }
    p_map_str_p_nda_t vs1;
    p_map_str_p_nda_t vs2;
    if( rtc_comp ) { 
      vs1 = make_shared<map_str_p_nda_t>();
      vs2 = make_shared<map_str_p_nda_t>();
    }
    uint32_t num_mad_fail = 0;
    for( vect_string::const_iterator i = in_lines->begin(); i != in_lines->end(); ++i ) {
      p_conv_op_base_t op = make_p_conv_op_base_t_init_and_check_unused_from_lexp( parse_lexp( *i ), 0 );
      op->set_and_check_coi();
      conv_op_info_to_latex_t to_latex;
      to_latex.init( op, print_format, inc_op_info_in_eff );
      if( op_tune.prof_variant ) { to_latex.emit_bw = 1; } // HACK; see comment in conv_op_info_to_latex_t
      if( oit_out ) { to_latex.info_row( oit_out.get() ); }

      if( rtc_comp ) { vs1->clear(); vs2->clear(); }
      // create rtc op
      p_conv_op_base_t anno_op = make_shared<conv_op_base_t>( *op );
      p_conv_op_base_t anno_op_comp = make_shared<conv_op_base_t>( *op );
      // generate boda variant according to tuning params (just opt and t_tile_sz currently)
      add_codegen_annotations( anno_op, op_tune, &per_op_tune );        
      if( rtc_comp ) { add_codegen_annotations( anno_op_comp, op_tune_comp, 0 ); }

      // for now, make generation only dependent on orig op type; this is convenient currently, but won't work well
      // if/when dealing with operations that require alternate data formats. maybe in those cases we'll need to deal
      // with auto-generating the need conversion functions anyway ...
      if( gen_data ) { gen_data->type = "gen_data_" + op->type; } 

      double rfc_dur_secs = NAN;
      string err;
      try { rfc_dur_secs = profile_rcg_call( anno_op, codegen, gen_data, vs1.get(), run_iter ) / 1000.0; }
      catch( rt_exception const & rte ) {
        if( rte.what_and_stacktrace().find( "CL_OUT_OF_HOST_MEMORY" ) != string::npos ) { 
          err = "CL_OUT_OF_HOST_MEMORY"; 
          codegen.clear();
        }
        else { throw; }
      }
      double rfc_dur_secs_comp = NAN;
      // (*out) << printf( "rfc_dur_secs=%s\n", str(rfc_dur_secs).c_str() );
      if( err.empty() && rtc_comp ) {
        rfc_dur_secs_comp = profile_rcg_call( anno_op_comp, codegen_comp, gen_data, vs2.get(), 1 ) / 1000.0;
        vect_string const vns1 = get_keys( *vs1 );
        vect_string const vns2 = get_keys( *vs2 );
        if( vns1 != vns2 ) { rt_err( strprintf( "reg/comp out var set mismatch: vns1=%s vns2=%s\n", 
                                                str(vns1).c_str(), str(vns2).c_str() ) ); }
        comp_vars( out.get(), num_mad_fail, mad_toler, &var_mad_toler, 0, max_err, vns1, vs1, vs2 );
      }
      if( oet_out ) { to_latex.eff_row( oet_out.get(), anno_op->type, rfc_dur_secs, peak_flops, rfc_dur_secs_comp ); }

    }

    if( enable_prof ) { rtc->profile_stop(); if( rtc_comp ) { rtc_comp->profile_stop(); } }
    rtc->finish_and_sync(); 
    if( rtc_comp ) { rtc_comp->finish_and_sync(); }

    if( !num_mad_fail ) { (*out) << strprintf( "***ALL IS WELL***\n" ); }
    else { (*out) << strprintf( "***MAD FAILS*** num_mad_fail=%s\n", str(num_mad_fail).c_str() ); }

  }

  struct cnn_prof_t : virtual public nesi, public has_main_t // NESI(help="profile set of rtc functions",
		      // bases=["has_main_t"], type_id="cnn_prof" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    op_tune_t op_tune; //NESI(default="()",help="tuning parameters / options")

    filename_t cnn_func_sigs_fn; //NESI(default="%(boda_test_dir)/cnn_func_sigs_tiny.txt",help="file to read cnn ops from")
    filename_t rtc_func_sigs_fn; //NESI(default="%(boda_output_dir)/cnn_rtc_func_sigs.txt",help="file to hold all generated rtc func signatures for the input cnn ops")
    virtual void main( nesi_init_arg_t * nia );
  };

  void cnn_prof_t::main( nesi_init_arg_t * nia ) {
    vect_p_conv_op_t sigs;
    p_ofstream out = ofs_open( rtc_func_sigs_fn );
    p_vect_string in_lines = readlines_fn( cnn_func_sigs_fn );

    for( vect_string::const_iterator i = in_lines->begin(); i != in_lines->end(); ++i ) {
      p_conv_op_base_t op = make_p_conv_op_base_t_init_and_check_unused_from_lexp( parse_lexp( *i ), 0 );
      op->set_and_check_coi();
      add_cnn_codegen_annotations( op.get(), op_tune, 0 );
      op->type = must_find( op->str_vals, "cts" );
      must_insert( op->str_vals, "conv_has_relu", str(1) );
      (*out) << str( *op ) << "\n";
    }
  }

#include"gen/cnn-prof.cc.nesi_gen.cc"

}
