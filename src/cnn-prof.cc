// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"latex-util.H"
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
  p_conv_op_base_t make_p_conv_op_base_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );



  struct cnn_op_info_t : virtual public nesi, public has_main_t // NESI(help="print info for set of CNN operations",
			 // bases=["has_main_t"], type_id="cnn_op_info" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    rtc_compile_opts_t compile_opts; // NESI(default="()",help="runtime compilation options")
    filename_t cnn_func_sigs_fn; //NESI(default="%(boda_test_dir)/conv-ops-tiny.txt",help="file to read cnn ops from")
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
    p_rtc_compute_t rtc; //NESI(default="(be=nvrtc)",help="rtc back-end to use")

    // comparison testing related options:
    p_rtc_compute_t rtc_comp; //NESI(help="rtc back-end to use for correctness-testing comparison")
    double mrd_toler; //NESI(default="2e-4",help="maximum maximum-absolute-difference over which a failure is declared")
    map_str_double var_mrd_toler; //NESI(default="()",help="per-layer custom maximum maximum-absolute-differences over which a failure is declared (overrides mrd_toler per-layer if specified")
    uint32_t max_err; //NESI(default="10",help="print at most this many differing elems")

    rtc_codegen_t codegen;
    rtc_codegen_t codegen_comp;

    virtual void main( nesi_init_arg_t * nia );
  };

  void cnn_op_info_t::main( nesi_init_arg_t * nia ) {
    vect_p_conv_op_t sigs;
    p_vect_string in_lines = readlines_fn( cnn_func_sigs_fn );
    p_ostream out = out_fn ? ofs_open( *out_fn ) : p_ostream( &std::cout, null_deleter<std::ostream>() );
    p_ostream oit_out = op_info_tab_fn ? ofs_open( *op_info_tab_fn ) : 0;
    p_ostream oet_out = op_eff_tab_fn ? ofs_open( *op_eff_tab_fn ) : 0;

    rtc->init(); codegen.init( rtc, make_cnn_custom_codegen_t(), compile_opts );
    if( rtc_comp ) { rtc_comp->init(); codegen_comp.init( rtc_comp, make_cnn_custom_codegen_t(), compile_opts ); }
    bool const enable_prof = 0;
    if( enable_prof ) { rtc->profile_start(); if(rtc_comp) { rtc_comp->profile_start(); } }
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
      to_latex.init( op, print_format, inc_op_info_in_eff, 1 );
      if( op_tune.prof_variant ) { to_latex.emit_bw = 1; } // HACK; see comment in conv_op_info_to_latex_t
      if( oit_out ) { to_latex.info_row( oit_out.get(), 0 ); } // note: always use non-brief info here

      if( rtc_comp ) { vs1->clear(); vs2->clear(); }
      // create rtc op
      p_conv_op_base_t anno_op = make_shared<conv_op_base_t>( *op );
      p_conv_op_base_t anno_op_comp = make_shared<conv_op_base_t>( *op );
      // generate boda variant according to tuning params (just opt and t_tile_sz currently)
      add_codegen_annotations( anno_op, op_tune, &per_op_tune );        
      if( rtc_comp ) { add_codegen_annotations( anno_op_comp, op_tune_comp, 0 ); }

      if( gen_data ) { assert_st( gen_data->get_type() == "gen_data" ); } // FIXME: remove assert after fixing existing usages

      double rfc_dur_secs = NAN;
      string err;
      try { rfc_dur_secs = profile_rcg_call( anno_op, codegen, gen_data, vs1.get(), run_iter, 0 ).rt_secs; }
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
      double rfc_dur_secs_comp = NAN;
      // (*out) << printf( "rfc_dur_secs=%s\n", str(rfc_dur_secs).c_str() );
      if( err.empty() && rtc_comp ) {
        rfc_dur_secs_comp = profile_rcg_call( anno_op_comp, codegen_comp, gen_data, vs2.get(), 1, 0 ).rt_secs;
        vect_string const vns1 = get_keys( *vs1 );
        vect_string const vns2 = get_keys( *vs2 );
        if( vns1 != vns2 ) { rt_err( strprintf( "reg/comp out var set mismatch: vns1=%s vns2=%s\n", 
                                                str(vns1).c_str(), str(vns2).c_str() ) ); }
        (*out) << strprintf( "vars_to_compare: %s\n", str(vns1).c_str() );
        comp_vars( out.get(), num_mad_fail, mrd_toler, &var_mrd_toler, 0, max_err, vns1, vs1, vs2 );
      }
      if( oet_out ) { to_latex.eff_row( oet_out.get(), anno_op->get_type(), rfc_dur_secs, peak_flops, rfc_dur_secs_comp ); }

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
    p_ostream out = ofs_open( rtc_func_sigs_fn );
    p_vect_string in_lines = readlines_fn( cnn_func_sigs_fn );

    for( vect_string::const_iterator i = in_lines->begin(); i != in_lines->end(); ++i ) {
      p_conv_op_base_t op = make_p_conv_op_base_t_init_and_check_unused_from_lexp( parse_lexp( *i ), 0 );
      op->set_and_check_coi();
      add_cnn_codegen_annotations( op.get(), op_tune, 0 );
      op->set_u32( "conv_has_relu", 1 );
      (*out) << str( *op ) << "\n";
    }
  }

#include"gen/cnn-prof.cc.nesi_gen.cc"

}
