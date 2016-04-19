// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"timers.H"
#include<boost/filesystem.hpp>
#include<boost/lexical_cast.hpp>
#include"lexp.H"
#include"has_main.H"
#include"rtc_func_gen.H"
#include"rtc_compute.H"

namespace boda 
{
  typedef shared_ptr< dims_t > p_dims_t; 

  struct rtc_prof_t : virtual public nesi, public has_main_t // NESI(help="profile set of rtc functions",
		      // bases=["has_main_t"], type_id="rtc_prof" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    uint32_t show_compile_log; //NESI(default=0,help="if 1, print compilation log")
    uint32_t show_rtc_calls; //NESI(default=0,help="if 1, print rtc calls")
    uint32_t enable_lineinfo; //NESI(default=0,help="if 1, enable lineinfo for ptx compilation")
    uint32_t show_func_attrs; //NESI(default=0,help="if 1, print func attrs after load")
    uint32_t eat_megs; //NESI(default=0,help="if non-zero, allocate unused var of size eat_mega Mfloats via rtc")
    filename_t rtc_func_sigs_fn; //NESI(default="%(boda_test_dir)/rtc_func_sigs_tiny.txt",help="file to hold all generated func signatures")
    p_dims_t dummy_dims; // NESI(help="HACK: dummy NESI var of type dims_t (otherwise unused) to force tinfo generation. see map_str_T FIXME in nesi.cc")

    p_rtc_compute_t rtc; //NESI(default="(be=ocl)",help="rtc back-end to use")
    filename_t per_call_fn; //NESI(default="%(boda_output_dir)/rtc_prof.py",help="if non-empty, write per-call profiling (timing via events) to given file.")

    vect_rcg_func_call_t calls;
    // rtc->create_var_with_dims_floats( name, cp->must_get_node(node_name)->dims );
    // calls.push_back( rcg_func_call_t{ gen_fn, oi->tag, oi->arg_map } );
    
    p_ofstream out;
    rtc_codegen_t codegen;

    virtual void main( nesi_init_arg_t * nia );

    string gen_func( op_base_t const & rfs );
    double run_call( string const & func_name, p_rtc_call_gen_t const & rcg );
    void run_calls( void );
  };
 
  string rtc_prof_t::gen_func( op_base_t const & rfs ) { 
    p_custom_codegen_t ccc = make_cnn_custom_codegen_t();
    return codegen.gen_func( ccc.get(), rfs ); 
  }

  double profile_rcg_call( p_rtc_compute_t const & rtc, rtc_codegen_t & codegen, bool const & show_rtc_calls,
			   p_rtc_call_gen_t const & rcg, 
			   p_op_base_t const & in_gen_op, map_str_p_nda_float_t * const outs ) 
  {
    timer_t t("profile_rcg_call");
    map_str_str arg_map;
    for( vect_arg_decl_t::const_iterator i = rcg->flat_arg_decls.begin(); i != rcg->flat_arg_decls.end(); ++i ) {
      if( i->io_type == "REF" ) { continue; }
      dims_t const & func_dims = rcg->get_arg_dims_by_name( i->vn );
      if( func_dims == dims_t() ) { continue; } // NULL case -- ignore
      must_insert( arg_map, i->vn, i->vn );
      if( outs && (endswith( i->io_type, "OUT" )) ) { must_insert( *outs, i->vn, p_nda_float_t() ); }
    }
    printf( "run: i->rtc_func_name=%s\n", str(rcg->gen_fn).c_str() );
    bool const show_compile_log = 0; bool const enable_lineinfo = 0; bool const show_func_attrs = 0;
    for( map_str_str::const_iterator j = arg_map.begin(); j != arg_map.end(); ++j ) {
      rtc->create_var_with_dims_floats( j->second, must_find( rcg->dims_vals, j->first ) );
    }
    rtc->compile( rcg->rtc_prog_str, show_compile_log, enable_lineinfo, {rcg->gen_fn}, show_func_attrs );
    if( in_gen_op ) { 
      for( vect_arg_decl_t::const_iterator i = rcg->flat_arg_decls.begin(); i != rcg->flat_arg_decls.end(); ++i ) {
	if( i->io_type != "IN" ) { continue; }

	in_gen_op->type = "gen_data_" + rcg->type + "_" + i->vn;
	in_gen_op->dims_vals.clear();
	must_insert( in_gen_op->dims_vals, i->vn, must_find( rcg->dims_vals, i->vn ) );

	string const in_gen_func_name = codegen.gen_func( make_cnn_custom_codegen_t().get(), *in_gen_op );
	p_rtc_call_gen_t const & rcg_in_gen = must_find( codegen.rtc_func_names_map, in_gen_func_name );

	rtc->compile( rcg_in_gen->rtc_prog_str, show_compile_log, enable_lineinfo, {rcg_in_gen->gen_fn}, show_func_attrs );
	rcg_func_call_t rfc_in_gen{ rcg_in_gen->gen_fn, "tag", arg_map };
	codegen.run_rfc( rtc, show_rtc_calls, rfc_in_gen, 0 );
      }
    }
    rcg_func_call_t rfc{ rcg->gen_fn, "tag", arg_map };
    codegen.run_rfc( rtc, show_rtc_calls, rfc, 0 );
    if( outs ) {
      for( map_str_p_nda_float_t::iterator j = outs->begin(); j != outs->end(); ++j ) {
	j->second = rtc->create_nda_from_var( j->first );
      }
    }
    for( map_str_str::const_iterator j = arg_map.begin(); j != arg_map.end(); ++j ) {
      rtc->release_var( j->second );
    }
    rtc->release_all_funcs();
    // get call duration
    //if( rfc.call_tag.empty() ) { release; return; } // FIXME: possible here? 
    rtc->finish_and_sync();
    double const rfc_dur = rtc->get_dur( rfc.call_id, rfc.call_id );
    rtc->release_per_call_id_data();
    return rfc_dur;
  }

  p_op_base_t make_p_op_base_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );

  void rtc_prof_t::main( nesi_init_arg_t * nia ) {
    out = ofs_open( per_call_fn );
    rtc->init();
    bool const enable_prof = 0;
    if( enable_prof ) { rtc->profile_start(); }
    if( eat_megs ) { rtc->create_var_with_dims_floats( "MEMEATER", dims_t{ {1024,1024,eat_megs}, {"a","b","M"}, 1 } ); }

    p_vect_string in_lines = readlines_fn( rtc_func_sigs_fn );
    for( vect_string::const_iterator i = in_lines->begin(); i != in_lines->end(); ++i ) {
      p_op_base_t v = make_p_op_base_t_init_and_check_unused_from_lexp( parse_lexp( *i ), 0 );
      codegen.gen_func( make_cnn_custom_codegen_t().get(), *v );
      assert( codegen.rtc_func_names_map.size() == 1 );
      p_rtc_call_gen_t const &rcg = codegen.rtc_func_names_map.begin()->second;
      string const & func_name = codegen.rtc_func_names_map.begin()->first;
      if( !rcg->blks ) { 
	printf( "skipping %s; dynamic block sizes todo\n", str(rcg->type).c_str() );
	continue; 
      }
      if( (rcg->type == "quantize") || (rcg->type == "dropout") ) {
	printf( "skipping %s; u32 arg handling todo\n", str(rcg->type).c_str() );
	continue; 
      }
      double const rfc_dur = profile_rcg_call( rtc, codegen, show_rtc_calls, rcg, 0, 0 );
      (*out) << strprintf( "per_layer_time['tag']=per_layer_time.get('tag',0.0) + %s # %s \n", 
			   str(rfc_dur/1000.0).c_str(), func_name.c_str() );
      codegen.rtc_func_names_map.clear();
      codegen.rtc_prog_str.clear();
    }
    if( enable_prof ) { rtc->profile_stop(); }
    rtc->finish_and_sync();
  }

#include"gen/rtc_prof.cc.nesi_gen.cc"

}
