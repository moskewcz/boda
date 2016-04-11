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

    p_rtc_compute_t rtc; //NESI(default="(be=nvrtc)",help="rtc back-end to use")
    filename_t per_call_fn; //NESI(default="%(boda_output_dir)/rtc_prof.py",help="if non-empty, write per-call profiling (timing via events) to given file.")

    vect_rcg_func_call_t calls;
    // rtc->create_var_with_dims_floats( name, cp->must_get_node(node_name)->dims );
    // calls.push_back( rcg_func_call_t{ gen_fn, oi->tag, oi->arg_map } );
    
    rtc_codegen_t codegen;

    virtual void main( nesi_init_arg_t * nia );

    string gen_func( rtc_func_sig_t const & rfs );
    void run_rfc( rcg_func_call_t & rfc );
    void run_call( string const & func_name, p_rtc_call_gen_t const & rcg );
    void run_calls( void );
  };
 
  string rtc_prof_t::gen_func( rtc_func_sig_t const & rfs ) { 
    p_custom_codegen_t ccc = make_cnn_custom_codegen_t();
    return codegen.gen_func( ccc.get(), rfs ); 
  }

  p_ofstream out;
  void rtc_prof_t::main( nesi_init_arg_t * nia ) {
    out = ofs_open( per_call_fn );
    rtc->init();
    bool const enable_prof = 0;
    if( enable_prof ) { rtc->profile_start(); }
    if( eat_megs ) { rtc->create_var_with_dims_floats( "MEMEATER", dims_t{ {1024,1024,eat_megs}, {"a","b","M"}, 1 } ); }
    codegen.read_rtc_func_sigs( rtc_func_sigs_fn );
    for( rtc_func_names_map_t::iterator i = codegen.rtc_func_names_map.begin(); i != codegen.rtc_func_names_map.end(); ++i ) { 
      p_rtc_call_gen_t const &rcg = i->second;
      if( !rcg->blks ) { 
	printf( "skipping %s; dynamic block sizes todo\n", str(rcg->type).c_str() );
	continue; 
      }
      if( (rcg->type == "quantize") || (rcg->type == "dropout") ) {
	printf( "skipping %s; u32 arg handling todo\n", str(rcg->type).c_str() );
	continue; 
      }
      run_call( i->first, rcg );
    }
    if( enable_prof ) { rtc->profile_stop(); }
    rtc->finish_and_sync();
  }

  void rtc_prof_t::run_rfc( rcg_func_call_t & rfc ) { codegen.run_rfc( rtc, show_rtc_calls, rfc, 0 );  }

  void rtc_prof_t::run_call( string const & func_name, p_rtc_call_gen_t const & rcg ) {
    timer_t t("rtc_prof_t::run_fwd");
    map_str_str arg_map;
    for( vect_arg_decl_t::const_iterator i = rcg->flat_arg_decls.begin(); i != rcg->flat_arg_decls.end(); ++i ) {
      if( i->io_type == "REF" ) { continue; }
      dims_t const & func_dims = rcg->get_arg_dims_by_name( i->vn );
      if( func_dims == dims_t() ) { continue; } // NULL case -- ignore
      must_insert( arg_map, i->vn, i->vn );
    }
    printf( "run: i->rtc_func_name=%s\n", str(func_name).c_str() );
    rtc->compile( rcg->rtc_prog_str, show_compile_log, enable_lineinfo, {func_name}, show_func_attrs );
    for( map_str_str::const_iterator j = arg_map.begin(); j != arg_map.end(); ++j ) {
      rtc->create_var_with_dims_floats( j->second, must_find( rcg->dims_vals, j->first ) );
    }
    rcg_func_call_t rfc{ func_name, "tag", arg_map };
    run_rfc( rfc ); 
    for( map_str_str::const_iterator j = arg_map.begin(); j != arg_map.end(); ++j ) {
      rtc->release_var( j->second );
    }
    rtc->release_all_funcs();
    // get call duration
    //if( rfc.call_tag.empty() ) { release; return; } // FIXME: possible here? 
    float const rfc_dur = rtc->get_dur( rfc.call_id, rfc.call_id );
    (*out) << strprintf( "per_layer_time['%s']=per_layer_time.get('%s',0.0) + %s # %s \n", 
			 str(rfc.call_tag).c_str(), str(rfc.call_tag).c_str(), str(rfc_dur/1000.0).c_str(), rfc.rtc_func_name.c_str() );
    rtc->release_per_call_id_data();
  }

#include"gen/rtc_prof.cc.nesi_gen.cc"

}
