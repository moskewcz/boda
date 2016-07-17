// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"timers.H"
#include<boost/filesystem.hpp>
#include<boost/lexical_cast.hpp>
#include"lexp.H"
#include"has_main.H"
#include"rtc_func_gen.H"
#include"rtc_compute.H"
#include"cnn_op.H"

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

  double profile_rcg_call( p_op_base_t const & anno_op, rtc_codegen_t & codegen, bool const & show_rtc_calls,
			   p_op_base_t const & in_gen_op_orig, map_str_p_nda_t * const outs,
                           uint32_t const & run_iter ) 
  {
    timer_t t("profile_rcg_call");
    // FIXME: pass these? fix some other way? at least group them into some opts sturct?
    bool const show_compile_log = 0;
    bool const enable_lineinfo = 0;
    bool const show_func_attrs = 0;

    string const func_name = codegen.gen_func( make_cnn_custom_codegen_t().get(), *anno_op );
    p_rtc_call_gen_t const &rcg = must_find( codegen.rtc_func_names_map, func_name );
#if 0
    if( (!rcg->blks) && (!op_tune.use_culibs) ) { 
      assert_st(0);
      //(*out) << strprintf( "skipping %s; dynamic block sizes todo\n", str(rcg->type).c_str() );
      //return 0.0;
    }
#endif
    if( (rcg->type == "quantize") || (rcg->type == "dropout") ) {
      assert_st(0);
      //(*out) << strprintf( "skipping %s; u32 arg handling todo\n", str(rcg->type).c_str() );
      //return 0.0; 
    }
    codegen.compile( show_compile_log, enable_lineinfo, show_func_attrs );

    map_str_str arg_map;
    for( vect_arg_decl_t::multi_iter i = rcg->rtc_func_template->arg_decls.multi_begin( rcg.get() ); !i.at_end(); ++i ) {
      if( i.ad().io_type == "REF" ) { continue; }
      dims_t const & func_dims = rcg->get_arg_dims_by_name( i.vn() );
      if( func_dims == dims_t() ) { continue; } // NULL case -- ignore
      must_insert( arg_map, i.vn(), i.vn() );
    }
    //printf( "run: i->rtc_func_name=%s\n", str(rcg->gen_fn).c_str() );
    for( map_str_str::const_iterator j = arg_map.begin(); j != arg_map.end(); ++j ) {
      codegen.rtc->create_var_with_dims( j->second, must_find( anno_op->dims_vals, j->first ) );
    }
    vect_string xpose_vars_to_release;
    if( in_gen_op_orig ) { 
      for( vect_arg_decl_t::multi_iter i = rcg->rtc_func_template->arg_decls.multi_begin( rcg.get() ); !i.at_end(); ++i ) {
        p_op_base_t in_gen_op = make_shared<op_base_t>( *in_gen_op_orig );
	if( i.ad().io_type != "IN" ) { continue; }
	in_gen_op->type += "_" + i.vn();
	in_gen_op->dims_vals.clear();
        dims_t const & in_dims = must_find( anno_op->dims_vals, i.vn() );
        dims_t const & ref_in_dims = get( anno_op->dims_vals, i.vn() + "_ref", in_dims );
	must_insert( in_gen_op->dims_vals, i.vn(), ref_in_dims );
        string gen_vn = i.vn();
        if( in_dims != ref_in_dims ) { 
          gen_vn += "_ref"; 
          codegen.rtc->create_var_with_dims( gen_vn, ref_in_dims ); 
          xpose_vars_to_release.push_back( gen_vn );
        }
	string const in_gen_func_name = codegen.gen_func( make_cnn_custom_codegen_t().get(), *in_gen_op );
        codegen.compile( show_compile_log, enable_lineinfo, show_func_attrs );
	rcg_func_call_t rfc_in_gen{ in_gen_func_name, "tag", map_str_str{{i.vn(),gen_vn}} };
	codegen.run_rfc( show_rtc_calls, rfc_in_gen, 0 );
        // check if xpose needed:
        if( gen_vn != i.vn() ) {
          // FIXME: some ugly, cut-n-paste, brittle stuff here ... but it's pending more global cleanup.
          string xpose_op = anno_op->type+"_xpose_"+i.vn();
          // FIXME: sigh.
          if( ( i.vn() == "filts" ) && is_k1_or_t_or_reg_conv(get( anno_op->str_vals, "cts", "" ))) { xpose_op = "xpose_filts"; }
          string const xpose_func = codegen.gen_func( make_cnn_custom_codegen_t().get(), 
                                                      op_base_t{ xpose_op, anno_op->dims_vals, anno_op->str_vals } );
          codegen.compile( show_compile_log, enable_lineinfo, show_func_attrs );
          rcg_func_call_t rfc_in_gen_xpose{ xpose_func, "tag", map_str_str{{gen_vn,gen_vn},{i.vn(),i.vn()}} };
          codegen.run_rfc( show_rtc_calls, rfc_in_gen_xpose, 0 );
        }
	//if( outs ) { must_insert( *outs, i.vn(), p_nda_float_t() ); } // include inputs in 'outputs'
      }
    }
    rcg_func_call_t rfc{ rcg->gen_fn, "tag", arg_map };
    for( uint32_t i = 0; i != run_iter; ++i ) { codegen.run_rfc( show_rtc_calls, rfc, 0 ); }

    // FIXME: xpose of OUTs is semi-dup'd with "IN"/gen_data handling above
    for( vect_arg_decl_t::multi_iter i = rcg->rtc_func_template->arg_decls.multi_begin( rcg.get() ); !i.at_end(); ++i ) {
      if( !endswith( i.ad().io_type, "OUT" ) ) { continue; }
      dims_t const & out_dims = must_find( anno_op->dims_vals, i.vn() );
      dims_t const & ref_out_dims = get( anno_op->dims_vals, i.vn() + "_ref", out_dims );
      string gen_vn = i.vn();
      if( out_dims != ref_out_dims ) { 
        gen_vn += "_ref"; 
        codegen.rtc->create_var_with_dims( gen_vn, ref_out_dims ); 
        xpose_vars_to_release.push_back( gen_vn );
      }
      // check if xpose needed:
      if( gen_vn != i.vn() ) {
        // FIXME: some ugly, cut-n-paste, brittle stuff here ... but it's pending more global cleanup.
        string xpose_op = anno_op->type+"_xpose_"+i.vn();
        string const xpose_func = codegen.gen_func( make_cnn_custom_codegen_t().get(), 
                                                    op_base_t{ xpose_op, anno_op->dims_vals, anno_op->str_vals } );
        codegen.compile( show_compile_log, enable_lineinfo, show_func_attrs );
        rcg_func_call_t rfc_in_gen_xpose{ xpose_func, "tag", map_str_str{{gen_vn,gen_vn},{i.vn(),i.vn()}} };
        codegen.run_rfc( show_rtc_calls, rfc_in_gen_xpose, 0 );
      }
      if( outs ) { must_insert( *outs, i.vn(), codegen.rtc->create_nda_from_var( gen_vn ) ); } 
    }
    for( map_str_str::const_iterator j = arg_map.begin(); j != arg_map.end(); ++j ) {
      codegen.rtc->release_var( j->second );
    }
    for( vect_string::const_iterator i = xpose_vars_to_release.begin(); i != xpose_vars_to_release.end(); ++i ) {
      codegen.rtc->release_var( *i );
    }
    codegen.clear();
    // get call duration
    //if( rfc.call_tag.empty() ) { release; return; } // FIXME: possible here? 
    codegen.rtc->finish_and_sync();
    double const rfc_dur = codegen.rtc->get_dur( rfc.call_id, rfc.call_id );
    codegen.rtc->release_per_call_id_data();
    return rfc_dur;
  }

  p_op_base_t make_p_op_base_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );

  void rtc_prof_t::main( nesi_init_arg_t * nia ) {
    out = ofs_open( per_call_fn );
    rtc->init(); codegen.init( rtc );
    bool const enable_prof = 0;
    if( enable_prof ) { rtc->profile_start(); }
    if( eat_megs ) { rtc->create_var_with_dims( "MEMEATER", dims_t{ {1024,1024,eat_megs}, {"a","b","M"}, "float" } ); }

    p_vect_string in_lines = readlines_fn( rtc_func_sigs_fn );
    for( vect_string::const_iterator i = in_lines->begin(); i != in_lines->end(); ++i ) {
      p_op_base_t v = make_p_op_base_t_init_and_check_unused_from_lexp( parse_lexp( *i ), 0 );
      double const rfc_dur = profile_rcg_call( v, codegen, show_rtc_calls, 0, 0, 1 );
      (*out) << strprintf( "per_layer_time['tag']=per_layer_time.get('tag',0.0) + %s\n", str(rfc_dur/1000.0).c_str() );
    }
    if( enable_prof ) { rtc->profile_stop(); }
    rtc->finish_and_sync();
  }

#include"gen/rtc_prof.cc.nesi_gen.cc"

}
