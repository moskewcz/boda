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
    rtc_compile_opts_t compile_opts; // NESI(default="()",help="runtime compilation options")
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

    double run_call( string const & func_name, p_rtc_call_gen_t const & rcg );
    void run_calls( void );
  };

  // semi-dupe'd with rtc_fwd gen_apply_func_to_var(). working toward convergence. note that in this use model, the
  // input and output variable names and arg names happen to be the same, hence the 'an_and_vn' arguments to this func.
  void run_xpose( p_op_base_t const & anno_op, rtc_codegen_t & codegen, string const & xpose_func_name, 
                  string const &out_an_and_vn, string const &in_an_and_vn )  {
    p_rtc_call_gen_t xpose_func = codegen.gen_func_override_func_name( xpose_func_name, *anno_op );
    rcg_func_call_t rcg{ xpose_func, "tag", map_str_str{{out_an_and_vn,out_an_and_vn},{in_an_and_vn,in_an_and_vn}} };
    codegen.run_func( rcg );
  }
  
  double profile_rcg_call( p_op_base_t const & anno_op, rtc_codegen_t & codegen,
			   p_op_base_t const & in_gen_op_orig, map_str_p_nda_t * const outs,
                           uint32_t const & run_iter ) 
  {
    timer_t t("profile_rcg_call");
    string const anno_op_func_name = anno_op->get_func_name();
    p_rtc_call_gen_t rcg = codegen.gen_func( *anno_op );

    map_str_str arg_map;
    for( vect_arg_decl_t::multi_iter i = rcg->rtc_func_template->arg_decls.multi_begin( &rcg->op ); !i.at_end(); ++i ) {
      if( i.ad().io_type == "REF" ) { continue; }
      if( i.vn() == "cucl_arg_info" ) { continue; } // FIXME: not-too-nice special case for cucl_arg_info argument 
      dims_t const & func_dims = rcg->get_arg_dims_by_name( i.vn() );
      if( func_dims == make_null_dims_t() ) { continue; } // NULL case -- ignore
      must_insert( arg_map, i.vn(), i.vn() );
    }
    //printf( "run: i->rtc_func_name=%s\n", str(rcg->gen_fn).c_str() );
    for( map_str_str::const_iterator j = arg_map.begin(); j != arg_map.end(); ++j ) {
      codegen.rtc->create_var_with_dims( j->second, anno_op->get_dims( j->first ) );
    }
    vect_string xpose_vars_to_release;
    if( in_gen_op_orig ) { 
      for( vect_arg_decl_t::multi_iter i = rcg->rtc_func_template->arg_decls.multi_begin( &rcg->op ); !i.at_end(); ++i ) {
        p_op_base_t in_gen_op = make_shared<op_base_t>( *in_gen_op_orig );
	if( i.ad().io_type != "IN" ) { continue; }
        if( i.vn() == "cucl_arg_info" ) { continue; } // FIXME: not-too-nice special case for cucl_arg_info argument 
        // note: gen_data variant choice based on gen type and op type (*not* op func_name)
	in_gen_op->set_func_name( in_gen_op->get_type()+"_"+anno_op->get_type()+"_"+i.vn() ); 
	in_gen_op->nda_vals.clear();
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
	p_rtc_call_gen_t in_gen_func = codegen.gen_func( *in_gen_op );
	rcg_func_call_t rfc_in_gen{ in_gen_func, "tag", map_str_str{{i.vn(),gen_vn}} };
	codegen.run_func( rfc_in_gen );
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

    rcg_func_call_t rfc{ rcg, "tag", arg_map }; 
    for( uint32_t i = 0; i != run_iter; ++i ) { codegen.run_func( rfc ); }

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
    for( map_str_str::const_iterator j = arg_map.begin(); j != arg_map.end(); ++j ) {
      codegen.rtc->release_var( j->second );
    }
    for( vect_string::const_iterator i = xpose_vars_to_release.begin(); i != xpose_vars_to_release.end(); ++i ) {
      codegen.rtc->release_var( *i );
    }
    // get call duration
    //if( rfc.call_tag.empty() ) { release; return; } // FIXME: possible here? 
    codegen.rtc->finish_and_sync();
    double const rfc_dur = codegen.rtc->get_dur( rfc.call_id, rfc.call_id );
    codegen.rtc->release_per_call_id_data();
    rcg.reset(); // optional. allows just-used function (which is no longer needed) to be released now if func-gc happens.
    codegen.gc_clear();
    return rfc_dur;
  }

  p_op_base_t make_p_op_base_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );

  void rtc_prof_t::main( nesi_init_arg_t * nia ) {
    out = ofs_open( per_call_fn );
    rtc->init(); codegen.init( rtc, make_cnn_custom_codegen_t(), compile_opts );
    bool const enable_prof = 0;
    if( enable_prof ) { rtc->profile_start(); }
    if( eat_megs ) { rtc->create_var_with_dims( "MEMEATER", dims_t{ {1024,1024,eat_megs}, {"a","b","M"}, "float" } ); }

    p_vect_string in_lines = readlines_fn( rtc_func_sigs_fn );
    for( vect_string::const_iterator i = in_lines->begin(); i != in_lines->end(); ++i ) {
      p_op_base_t v = make_p_op_base_t_init_and_check_unused_from_lexp( parse_lexp( *i ), 0 );
      double const rfc_dur = profile_rcg_call( v, codegen, 0, 0, 1 );
      (*out) << strprintf( "per_layer_time['tag']=per_layer_time.get('tag',0.0) + %s\n", str(rfc_dur/1000.0).c_str() );
    }
    if( enable_prof ) { rtc->profile_stop(); }
    rtc->finish_and_sync();
  }

#include"gen/rtc_prof.cc.nesi_gen.cc"

}
