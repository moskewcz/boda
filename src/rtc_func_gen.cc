// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"rtc_func_gen.H"

namespace boda 
{

  void insert_nda_dims_sz( map_str_str & mss, string const & nda_vn, dims_t const & dims, bool const & dims_only ) {
    assert_st( dims.valid() );
    must_insert( mss, nda_vn+"_tn", str(dims.tn) );
    for( uint32_t i = 0; i != dims.sz(); ++i ) {
      must_insert( mss, nda_vn+"_"+dims[i].name+"_dim", str(dims[i].sz) );
      if( !dims_only ) { 
	assert_st( dims[i].has_sz_and_stride_and_name() );
	must_insert( mss, nda_vn+"_"+dims[i].name+"_stride", str(dims[i].stride) );
      }
    }
    if( !dims_only ) { must_insert( mss, nda_vn+"_dims_prod", str(dims.dims_prod()) ); } // also emit dim(0)*stride(0)?
  }

  void insert_nda_ix_exprs( map_str_str & mss, string const & ix_vn, dims_t const & dims, string ix_expr ) {
    if( ix_expr.empty() ) { ix_expr = ix_vn; } // if no custom ix expression, use ix_vn directly as ix expr
    for( uint32_t i = 0; i != dims.sz(); ++i ) {
      assert_st( dims[i].has_sz_and_stride_and_name() );
      string v = (dims[i].stride > 1) ? "("+ix_expr+"/"+str(dims[i].stride)+")" : ix_expr;
      must_insert( mss, ix_vn+"_"+dims[i].name+"_nomod", v );      
      // note: we supress the module on the outermost dim, allowing it to overflow (even though we know it's size and
      // could wrap it. i guess if you want such wrapping, add another outmost dim, and you can set it's size to 1? then
      // the value of that dim becomes 0-if-valid, 1-or-more-if-OOB. but then there's unneeded modulos on the outmost
      // dim value, assuming OOB access is guarded against ... ehh, the current system seems okay.
      if( i ) { 
	uint32_t const dsz = dims[i].sz;
	assert_st( dsz );
	if( dsz > 1 ) { v = "("+v+"%%"+str(dims[i].sz)+")"; }
	else { v = "0"; }
      }
      must_insert( mss, ix_vn+"_"+dims[i].name, v );
    }
    must_insert( mss, ix_vn+"_dims_prod", str(dims.dims_prod()) ); // also emit dim(0)*stride(0)?
  }

  string dyn_dims( string const & vn, dims_t const & dims, uint32_t const & dix ) {
    return strprintf( "cucl_arg_info.%s_%s", vn.c_str(), dims[dix].name.c_str() );
  }

  void insert_nda_dyn_ix_exprs( map_str_str & mss, string const & ix_vn, dims_t const & dims, string ix_expr ) {
    if( ix_expr.empty() ) { ix_expr = ix_vn; } // if no custom ix expression, use ix_vn directly as ix expr
    for( uint32_t i = 0; i != dims.sz(); ++i ) {
      assert_st( dims[i].has_name() );
      string v = "("+ix_expr+"/"+dyn_dims(ix_vn,dims,i)+"_stride"+")";
      must_insert( mss, ix_vn+"_"+dims[i].name+"_nomod", v );      
      // note: we supress the module on the outermost dim, allowing it to overflow (even though we know it's size and
      // could wrap it. i guess if you want such wrapping, add another outmost dim, and you can set it's size to 1? then
      // the value of that dim becomes 0-if-valid, 1-or-more-if-OOB. but then there's unneeded modulos on the outmost
      // dim value, assuming OOB access is guarded against ... ehh, the current system seems okay.
      if( i ) { v = "("+v+"%%"+dyn_dims(ix_vn,dims,i)+"_dim"+")"; }
      must_insert( mss, ix_vn+"_"+dims[i].name, v );
    }
    must_insert( mss, ix_vn+"_dims_prod", "cucl_arg_info."+ix_vn+"_dims_prod" ); // also emit dim(0)*stride(0)?
  }

  dims_t dims_from_nda_spec( string const & tn, string const & nda_spec ) {
    vect_string const dim_names = split( nda_spec, ':' );
    dims_t arg_dims;
    // for now, assume template: (1) can handle any size for all nda dims (to relax, could allow spec of size,
    // then use that instead of 0/wild for sz below) (2) all dims are static (to relax: unclear what
    // syntax/encoding would be)
    for( vect_string::const_iterator i = dim_names.begin(); i != dim_names.end(); ++i ) { arg_dims.add_dims( *i, 0 ); }
    arg_dims.tn = tn; // note: may be empty string for wild type
    arg_dims.calc_strides(1);
    return arg_dims;
  }


  string gen_unused_fn( op_base_t const & op, set_string const & used_names ) {
    string maybe_fn_base = op.type;
    set_string unique_dims;
    for( map_str_dims_t::const_iterator ra = op.dims_vals.begin(); ra != op.dims_vals.end(); ++ra ) {
      dims_t const & dims = ra->second;
      for( uint32_t i = 0; i != dims.sz(); ++i ) {
	string const & dn = dims.names(i);
	bool const did_ins = unique_dims.insert( dn ).second;
	if( did_ins ) { maybe_fn_base += "__"+dn+"_"+str(dims.dims(i)); }
      }
    }
    // not the best/most-robust idea, but for now we can avoid most namegenconflicts by (questionally) stuffing the
    // str_vals into the function name as well. it'll be fine, right? this could be removed if problematic.
    for( map_str_str::const_iterator ra = op.str_vals.begin(); ra != op.str_vals.end(); ++ra ) {
      maybe_fn_base += "__"+ra->first+"_"+as_pyid_fixme(ra->second);
    }

    string maybe_fn = maybe_fn_base;
    uint32_t uix = 0;
    while( has( used_names, maybe_fn ) ) { ++uix; maybe_fn = maybe_fn_base + "__namegenconflict_" + str(uix); }
    return maybe_fn;
  }

  void rtc_call_gen_t::run_rfc( p_rtc_compute_t const & rtc, bool const & show_rtc_calls, 
				rcg_func_call_t & rcg_func_call, uint32_t const & flags ) {
    rtc_func_call_t rfc;
    rfc.rtc_func_name = rcg_func_call.func->gen_fn;
    rfc.call_tag = rcg_func_call.call_tag;
    rfc.nda_args = rcg_func_call.nda_args;
    rtc_call_geom_t dyn_rtc_call_geom = rtc_call_geom;

    for( vect_arg_decl_t::const_iterator i = rtc_func_template->arg_decls.begin(); i != rtc_func_template->arg_decls.end(); ++i ) {
      if( i->io_type == "REF" ) { continue; } // note: could move up scope, but hoping to factor out iter
      rfc.args.push_back( vect_string{} );
      vect_string & args = rfc.args.back();
      uint32_t const multi_sz = i->get_multi_sz( *this );
      for( uint32_t mix = 0; mix != multi_sz; ++mix ) {
        string const vn = i->get_multi_vn(mix);
        dims_t const & func_dims = get_arg_dims_by_name( vn );
        if( func_dims == dims_t() ) { args.push_back("<NULL>"); continue; } // NULL, pass though to rtc
        map_str_str::const_iterator an = rcg_func_call.arg_map.find( vn );
        if( an == rcg_func_call.arg_map.end() ) {
          rt_err( "specified "+i->io_type+" arg '"+vn+"' not found in arg_map at call time." ); 
        }
        dims_t const & call_dims = rtc->get_var_dims( an->second );
        // check that the passed vars are the expected sizes. for non-dyn vars, the sizes must be fully specificed (no
        // wildcards) and be exactly equal. for dyn vars, in particular at least the # of dims per var better match as the
        // cucl_arg_info code assumes this (but here we'll check the dim names too).
        if( !call_dims.matches_template( func_dims ) ) { 
          printf( "error: dims mismatch at call time. call_tag=%s arg=%s: func_dims=%s call_dims=%s call_vn=%s\n", 
                  rfc.call_tag.c_str(), vn.c_str(), str(func_dims).c_str(), str(call_dims).c_str(), an->second.c_str() );
        }	  
        args.push_back( an->second );
      }
    }

    assert_st( rtc_func_template->has_cucl_arg_info.v || dyn_vars.empty() );
    rfc.has_cucl_arg_info = rtc_func_template->has_cucl_arg_info;
    for( vect_dyn_dim_info_t::const_iterator i = dyn_vars.begin(); i != dyn_vars.end(); ++i ) {
      //printf( "i->nda_vn=%s i->src_vn=%s i->use_dims=%s\n", str(i->nda_vn).c_str(), str(i->src_vn).c_str(), str(i->use_dims).c_str() );
      map_str_str::const_iterator an = rcg_func_call.arg_map.find( i->src_vn );
      if( an == rcg_func_call.arg_map.end() ) { rt_err( "<internal error> src arg for dynamic size info '"+i->src_vn+"' not found in arg_map at call time." ); }
      dims_t const & arg_call_dims = rtc->get_var_dims( an->second );
      dims_t dyn_call_dims = apply_use_dims( arg_call_dims, i->use_dims );
      //printf( "dyn_call_dims=%s\n", str(dyn_call_dims).c_str() );
      add_arg_info_for_dims( dyn_call_dims, rfc.cucl_arg_info );
      if( i->nda_vn != i->src_vn ) { // see earlier FIXME, but for now we use this cond to select IX-derived dyn dims
        dyn_rtc_call_geom.maybe_update_for_special_cucl_ixs( i->nda_vn, dyn_call_dims );
      }
    }
    if( show_rtc_calls ) { 
      printf( "%s( args{%s} -- ndas{%s} ) tpb=%s call_blks=%s\n", str(rfc.rtc_func_name).c_str(), 
	      str(rfc.args).c_str(), str(rfc.nda_args).c_str(),
	      str(dyn_rtc_call_geom.tpb).c_str(), str(dyn_rtc_call_geom.blks).c_str() );
      //if( !rfc.cucl_arg_info.empty() ) { printf( "  rfc.cucl_arg_info=%s\n", str(rfc.cucl_arg_info).c_str() ); }
    }
    rfc.tpb.v = dyn_rtc_call_geom.tpb;
    rfc.blks.v = dyn_rtc_call_geom.blks;
    if( has_final_flags_arg ) { rfc.nda_args.push_back( make_scalar_nda<uint32_t>(flags) ); }
    rtc->run( rfc );
    rcg_func_call.call_id = rfc.call_id;
    // note: temporary rfc is gone after this
  }

  p_op_base_t make_p_op_base_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );

  using boost::filesystem::is_regular_file;

  p_rtc_call_gen_t rtc_codegen_t::gen_func( op_base_t const & rfs_full ) {
    // first, get template
    p_rtc_template_t & rtc_template = rtc_templates[rfs_full.type];
    if( !rtc_template ) { rtc_template.reset( new rtc_template_t ); rtc_template->init( rfs_full.type ); }

    p_op_base_t rfs_reduced = rtc_template->check_args( rfs_full );

    p_rtc_call_gen_t & rcg = rtc_func_sigs_map[*rfs_reduced];
    if( !rcg ) { // need to instatiate function and pick unused name
      rcg.reset( new rtc_call_gen_t( *rfs_reduced ) );
      string gen_fn = gen_unused_fn( *rfs_reduced, used_names );
      rcg->init( rtc_template, cc.get(), gen_fn );
      used_names.insert( gen_fn );
      compile_pend.push_back( rcg );
    }    
    return rcg;
  }

  // compile pending (generated but not compiled) functions 
  void rtc_codegen_t::compile( void ) {
    // nothing pending? do nothing. note that running code below is correct, but calls down to rtc->compile() with no
    // functions, which is probably not the best idea.
    if( compile_pend.empty() ) { return; } 
    vect_rtc_func_info_t rtc_prog_infos;
    for( vect_p_rtc_call_gen_t::const_iterator i = compile_pend.begin(); i != compile_pend.end(); ++i ) {
      rtc_prog_infos.push_back( {(*i)->gen_fn,(*i)->rtc_prog_str,static_cast<op_base_t const &>(*(*i))} );
      (*i)->is_compiled.v = 1; // well, or at least we're going to *try* to compile this func anyway ...
    }
    rtc->compile( rtc_prog_infos, rtc_compile_opts );
    compile_pend.clear();
  }

  void rtc_codegen_t::run_func( rcg_func_call_t & call, uint32_t const & flags ) {
    compile(); // compile any pending funcs
    call.func->run_rfc( rtc, rtc_compile_opts.show_rtc_calls, call, flags ); // hmm, a bit bizarre/contorted.
  }

  // clear functions that aren't externally referenced
  void rtc_codegen_t::clear( void ) {
    // note: this process is a bit tricky. need to be careful about order of operations and maintaining state in all cases
    compile_pend.clear(); // will be recreated on the fly, but need to clear refs temp. for unique ref check to be correct
    for( rtc_func_sigs_map_t::const_iterator i = rtc_func_sigs_map.begin(); i != rtc_func_sigs_map.end(); /*no inc*/ ) {
      if( i->second.unique() ) { // not externally referenced? then remove func.
        if( i->second->is_compiled.v ) { rtc->release_func( i->second->gen_fn ); } // only release from rtc if compiled
        must_erase( used_names, i->second->gen_fn ); // name no longer in use
        i = rtc_func_sigs_map.erase(i); // remove from map
      } else { // ... else, keep func.
        if( !i->second->is_compiled.v ) { compile_pend.push_back( i->second ); } // if pending compiling, re-track
        ++i; // keep func in map
      }
    }
  }
  
  void rtc_codegen_t::gc_clear( void ) { if( rtc_func_sigs_map.size() > 1000 ) { clear(); } }

  void rtc_codegen_t::read_rtc_func_sigs( filename_t const & rtc_func_sigs_fn ) {
    p_vect_string in_lines = readlines_fn( rtc_func_sigs_fn );
    for( vect_string::const_iterator i = in_lines->begin(); i != in_lines->end(); ++i ) {
      p_op_base_t v = make_p_op_base_t_init_and_check_unused_from_lexp( parse_lexp( *i ), 0 );
      gen_func( *v );
      uint32_t const ix = i - in_lines->begin();
      if( !(ix % 100000)) { printf( "ix=%s\n", str(ix).c_str() ); }
    }
  }

  void rtc_codegen_t::write_rtc_func_sigs( filename_t const & rtc_func_sigs_fn ) {
    set_op_base_t all_sigs;
    if( is_regular_file( rtc_func_sigs_fn.exp ) ) {  // read in existing contents of file if it exists
      p_vect_string in_lines = readlines_fn( rtc_func_sigs_fn );
      for( vect_string::const_iterator i = in_lines->begin(); i != in_lines->end(); ++i ) {
	p_op_base_t v = make_p_op_base_t_init_and_check_unused_from_lexp( parse_lexp( *i ), 0 );
	all_sigs.insert( *v );
      }
    }
    // insert func sigs from current codegen set
    for( rtc_func_sigs_map_t::const_iterator i = rtc_func_sigs_map.begin(); i != rtc_func_sigs_map.end(); ++i ) { 
      all_sigs.insert( i->first );
    }
    // write set back out
    p_ofstream out = ofs_open( rtc_func_sigs_fn );
    for( set_op_base_t::const_iterator i = all_sigs.begin(); i != all_sigs.end(); ++i ) { (*out) << str( *i ) << "\n"; }
  }

}
