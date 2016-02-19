// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"rtc_func_gen.H"

namespace boda 
{

  void insert_nda_dims_sz( map_str_str & mss, string const & nda_vn, dims_t const & dims, bool const & dims_only ) {
    for( uint32_t i = 0; i != dims.sz(); ++i ) {
      must_insert( mss, nda_vn+"_"+dims[i].name+"_dim", str(dims[i].sz) );
      if( !dims_only ) { 
	assert_st( dims[i].has_sz_and_stride_and_name() );
	must_insert( mss, nda_vn+"_"+dims[i].name+"_sz", str(dims[i].stride) );
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

  dims_t dims_from_nda_spec( string const & nda_spec ) {
    vect_string const dim_names = split( nda_spec, ':' );
    dims_t arg_dims;
    // for now, assume template: (1) can handle any size for all nda dims (to relax, could allow spec of size,
    // then use that instead of 0/wild for sz below) (2) all dims are static (to relax: unclear what
    // syntax/encoding would be)
    for( vect_string::const_iterator i = dim_names.begin(); i != dim_names.end(); ++i ) { arg_dims.add_dims( *i, 0 ); }
    return arg_dims;
  }


  string rtc_func_sig_t::gen_unused_fn( rtc_func_names_map_t & fns ) const {
    string maybe_fn_base = fn;
    set_string unique_dims;
    for( map_str_dims_t::const_iterator ra = ref_dims.begin(); ra != ref_dims.end(); ++ra ) {
      dims_t const & dims = ra->second;
      for( uint32_t i = 0; i != dims.sz(); ++i ) {
	string const & dn = dims.names(i);
	bool const did_ins = unique_dims.insert( dn ).second;
	if( did_ins ) { maybe_fn_base += "__"+dn+"_"+str(dims.dims(i)); }
      }
    }
    string maybe_fn = maybe_fn_base;
    uint32_t uix = 0;
    while( has( fns, maybe_fn ) ) { ++uix; maybe_fn = maybe_fn_base + "__namegenconflict_" + str(uix); }
    return maybe_fn;
  }

  void rtc_call_gen_t::run_rfc( p_rtc_compute_t const & rtc, bool const & show_rtc_calls, 
				rcg_func_call_t & rcg_func_call, uint32_t const & flags ) {
    rtc_func_call_t rfc;
    rfc.rtc_func_name = rcg_func_call.rtc_func_name;
    rfc.call_tag = rcg_func_call.call_tag;
    rfc.u32_args = rcg_func_call.u32_args;

    for( vect_arg_decl_t::const_iterator i = flat_arg_decls.begin(); i != flat_arg_decls.end(); ++i ) {
      if( i->io_type == "REF" ) { continue; }
      dims_t const & func_dims = get_arg_dims_by_name( i->vn );
      if( func_dims == dims_t() ) { rfc.in_args.push_back("<NULL>"); continue; } // NULL case -- pass though to rtc
      map_str_str::const_iterator an = rcg_func_call.arg_map.find( i->vn );
      if( an == rcg_func_call.arg_map.end() ) {
	rt_err( "specified "+i->io_type+" arg '"+i->vn+"' not found in arg_map at call time." ); 
      }
      // FIXME/note: we can't/don't check call dims for variable-sized ref_dims. this seems less than ideal.
      // one exampl of such usage is in var_stats.
      if( func_dims.has_sz_and_stride_and_name() ) { 
	dims_t const & call_dims = rtc->get_var_dims_floats( an->second );
	if( call_dims != func_dims ) {
	  rt_err( strprintf( "error: dims mismatch at call time for arg %s: call_dims=%s func_dims=%s\n", 
			     i->vn.c_str(), str(call_dims).c_str(), str(func_dims).c_str() ) );
	}	  
      }
      rfc.in_args.push_back( an->second );
    }
    uint32_t call_blks = blks; // if non-zero, blks is static, and we can check arg sizes
    if( !call_blks ) { // handle dynamic # of blks case
      // FIXME: pretty limited / special cased here
      assert_st( rfc.u32_args.size() > 0 );
      call_blks = u32_ceil_div( rfc.u32_args[0], tpb );
    }
    if( show_rtc_calls ) { 
      printf( "%s( in{%s} inout{%s} out{%s} -- u32{%s} ) tpb=%s call_blks=%s\n", str(rfc.rtc_func_name).c_str(), 
	      str(rfc.in_args).c_str(), str(rfc.inout_args).c_str(), str(rfc.out_args).c_str(), str(rfc.u32_args).c_str(),
	      str(tpb).c_str(), str(call_blks).c_str() );
    }
    rfc.tpb.v = tpb;
    rfc.blks.v = call_blks;
    if( has_final_flags_arg ) { rfc.u32_args.push_back( flags ); }
    rtc->run( rfc );
    rcg_func_call.call_id = rfc.call_id;
    // note: temporary rfc is gone after this
  }

#include"gen/rtc_func_gen.H.nesi_gen.cc"
}
