// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"rtc_compute.H"
#include"str_util.H"

namespace boda 
{

  std::ostream & operator << ( std::ostream & out, rtc_func_info_t const & o ) {
    out << strprintf( "o.func_name=%s o.func_src.size()=%s\n", str(o.func_name).c_str(), str(o.func_src.size()).c_str() );
    return out;
  }

  std::ostream & operator << ( std::ostream & out, rtc_arg_t const & o ) {
    out << strprintf( "o.n=%s o.v=%s\n", str(o.n).c_str(), str(o.v).c_str() );
    return out;
  }

  void rtc_launch_check_blks_and_tpb( std::string const & rtc_func_name, uint64_t const blks, uint64_t const tpb ) {
    if( !( (blks > 0) && (tpb > 0) ) ) {
      rt_err( strprintf( "boda/rtc: can't launch kernel; blks or tpb is zero: rtc_func_name=%s blks=%s tpb=%s;"
                         " perhaps is a culibs stub function that should not have been attempted to be run?", 
                         str(rtc_func_name).c_str(), str(blks).c_str(), str(tpb).c_str() ) );
    }
  }

  void rtc_reshape_check( dims_t const & dims, dims_t const & src_dims ) {
    // check that reshape is valid; for now, check that types and dims_prod() match. a weaker (but maybe
    // valid/okay/useful) check would be just that bytes_sz() is the same (and in fact this is checked in the
    // var_info_t ctor).
    if( dims.tn != src_dims.tn ) { 
      rt_err( strprintf( "invalid reshape; types don't match: dims.tn=%s src_vi.tn=%s\n", 
                         str(dims.tn).c_str(), str(src_dims.tn).c_str() ) );
    }
    if( dims.dims_prod() != src_dims.dims_prod() ) { 
      rt_err( strprintf( "invalid reshape; types match but sizes don't: dims.dims_prod()=%s src_dims.dims_prod()=%s\n", 
                         str(dims.dims_prod()).c_str(), str(src_dims.dims_prod()).c_str() ) );
    }
  }

  void rtc_compute_t::init_var_from_vect_float( string const & vn, vect_float const & v ) { 
    p_nda_t nda = make_shared<nda_t>( dims_t{ vect_uint32_t{uint32_t(v.size())}, "float" }, (void*)&v[0] );
    create_var_with_dims( vn, nda->dims ); 
    copy_nda_to_var( vn, nda );
  }
  void rtc_compute_t::set_vect_float_from_var( vect_float & v, string const & vn) {
    dims_t vn_dims = get_var_dims( vn );
    assert_st( vn_dims.sz() == 1 );
    assert_st( v.size() == vn_dims.dims(0) );
    p_nda_t nda = make_shared<nda_t>( vn_dims, (void*)&v[0] );
    copy_var_to_nda( nda, vn );
  }
  void rtc_compute_t::create_var_from_nda( p_nda_t const & nda, string const & vn ) {
    create_var_with_dims( vn, nda->dims );
    copy_nda_to_var( vn, nda );
  }
  p_nda_t rtc_compute_t::create_nda_from_var( string const & vn ) {
    p_nda_t ret = make_shared<nda_t>( get_var_dims( vn ) );
    copy_var_to_nda( ret, vn );
    return ret;
  }
  // create new flat nda from var
  p_nda_t rtc_compute_t::copy_var_as_flat_nda( string const & vn ) {
    p_nda_t ret = create_nda_from_var( vn );
    ret->dims.init( vect_uint32_t{uint32_t(ret->dims.strides_sz)}, 0, ret->dims.tn ); // reshape to flat (no names)
    return ret;
  }
  // batch nda<->var copies
  void rtc_compute_t::copy_ndas_to_vars( vect_string const & names, map_str_p_nda_t const & ndas ) {
    for( vect_string::const_iterator i = names.begin(); i != names.end(); ++i ) {
      copy_nda_to_var( *i, must_find( ndas, *i ) );
    }
  }
  // assumes that names do not exist in ndas, or alreay exist with proper dims
  void rtc_compute_t::copy_vars_to_ndas( vect_string const & names, map_str_p_nda_t & ndas ) {
    for( vect_string::const_iterator i = names.begin(); i != names.end(); ++i ) { 
      if( has( ndas, *i ) ) { copy_var_to_nda( ndas[*i], *i ); }
      else { must_insert( ndas, *i, create_nda_from_var( *i ) ); }
    } 
  }

  // FIXME_TNDA: dup'd for now, see FIXME_TNDA in header
  // batch nda<->var copies
  void rtc_compute_t::copy_ndas_to_vars( vect_string const & names, map_str_p_nda_float_t const & ndas ) {
    for( vect_string::const_iterator i = names.begin(); i != names.end(); ++i ) {
      copy_nda_to_var( *i, must_find( ndas, *i ) );
    }
  }
  // assumes that names do not exist in ndas, or alreay exist with proper dims
  void rtc_compute_t::copy_vars_to_ndas( vect_string const & names, map_str_p_nda_float_t & ndas ) {
    for( vect_string::const_iterator i = names.begin(); i != names.end(); ++i ) { 
      if( has( ndas, *i ) ) { copy_var_to_nda( ndas[*i], *i ); }
      else { must_insert( ndas, *i, make_shared< nda_float_t >( create_nda_from_var( *i ) ) ); }
    } 
  }

  string const & rtc_compute_t::get_rtc_base_decls( void ) { // note: caches results in rtc_base_decls
    if( rtc_base_decls.empty() ) { // first call, fill in
      rtc_base_decls += "\n// begin rtc_base_decls\n";
#if 0
      // for now, this is disabled, since it's unused/doesn't-work. in OpenCL, there seems to be no way to pass dynamic
      // numbers of points or put them in structs. so any pointer we want to pass needs to be a kernel arg by
      // itself. double-indirection of args is similarly a no-no. so, if we ever want to pass a variable number of
      // kernel arguments, we'll still need something like this (i.e. to pass arrays of indexes by-value, when the
      // number of values is small by unknown). in particular, if we move to an arena allocator, we'd be able to pass a
      // variable-number of indexes this way, as if they were seperate args, but be able to loop over them without
      // flattening/codegen.
      for( ndat_infos_t::const_iterator i = ndat_infos.begin(); i != ndat_infos.end(); ++i ) {
        string const & tn = i->first;
        // FIXME: for now, just omit double (since we don't know if it's supported here, and we don't use it anywhere
        // yet. could add a virtual to rtc_compute and use it here to determine if double is supported)
        if( i->second == &double_ndat ) { continue; } 
        uint32_t const max_multi_args = 20;
        for( uint32_t i = 0; i != max_multi_args; ++i ) {
          string const sn = strprintf( "%s_multi_%s", tn.c_str(), str(i).c_str() );
          rtc_base_decls += strprintf( "typedef struct %s { GASQ %s * marg[%s]; } %s;\n", 
                                       sn.c_str(), tn.c_str(), str(i).c_str(), sn.c_str() );
        }
      }
#endif
      rtc_base_decls += "\n// end rtc_base_decls\n";
    }
    return rtc_base_decls;
  }
}

// extra includes only for test mode
#include"rand_util.H"
#include"has_main.H"
namespace boda 
{

  struct rtc_test_t : virtual public nesi, public has_main_t // NESI(help="test basic usage of rtc", bases=["has_main_t"], type_id="rtc_test")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t out_fn; //NESI(default="%(boda_output_dir)/rtc_test.txt",help="output: test results as error or 'all is well'.")
    filename_t prog_fn; //NESI(default="%(boda_test_dir)/rtc/dot.cucl",help="rtc program source filename")
    uint32_t data_sz; //NESI(default=10000,help="size in floats of test data")
    p_rtc_compute_t rtc; //NESI(default="(be=nvrtc)",help="rtc back-end to use")

    boost::random::mt19937 gen;

    virtual void main( nesi_init_arg_t * nia ) { 
      p_ofstream out = ofs_open( out_fn.exp );
      rtc->init();
      p_string prog_str = read_whole_fn( prog_fn );
      // normally, one would go though the rtc_codegen_t interface, not use the rtc_compute_t interface directly. but,
      // for this minimal test, we do use the rtc layer directly. however, we're stubbing-out and/or hard-coding quite a
      // few bits here. one interesting thing is that the op_base_t is *almost* unused at the rtc level, but the
      // "func_name" value is tested by the nvrtc backend to determine if it should call out to an external library (and
      // if so, the func_name tells it what library and function to call). so we must set it here, since it's
      // unconditionally looked-up. that's the current state anyway, but it subject to change/improvement ...
      rtc->compile( vect_rtc_func_info_t{rtc_func_info_t{"my_dot",*prog_str,
              op_base_t{"my_dot",{},{{"func_name","my_dot"}}}}}, rtc_compile_opts_t() );

      vect_float a( data_sz, 0.0f );
      rand_fill_vect( a, 2.5f, 7.5f, gen );
      vect_float b( data_sz, 0.0f );
      rand_fill_vect( b, 2.5f, 7.5f, gen );
      vect_float c( data_sz, 123.456f );

      rtc->init_var_from_vect_float( "a", a );
      rtc->init_var_from_vect_float( "b", b );
      rtc->init_var_from_vect_float( "c", c );
      
      rtc_func_call_t rfc{ "my_dot", {{"a"},{"b"},{"c"},{"",make_scalar_nda(data_sz)} } }; 
      rfc.tpb.v = 256;
      rfc.blks.v = u32_ceil_div( data_sz, rfc.tpb.v );

      rtc->run( rfc );
      rtc->finish_and_sync();
      rtc->set_vect_float_from_var( c, "c" );
      rtc->release_all_funcs();
      assert_st( b.size() == a.size() );
      assert_st( c.size() == a.size() );
      for( uint32_t i = 0; i != c.size(); ++i ) {
	if( fabs((a[i]+b[i]) - c[i]) > 1e-6f ) {
	  (*out) << strprintf( "bad res: i=%s a[i]=%s b[i]=%s c[i]=%s\n", str(i).c_str(), str(a[i]).c_str(), str(b[i]).c_str(), str(c[i]).c_str() );
	  return;
	}
      }
      (*out) << "All is Well.\n";
    }
  };

#include"gen/rtc_compute.H.nesi_gen.cc"
#include"gen/rtc_compute.cc.nesi_gen.cc"
}
