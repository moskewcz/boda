// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"rtc_compute.H"
#include"str_util.H"

namespace boda 
{
  void rtc_launch_check_blks_and_tpb( std::string const & rtc_func_name, uint64_t const blks, uint64_t const tpb ) {
    if( !( (blks > 0) && (tpb > 0) ) ) {
      rt_err( strprintf( "boda/rtc: can't launch kernel; blks or tpb is zero: rtc_func_name=%s blks=%s tpb=%s;"
                         " perhaps is a culibs stub function that should not have been attempted to be run?", 
                         str(rtc_func_name).c_str(), str(blks).c_str(), str(tpb).c_str() ) );
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
      rtc->compile( *prog_str, 0, 0, vect_rtc_func_info_t{rtc_func_info_t{"my_dot",op_base_t()}}, 0 );

      vect_float a( data_sz, 0.0f );
      rand_fill_vect( a, 2.5f, 7.5f, gen );
      vect_float b( data_sz, 0.0f );
      rand_fill_vect( b, 2.5f, 7.5f, gen );
      vect_float c( data_sz, 123.456f );

      rtc->init_var_from_vect_float( "a", a );
      rtc->init_var_from_vect_float( "b", b );
      rtc->init_var_from_vect_float( "c", c );
      
      rtc_func_call_t rfc{ "my_dot", {"a","b"},{},{"c"}, {data_sz} }; 
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
