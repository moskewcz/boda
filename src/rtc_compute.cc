// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"rtc_compute.H"
#include"str_util.H"

namespace boda 
{
  void rtc_compute_t::create_var_with_sz_floats( string const & vn, uint32_t const & sz ) { 
    create_var_with_dims_floats( vn, dims_t( vect_uint32_t{sz}, 1) ); 
  }
  void rtc_compute_t::init_var_from_vect_float( string const & vn, vect_float const & v ) { 
    create_var_with_dims_floats( vn, dims_t( vect_uint32_t{uint32_t(v.size())}, 1) ); 
    copy_to_var( vn, &v[0] );
  }
  void rtc_compute_t::set_vect_float_from_var( vect_float & v, string const & vn) {
    assert_st( v.size() == get_var_sz_floats( vn ) );
    copy_from_var( &v[0], vn );
  }
  // nda_float <-> var copies
  void rtc_compute_t::copy_nda_to_var( string const & vn, p_nda_float_t const & nda ) {
    assert_st( nda->elems.sz == get_var_sz_floats( vn ) );
    copy_to_var( vn, &nda->elems[0] );
  }
  void rtc_compute_t::copy_var_to_nda( p_nda_float_t const & nda, string const & vn ) {
    assert_st( nda->elems.sz == get_var_sz_floats( vn ) );
    copy_from_var( &nda->elems[0], vn );
  }
  // create new flat nda from var
  p_nda_float_t rtc_compute_t::copy_var_as_flat_nda( string const & vn ) {
    dims_t cup_dims( vect_uint32_t{get_var_sz_floats( vn )} ); 
    cup_dims.calc_strides();
    p_nda_float_t nda = make_shared<nda_float_t>( cup_dims );
    copy_var_to_nda( nda, vn );
    return nda;
  }
  // batch nda_float<->var copies
  void rtc_compute_t::copy_ndas_to_vars( vect_string const & names, map_str_p_nda_float_t const & ndas ) {
    for( vect_string::const_iterator i = names.begin(); i != names.end(); ++i ) {
      copy_nda_to_var( *i, must_find( ndas, *i ) );
    }
  }
  void rtc_compute_t::copy_vars_to_ndas( vect_string const & names, map_str_p_nda_float_t & ndas ) {
    for( vect_string::const_iterator i = names.begin(); i != names.end(); ++i ) {
      copy_var_to_nda( must_find( ndas, *i ), *i );
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
    filename_t prog_fn; //NESI(default="%(boda_test_dir)/nvrtc_test_dot.cu",help="rtc program source filename")
    uint32_t data_sz; //NESI(default=10000,help="size in floats of test data")
    p_rtc_compute_t rtc; //NESI(default="(be=nvrtc)",help="rtc back-end to use")

    boost::random::mt19937 gen;

    virtual void main( nesi_init_arg_t * nia ) { 
      p_ofstream out = ofs_open( out_fn.exp );
      rtc->init();
      p_string prog_str = read_whole_fn( prog_fn );
      rtc->compile( *prog_str, 0, 0 );

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

      rtc->check_runnable( rfc.rtc_func_name, 0 );

      rtc->run( rfc );
      rtc->finish_and_sync();
      rtc->set_vect_float_from_var( c, "c" );
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
