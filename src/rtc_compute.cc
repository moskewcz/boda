// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"rtc_compute.H"
#include"str_util.H"

namespace boda 
{
  void rtc_compute_t::init_var_from_vect_float( string const & vn, vect_float const & v ) { 
    create_var_with_sz_floats( vn, v.size() ); 
    copy_to_var( vn, &v[0] );
  }
  void rtc_compute_t::set_vect_float_from_var( vect_float & v, string const & vn) {
    assert_st( v.size() == get_var_sz( vn ) );
    copy_from_var( &v[0], vn );
  }
  // nda_float <-> var copies
  void rtc_compute_t::copy_nda_to_var( string const & vn, p_nda_float_t const & nda ) {
    assert_st( nda->elems.sz == get_var_sz( vn ) );
    copy_to_var( vn, &nda->elems[0] );
  }
  void rtc_compute_t::copy_var_to_nda( p_nda_float_t const & nda, string const & vn ) {
    assert_st( nda->elems.sz == get_var_sz( vn ) );
    copy_from_var( &nda->elems[0], vn );
  }
  // create new flat nda from var
  p_nda_float_t rtc_compute_t::copy_var_as_flat_nda( string const & vn ) {
    dims_t cup_dims( vect_uint32_t{get_var_sz( vn )} ); 
    cup_dims.calc_strides();
    p_nda_float_t nda = make_shared<nda_float_t>( cup_dims );
    copy_var_to_nda( nda, vn );
    return nda;
  }
  // batch nda_float<->var copies
  void rtc_compute_t::copy_ndas_to_vars( vect_string const & names, map_str_p_nda_float_t const & ndas ) {
    for( vect_string::const_iterator i = names.begin(); i != names.end(); ++i ) {
      string const pyid = as_pyid( *i ); // FIXME: move to callers/outside?
      copy_nda_to_var( pyid, must_find( ndas, pyid ) );
    }
  }
  void rtc_compute_t::copy_vars_to_ndas( vect_string const & names, map_str_p_nda_float_t & ndas ) {
    for( vect_string::const_iterator i = names.begin(); i != names.end(); ++i ) {
      string const pyid = as_pyid( *i );
      copy_var_to_nda( must_find( ndas, pyid ), pyid );
    }
  }

//#include"gen/rtc_compute.H.nesi_gen.cc"
}
