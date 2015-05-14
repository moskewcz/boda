// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _RAND_UTIL_H_
#define _RAND_UTIL_H_

#include "boda_base.H"
#include "geom_prim.H"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

namespace boda 
{
  template< typename GEN, typename DIST = boost::random::uniform_int_distribution<> >
  u32_pt_t random_pt( u32_pt_t const & max_val, GEN & gen ) {
    return u32_pt_t{ DIST( 0, max_val.d[0] )(gen), DIST( 0, max_val.d[1] )(gen) }; }

  template< typename T, typename GEN, typename DIST = boost::random::uniform_real_distribution<T> >
  void rand_fill_vect( vector< T > & v, T const & min_val, T const & max_val, GEN & gen ) {
    DIST dist( min_val, max_val );
    for( uint32_t i = 0; i != v.size(); ++i ) { v[i] = dist(gen); }
  }
}

#endif /* _RAND_UTIL_H_ */

