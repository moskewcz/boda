// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"op_base.H"
#include"str_util.H"
#include<algorithm>

namespace boda 
{
  template< typename T > struct lt_pair_key_p_value { 
    bool operator()( T const & a, T const & b ) const { 
      if( a.first != b.first ) { return a.first < b.first; }
      return (*a.second) < (*b.second); 
    } 
  };

  bool op_base_t::operator < ( op_base_t const & o ) const { 
    if( type != o.type ) { return type < o.type; }
    if( dims_vals != o.dims_vals ) { return dims_vals < o.dims_vals; }
    if( str_vals != o.str_vals ) { return str_vals < o.str_vals; }
    // note: we don't need equals on nda_vals here (or for op_base_t), but if we did, we'd need to use custom comparison
    // there too (or we'd be comparing p_nda_t's, not the underlying nda's. maybe we should wrap the map type and
    // overload op < and ==? luckily we're not often putting nda_t's into maps.
    return std::lexicographical_compare( nda_vals.begin(), nda_vals.end(), o.nda_vals.begin(), o.nda_vals.end(),
                                         lt_pair_key_p_value<map_str_p_nda_t::value_type>() );
  }
  
  dims_t const & op_base_t::get_dims( string const & an ) const { return must_find( dims_vals, an ); }
  string const & op_base_t::get_str( string const & an ) const { return must_find( str_vals, an ); }
  uint32_t op_base_t::get_u32( string const & an ) const { return lc_str_u32( must_find( str_vals, an ) ); }
  double op_base_t::get_double( string const & an ) const { return lc_str_d( must_find( str_vals, an ) ); }
  
#include"gen/op_base.H.nesi_gen.cc"
}
