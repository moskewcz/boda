// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"op_base.H"
#include"str_util.H"

namespace boda 
{
  bool op_base_t::operator < ( op_base_t const & o ) const { 
    if( type != o.type ) { return type < o.type; }
    if( dims_vals != o.dims_vals ) { return dims_vals < o.dims_vals; }
    if( str_vals != o.str_vals ) { return str_vals < o.str_vals; }
    return 0;
  }
  
  dims_t const & op_base_t::get_dims( string const & an ) const { return must_find( dims_vals, an ); }
  string const & op_base_t::get_str( string const & an ) const { return must_find( str_vals, an ); }
  uint32_t op_base_t::get_u32( string const & an ) const { return lc_str_u32( must_find( str_vals, an ) ); }
  double op_base_t::get_double( string const & an ) const { return lc_str_d( must_find( str_vals, an ) ); }
  
#include"gen/op_base.H.nesi_gen.cc"
}
