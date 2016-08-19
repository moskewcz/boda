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
    if( str_vals != o.str_vals ) { return str_vals < o.str_vals; }
    // note: we don't need equals on nda_vals here (or for op_base_t), but if we did, we'd need to use custom comparison
    // there too (or we'd be comparing p_nda_t's, not the underlying nda's. maybe we should wrap the map type and
    // overload op < and ==? luckily we're not often putting nda_t's into maps.
    return std::lexicographical_compare( nda_vals.begin(), nda_vals.end(), o.nda_vals.begin(), o.nda_vals.end(),
                                         lt_pair_key_p_value<map_str_p_nda_t::value_type>() );
  }
  
  bool op_base_t::has( string const & an ) const { return boda::has( nda_vals, an ); }
  void op_base_t::set( string const & an, p_nda_t const & nda ) { must_insert( nda_vals, an, nda); }
  void op_base_t::set_dims( string const & an, dims_t const & dims ) { set( an, make_dims_nda(dims) ); }
  void op_base_t::erase( string const & an ) { must_erase( nda_vals, an ); }
  void op_base_t::reset_dims( string const & an, dims_t const & dims ) { erase( an ); set_dims( an, dims ); }

  bool op_base_t::has_func_name( void ) const { return boda::has( str_vals, "func_name" ); }
  string const & op_base_t::get_func_name( void ) const { return must_find( str_vals, "func_name" ); }
  void op_base_t::set_func_name( string const & func_name_ ) { must_insert( str_vals, "func_name", func_name_ ); }
  void op_base_t::erase_func_name( void ) { must_erase( str_vals, "func_name" ); }

  bool op_base_t::has_type( void ) const { return boda::has( str_vals, "type" ); }
  string const & op_base_t::get_type( void ) const { return must_find( str_vals, "type" ); }
  void op_base_t::set_type( string const & type_ ) { must_insert( str_vals, "type", type_ ); }
  void op_base_t::erase_type( void ) { must_erase( str_vals, "type" ); }

  
  p_nda_t const & op_base_t::get( string const & an ) const { return must_find( nda_vals, an ); }
  dims_t const & op_base_t::get_dims( string const & an ) const { return get(an)->dims; }
  string const & op_base_t::get_str( string const & an ) const { return must_find( str_vals, an ); }
  uint32_t op_base_t::get_u32( string const & an ) const { return SNE<uint32_t>(*get( an )); }
  void op_base_t::set_u32( string const & an, uint32_t const & v ) { set( an, make_scalar_nda( v ) ); }

#include"gen/op_base.H.nesi_gen.cc"
}
