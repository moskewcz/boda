// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"op_base.H"
#include"str_util.H"
#include<algorithm>

namespace boda 
{
  op_base_t::op_base_t( string const & type_, map_str_dims_t const & dims_vals_, map_str_str const & str_vals_ ) :
    type(type_), str_vals( str_vals_ ) 
  {
    for( map_str_dims_t::const_iterator i = dims_vals_.begin(); i != dims_vals_.end(); ++i ) {
      set_dims( i->first, i->second );
    }
  }

  template< typename T > struct lt_pair_key_p_value { 
    bool operator()( T const & a, T const & b ) const { 
      if( a.first != b.first ) { return a.first < b.first; }
      return (*a.second) < (*b.second); 
    } 
  };

  bool op_base_t::operator < ( op_base_t const & o ) const { 
    if( type != o.type ) { return type < o.type; }
    if( str_vals != o.str_vals ) { return str_vals < o.str_vals; }
    // note: we don't need equals on nda_vals here (or for op_base_t), but if we did, we'd need to use custom comparison
    // there too (or we'd be comparing p_nda_t's, not the underlying nda's. maybe we should wrap the map type and
    // overload op < and ==? luckily we're not often putting nda_t's into maps.
    return std::lexicographical_compare( nda_vals.begin(), nda_vals.end(), o.nda_vals.begin(), o.nda_vals.end(),
                                         lt_pair_key_p_value<map_str_p_nda_t::value_type>() );
  }
  
  bool op_base_t::has_dims( string const & an ) const { return has( nda_vals, an ); }
  void op_base_t::set_dims( string const & an, dims_t const & dims ) { 
    must_insert( nda_vals, an, make_dims_nda(dims) ); }
  void op_base_t::erase_dims( string const & an ) { must_erase( nda_vals, an ); }
  void op_base_t::reset_dims( string const & an, dims_t const & dims ) {
    erase_dims( an ); must_insert( nda_vals, an, make_dims_nda( dims ) ); }
  dims_t const & op_base_t::get_dims( string const & an ) const { return must_find( nda_vals, an )->dims; }
  string const & op_base_t::get_str( string const & an ) const { return must_find( str_vals, an ); }
  uint32_t op_base_t::get_u32( string const & an ) const { return lc_str_u32( must_find( str_vals, an ) ); }
  
#include"gen/op_base.H.nesi_gen.cc"
}
