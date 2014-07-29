// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"str_util.H"

namespace boda 
{

  template<> void u32_pt_t::read_from_line_parts( vect_string const & parts, uint32_t const init_ix ) {
    if( parts.size() < (init_ix+2) ) { lc_str_u32("not enough parts"); } // will throw bad_lexical_cast
    d[0] = lc_str_u32( parts[init_ix+0] );
    d[1] = lc_str_u32( parts[init_ix+1] );
  }

  void u32_box_t::read_from_line_parts( vect_string const & parts, uint32_t const init_ix ) {
    p[0].read_from_line_parts( parts, init_ix   );
    p[1].read_from_line_parts( parts, init_ix+2 );
  }

  void u32_box_t::from_pascal_coord_adjust( void )
  {
    for( uint32_t d = 0; d < 2; ++d ) {
      if( !p[0].d[d] ) { rt_err( "during from_pascal_coord_adjust(), box had 0 coord, expected >= 1" ); }
      --p[0].d[d]; // adjust 1 based [] coords into 0 based [) ones.
      // check for strict normalization
      if( p[1].d[d] <= p[0].d[d] ) { rt_err( "denormalized box after from_pascal_coord_adjust()" ); } 
    }
  }

  void u32_box_t::to_pascal_coord_adjust( void )
  {
    for( uint32_t d = 0; d < 2; ++d ) {
      ++p[0].d[d]; // adjust 0 based [) coords into 1 based [] ones.
      assert( p[1].d[d] >= p[0].d[d] ); // check for non-strict normalization
    }
  }

  std::string u32_box_t::pascal_str( void ) const
  {
    u32_box_t pascal_box( *this );
    pascal_box.to_pascal_coord_adjust();
    return str(pascal_box.p[0]) + " " + str(pascal_box.p[1]);
  }

  std::ostream & operator <<(std::ostream & os, u32_pt_t const & v)
  {
    return os << v.d[0] << " " << v.d[1];
  }

  std::ostream & operator <<(std::ostream & os, u32_pt_w_t const & v)
  {
    return os << v.d[0] << " " << v.d[1] << " w=" << v.w;
  }

  std::ostream & operator <<(std::ostream & os, u32_box_t const & v)
  {
    return os << "(" << v.p[0] << ")(" << v.p[1] << ")";
  }

  void u32_box_t1( void ) {
    u32_box_t b;
    b.p[1] = u32_pt_t( 3, 4 );
    b.from_pascal_coord_adjust();
  }
  void u32_box_t2( void ) {
    // this is low-level enough that we're not worried about failure reporting. we just run and expect no errors/asserts.
    u32_box_t b;
    assert_st( b.p[0].is_zeros() );
    b.p[1] = u32_pt_t( 0, 4 );
    assert_st( !b.p[1].is_zeros() && !b.is_strictly_normalized() );
    b.p[1] = u32_pt_t( 3, 0 );
    assert_st( !b.p[1].is_zeros() && !b.is_strictly_normalized() );
    b.p[1] = u32_pt_t( 3, 4 );
    assert_st( !b.p[1].is_zeros() && b.is_strictly_normalized() );
    assert_st( b.get_area() == 12 );
    assert_st( b.get_overlap_with(b) == 12 );
    u32_box_t b2 = b;
    b2.p[0] = u32_pt_t( 1, 2 );
    assert_st( b2.get_overlap_with(b) == 4 );
    b2.p[1] = u32_pt_t( 4, 5 );
    assert_st( b2.get_overlap_with(b) == 4 );
    assert_st( !( b2 == b ) );
    u32_box_t b3 = b2;
    b3.from_pascal_coord_adjust();
    b3.to_pascal_coord_adjust();
    assert_st( b3 == b2 );

  }


}
