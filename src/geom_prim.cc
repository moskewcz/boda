#include"boda_tu_base.H"
#include"geom_prim.H"
#include"str_util.H"

namespace boda 
{
  void u32_box_t::from_pascal_coord_adjust( void )
  {
    for( uint32_t d = 0; d < 2; ++d ) {
      if( !p[0].d[d] ) { rt_err( "during from_pascal_coord_adjust(), box had 0 coord, expected >= 1" ); }
      --p[0].d[d]; // adjust 1 based [] coords into 0 based [) ones.
      assert( p[1].d[d] > p[0].d[d] ); // check for strict normalization
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

  std::ostream & operator<<(std::ostream & os, const u32_pt_t & v)
  {
    return os << v.d[0] << " " << v.d[1];
  }

  std::ostream & operator<<(std::ostream & os, const u32_box_t & v)
  {
    return os << "(" << v.p[0] << ")(" << v.p[1] << ")";
  }

}
