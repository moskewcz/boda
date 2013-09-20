#include"boda_tu_base.H"
#include"geom_prim.H"

namespace boda 
{
  void u32_box_t::from_pascal_coord_adjust( void )
  {
    for( uint32_t d = 0; d < 2; ++d ) {
      if( !p[0].d[d] ) { rt_err( "during one_to_zero_coord_adj(), box had 0 coord, expected >= 1" ); }
      --p[0].d[d]; // adjust 1 based coords into 0 based ones.
      assert( p[1].d[d] > p[0].d[d] ); // check for strict normalization
    }
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
