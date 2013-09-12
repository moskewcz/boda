#include"boda_tu_base.H"
#include"geom_prim.H"

namespace boda 
{
  void u32_box_t::one_to_zero_coord_adj( void )
  {
    for( uint32_t e = 0; e < 2; ++e ) { 
      for( uint32_t d = 0; d < 2; ++d ) {
	if( !p[e].d[d] ) { rt_err( "during one_to_zero_coord_adj(), box had 0 coord, expected >= 1" ); }
	--p[e].d[d]; // adjust 1 based coords into 0 based ones.
      } 
    }
  }
}
