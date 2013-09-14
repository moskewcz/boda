#ifndef GEOM_PRIM_H
#define GEOM_PRIM_H
#include"boda_base.H"
#include<cstdint>

namespace boda 
{
  struct u32_pt_t
  {
    uint32_t d[2];
    u32_pt_t( void ) : d{0,0} { } // no c++11? --> remove : d{0,0}, use { d[0]=0;d[1]=0; }
  };

  struct u32_box_t
  {
    u32_pt_t p[2];
    void one_to_zero_coord_adj( void );
    bool is_strictly_normalized( void ) const { return (p[0].d[0]<p[1].d[0]) && (p[0].d[1]<p[1].d[1]); }
    uint64_t get_overlap_with( u32_box_t const & o ) const
    {
      uint32_t lb0 = std::max( p[0].d[0], o.p[0].d[0] );
      uint32_t ub0 = std::min( p[1].d[0], o.p[1].d[0] );
      if( lb0 >= ub0 ) { return 0; }
      uint32_t lb1 = std::max( p[0].d[1], o.p[0].d[1] );
      uint32_t ub1 = std::min( p[1].d[1], o.p[1].d[1] );
      if( lb1 >= ub1 ) { return 0; }
      return uint64_t(ub0-lb0)*(ub1-ub0);
    }
    
  };


};
#endif // GEOM_PRIM_H
