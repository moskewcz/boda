#ifndef GEOM_PRIM_H
#define GEOM_PRIM_H
#include"boda_base.H"
#include<cstdint>

namespace boda 
{
  struct u32_pt_t
  {
    uint32_t d[2];
  };

  struct u32_box_t
  {
    u32_pt_t p[2];
    void one_to_zero_coord_adj( void );
    bool is_strictly_normalized( void ) const { return (p[0].d[0]<p[1].d[0]) && (p[0].d[1]<p[1].d[1]); }
  };


};
#endif // GEOM_PRIM_H
