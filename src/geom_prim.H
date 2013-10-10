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
    u32_pt_t( uint32_t const d0, uint32_t const d1 ) : d{d0,d1} { } 
    bool is_zeros( void ) const { return (!d[0])&&(!d[1]); }
  };
  std::ostream & operator<<(std::ostream & os, const u32_pt_t & v);

  struct u32_box_t
  {
    u32_pt_t p[2];
    bool is_strictly_normalized( void ) const { return (p[0].d[0]<p[1].d[0]) && (p[0].d[1]<p[1].d[1]); }
    uint64_t get_overlap_with( u32_box_t const & o ) const
    {
      uint32_t lb0 = std::max( p[0].d[0], o.p[0].d[0] );
      uint32_t ub0 = std::min( p[1].d[0], o.p[1].d[0] );
      if( lb0 >= ub0 ) { return 0; }
      uint32_t lb1 = std::max( p[0].d[1], o.p[0].d[1] );
      uint32_t ub1 = std::min( p[1].d[1], o.p[1].d[1] );
      if( lb1 >= ub1 ) { return 0; }
      return uint64_t(ub0-lb0)*(ub1-lb1);
    }
    uint64_t get_area( void ) const { return uint64_t(p[1].d[0]-p[0].d[0])*(p[1].d[1]-p[0].d[1]); } 

    void from_pascal_coord_adjust( void );
    void to_pascal_coord_adjust( void );
    std::string pascal_str( void ) const;
  };

  std::ostream & operator<<(std::ostream & os, const u32_box_t & v);


};
#endif // GEOM_PRIM_H
