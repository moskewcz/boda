#ifndef DATA_STREAM_VELO_H
#define DATA_STREAM_VELO_H

#include"boda_base.H"

namespace boda
{

  // in stream order. note that in XML, typically the order is the same expect the vert/rot entries are swapped ... but
  // in xml the order should not be (treated as) significant.
  struct laser_corr_t {
    float vert_corr;
    float rot_corr;
    float dist_corr;    
    float dist_corr_x;
    float dist_corr_y;
    float off_corr_vert;
    float off_corr_horiz;
    float focal_dist;
    float focal_slope;
  };
  typedef vector< laser_corr_t > vect_laser_corr_t;
  std::ostream & operator <<(std::ostream & os, laser_corr_t const & v);
  void read_velo_config( filename_t const & velo_config_xml_fn, vect_laser_corr_t & laser_corrs );

  struct laser_corr_t_by_vert_corr {
    bool operator()( laser_corr_t const & a, laser_corr_t const & b ) const { return a.vert_corr < b.vert_corr; }
  };
  
}

#endif /* DATA_STREAM_VELO_H */
