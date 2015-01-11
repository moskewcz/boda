#ifndef _ANNO_UTIL_H_
#define _ANNO_UTIL_H_

#include"boda_base.H"
#include"geom_prim.H"

namespace boda {
  struct anno_t {
    i32_box_t box;
    uint32_t box_color;
    bool fill;
    string str;
    uint32_t str_color;
    uint32_t item_cnt;
    set_uint32_t seen_label_ixs;
    //anno_t( void ) : box_color( uint32_t_const_max ), fill(0), str_color( uint32_t_const_max ) { }
  };

  typedef vector< anno_t > vect_anno_t; 
  typedef shared_ptr< vect_anno_t > p_vect_anno_t; 

}

#endif /* _ANNO_UTIL_H_ */
