#ifndef _CONV_COMMON_H_
#define _CONV_COMMON_H_

#include"geom_prim.H"

namespace boda 
{
  struct conv_support_info_t {  
    u32_pt_t support_sz;
    u32_pt_t support_stride;
    u32_box_t eff_tot_pad;
  };

  typedef vector< conv_support_info_t > vect_conv_support_info_t; 
  typedef shared_ptr< conv_support_info_t > p_conv_support_info_t; 
  typedef vector< p_conv_support_info_t > vect_p_conv_support_info_t;

  // struct metadata about inputs/outputs of conv ops
  struct conv_io_t {
    u32_pt_t sz;
    u32_pt_t used_sz;
  };
  
  typedef vector< conv_io_t > vect_conv_io_t; 
  typedef shared_ptr< conv_io_t > p_conv_io_t; 
  typedef vector< p_conv_io_t > vect_p_conv_io_t;
  typedef shared_ptr< vect_conv_io_t > p_vect_conv_io_t; 

}
#endif /* _CONV_COMMON_H_ */
