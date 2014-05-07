#ifndef _CONV_UTIL_H_
#define _CONV_UTIL_H_

#include"boda_base.H"
#include"geom_prim.H"

namespace boda 
{
  struct conv_op_t : virtual public nesi // NESI(help="conv_op descriptor") 
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string tag; //NESI(help="tag to refer to conv op by",req=1)
    
    u32_box_t in_pad; //NESI(default="0 0 0 0",help="input padding")
    u32_pt_t kern_sz; //NESI(default="0 0",help="convolutional kernel size")
    u32_pt_t stride; //NESI(default="1 1",help="step/stride in input")

    // related to depth (currently unused; might be optional?)
    // uint32_t channels; NESI(default="1",help="number of channels")
    // uint32_t groups; NESI(default="1",help="number of groups (equal partitions of inputs and outputs)")

    u32_pt_t out_sz_to_in_sz( u32_pt_t const & out_sz, bool const ignore_padding ) const;
    u32_pt_t in_sz_to_out_sz( u32_pt_t const & in_sz, bool const ignore_padding ) const;
  };

  typedef vector< conv_op_t > vect_conv_op_t; 
  typedef shared_ptr< conv_op_t > p_conv_op_t; 
  typedef vector< p_conv_op_t > vect_p_conv_op_t;
  typedef shared_ptr< vect_conv_op_t > p_vect_conv_op_t; 

  // struct metadata about inputs/outputs of conv ops
  struct conv_io_t {
    u32_pt_t sz;
    u32_pt_t used_sz;
  };
  
  typedef vector< conv_io_t > vect_conv_io_t; 
  typedef shared_ptr< conv_io_t > p_conv_io_t; 
  typedef vector< p_conv_io_t > vect_p_conv_io_t;
  typedef shared_ptr< vect_conv_io_t > p_vect_conv_io_t; 

  struct conv_support_info_t {  
    u32_pt_t support_sz;
    u32_pt_t support_stride;
    u32_box_t eff_tot_pad;
  };

  typedef vector< conv_support_info_t > vect_conv_support_info_t; 
  typedef shared_ptr< conv_support_info_t > p_conv_support_info_t; 
  typedef vector< p_conv_support_info_t > vect_p_conv_support_info_t;

  struct conv_pipe_t : virtual public nesi // NESI(help="pipeline of convolution-ish opa with strides, padding, and kernel sizes. ")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    p_vect_conv_op_t convs; //NESI(help="set of conv-ish ops",req=1)
    uint32_t ignore_padding_for_support; //NESI(default=1,help="if 1, ignore any padding specified when calculating the support_size for a single pel for each layer")
    vect_conv_support_info_t conv_sis; // note: conv_sis.size() == ( convs.size() + 1 )
    void zero_conv_ios( vect_conv_io_t & conv_ios );
    void calc_support_info( void );
    p_vect_conv_io_t calc_sizes_back( u32_pt_t const & out_sz, bool const ignore_padding );
    p_vect_conv_io_t calc_sizes_forward( u32_pt_t const & in_sz, bool const ignore_padding );
    void dump_pipe( std::ostream & out );
    void dump_ios( std::ostream & out, p_vect_conv_io_t const & conv_ios );
  };

  typedef vector< conv_pipe_t > vect_conv_pipe_t; 
  typedef shared_ptr< conv_pipe_t > p_conv_pipe_t; 
  typedef vector< p_conv_pipe_t > vect_p_conv_pipe_t;

}

#endif /* _CONV_UTIL_H_ */
