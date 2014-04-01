// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"io_util.H"
#include"nesi.H"

namespace boda 
{
  using namespace boost;

  struct conv_op_t : virtual public nesi // NESI(help="conv_op descriptor") 
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string tag; //NESI(help="tag to refer to conv op by",req=1)
    u32_box_t in_pad; //NESI(default="0 0 0 0",help="input padding")
    u32_pt_t kern_sz; //NESI(default="0 0",help="convolutional kernel size")
    u32_pt_t stride; //NESI(default="1 1",help="step/stride in input")
  };

  typedef vector< conv_op_t > vect_conv_op_t; 
  typedef shared_ptr< conv_op_t > p_conv_op_t; 
  typedef vector< p_conv_op_t > vect_p_conv_op_t;

  struct conv_ana_t : virtual public nesi, public has_main_t // NESI(help="blf rectangle packing",bases=["has_main_t"], type_id="conv_ana")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="text output filename")
    vect_conv_op_t convs; //NESI(help="set of convs")
    // filename_t convs_fn; NESI(help="input: filename for list of convs",req=1)
    // uint32_t in_sz; NESI(help="input size ",req=1)
    virtual void main( nesi_init_arg_t * nia ) { 
      std::ostream * out = &std::cout; // ofs_open( out_fn.exp );
      (*out) << convs << "\n";      
    }
  };

#include"gen/conv_util.cc.nesi_gen.cc"

};
