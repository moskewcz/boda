// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _HAS_CONV_FWD_H_
#define _HAS_CONV_FWD_H_

#include"boda_base.H"

namespace boda
{
  using std::string;
  typedef shared_ptr< string > p_string;

  struct conv_pipe_t; typedef shared_ptr< conv_pipe_t > p_conv_pipe_t; 

  struct lexp_name_val_map_t;
  typedef lexp_name_val_map_t nesi_init_arg_t;
  struct has_conv_fwd_t : virtual public nesi // NESI(help="modes that can be used for fwd computation of conv pipe"
                                              // ,is_abstract=1,tid_vn="mode")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string mode; //NESI(help="name of mode to use for fwd computation of conv pipe",req=1)
    virtual void init( p_conv_pipe_t const & cp, nesi_init_arg_t * const nia ) = 0;
    virtual void set_det_drop_seed( uint32_t const & det_drop_seed_ ) = 0;
    virtual void run_fwd( vect_string const & to_set_vns, p_map_str_p_nda_float_t const & fwd, vect_string const & to_get_vns ) = 0;
    virtual string get_info_log( void ) = 0;
  };
  typedef shared_ptr< has_conv_fwd_t > p_has_conv_fwd_t; 
}

#endif /* _HAS_CONV_FWD_H_ */
