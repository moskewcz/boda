// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _NVRTC_UTIL_H_
#define _NVRTC_UTIL_H_

#include"boda_base.H"

namespace boda 
{
  struct conv_pipe_t; typedef shared_ptr< conv_pipe_t > p_conv_pipe_t; 
  struct conv_pipe_fwd_t; typedef shared_ptr< conv_pipe_fwd_t > p_conv_pipe_fwd_t; 

  p_conv_pipe_fwd_t make_conv_pipe_fwd_t( p_conv_pipe_t const & cp );
  void conv_pipe_fwd_t_run( p_conv_pipe_fwd_t const & cpf, p_map_str_p_nda_float_t const & fwd );
}

#endif /* _NVRTC_UTIL_H_ */
