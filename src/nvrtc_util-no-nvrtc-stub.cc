// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"nvrtc_util.H"

namespace boda 
{
  p_conv_pipe_fwd_t make_conv_pipe_fwd_t( p_conv_pipe_t const & cp ) { rt_err( "nvrtc support disabled" ); }
  void conv_pipe_fwd_t_run( p_conv_pipe_fwd_t const & cpf, p_map_str_p_nda_float_t const & fwd ) { }
}
