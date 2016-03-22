// Copyright (c) 2013-2016, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"stacktrace_util.H"

namespace boda 
{
  p_vect_rp_void get_backtrace( void )  { return p_vect_rp_void(); }
  string stacktrace_str( p_vect_rp_void bt, uint32_t strip_frames ) {
    // note: bt is null
    return string( "----STACK TRACE <not supported in this build of boda>----\n" );
  }
}
