// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"culibs-wrap.H"
#include"str_util.H"
namespace boda 
{
  struct culibs_wrap_t { };
  p_culibs_wrap_t culibs_wrap_init( void ) { return make_shared< culibs_wrap_t >(); }
  void culibs_wrap_call( p_culibs_wrap_t const & cw, string const & fn, p_map_str_p_nda_t const & args ) {
    rt_err( "culibs support not enabled, can't execute cublas_sgemm" );
  }
}
