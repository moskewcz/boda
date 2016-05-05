// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"culibs-wrap.H"
#include"str_util.H"
namespace boda 
{
  struct culibs_wrap_t { };
  p_culibs_wrap_t culibs_wrap_init( void ) { return make_shared< culibs_wrap_t >(); }
  void cublas_sgemm_wrap( p_culibs_wrap_t const & cw, uint64_t const & M, uint64_t const & N, uint64_t const & K, 
			  vect_rp_void const & args ) { 
    rt_err( "culibs support not enabled, can't execute cublas_sgemm" );
  }
}
