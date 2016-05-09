// Copyright (c) 2016, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _CULIBS_WRAP_H_
#define _CULIBS_WRAP_H_

#include"boda_base.H"

namespace boda 
{
  struct culibs_wrap_t; typedef shared_ptr< culibs_wrap_t > p_culibs_wrap_t; 
  struct rtc_func_info_t;
  p_culibs_wrap_t culibs_wrap_init( void );
  void culibs_wrap_call( p_culibs_wrap_t const & cw, rtc_func_info_t const & fi, p_map_str_p_nda_raw_t const & args );

}
#endif /* _CULIBS_WRAP_H_ */
