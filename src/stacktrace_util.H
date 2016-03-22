#ifndef _STACKTRACE_UTIL_H_
#define _STACKTRACE_UTIL_H_

#include"boda_base.H"

namespace boda
{
  string stacktrace_str( p_vect_rp_void bt, uint32_t strip_frames );
  p_vect_rp_void get_backtrace( void );
}

#endif /* _STACKTRACE_UTIL_H_ */
