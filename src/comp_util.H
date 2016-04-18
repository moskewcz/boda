#ifndef _COMP_UTIL_H_
#define _COMP_UTIL_H_

#include"boda_base.H"

namespace boda {

  void comp_vars( std::ostream * const out, uint32_t & num_mad_fail,
		  double const mad_toler, map_str_double const * const var_mad_toler,
		  bool const diff_show_mad_only, uint32_t const & max_err,
		  vect_string const & vns, p_map_str_p_nda_float_t const & vs1, p_map_str_p_nda_float_t const & vs2 );
  
}

#endif /* _COMP_UTIL_H_ */
