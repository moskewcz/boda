#ifndef _COMP_UTIL_H_
#define _COMP_UTIL_H_

#include"boda_base.H"

namespace boda {
  void read_and_check_boda_magic( std::istream & in );
  void must_bread_string( std::istream & in, string const & s );
  void must_bread_id( std::istream & in, string const & s );

  // two versions, since we have the usual c++ containers+inheritance problem that we can't cast from
  // p_map_str_p_nda_float_t to p_map_str_p_nda_t (legally, anyway)
  void comp_vars( std::ostream * const out, uint32_t & num_mad_fail,
		  double const mrd_toler, map_str_double const * const var_mrd_toler,
		  bool const diff_show_mrd_only, uint32_t const & max_err,
		  vect_string const & vns, p_map_str_p_nda_float_t const & vs1, p_map_str_p_nda_float_t const & vs2 );

  void comp_vars( std::ostream * const out, uint32_t & num_mad_fail,
		  double const mrd_toler, map_str_double const * const var_mrd_toler,
		  bool const diff_show_mrd_only, uint32_t const & max_err,
		  vect_string const & vns, p_map_str_p_nda_t const & vs1, p_map_str_p_nda_t const & vs2 );
  
}

#endif /* _COMP_UTIL_H_ */
