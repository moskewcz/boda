// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _NESI_H_
#define _NESI_H_
#include"boda_base.H"
#include<cstdint>
#include<ostream>
namespace boda {

#include"nesi_decls.H"

  struct lexp_t; typedef shared_ptr< lexp_t > p_lexp_t; 
  void nesi_init_and_check_unused_from_nia( nesi_init_arg_t * nia, tinfo_t const * ti, void * o );
  std::ostream & operator <<(std::ostream & os, nesi const & v);
  void nesi_dump_xml(std::ostream & top_ostream, nesi const & v, char const * const root_elem_name );
  std::string nesi_filename_t_expand( nesi_init_arg_t * nia, std::string const & s );


}

#endif /* _NESI_H_ */
