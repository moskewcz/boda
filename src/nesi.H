#ifndef _NESI_H_
#define _NESI_H_
#include<cstdint>
#include<ostream>

namespace boda {

#include"nesi_decls.H"

  std::ostream & operator<<(std::ostream & os, nesi const & v);
  void nesi_dump_xml(std::ostream & top_ostream, nesi const & v, char const * const root_elem_name );

}

#endif /* _NESI_H_ */
