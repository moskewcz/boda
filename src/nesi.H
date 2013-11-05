#ifndef _NESI_H_
#define _NESI_H_
#include<cstdint>
#include<ostream>

namespace boda {

#include"nesi_decls.H"

  std::ostream & operator<<(std::ostream & os, nesi const & v);
  
}

#endif /* _NESI_H_ */
