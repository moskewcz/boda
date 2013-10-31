#ifndef _HAS_MAIN_H_
#define _HAS_MAIN_H_

#include"boda_base.H"

namespace boda
{
  using std::string;
  struct has_main_t : virtual public nesi // NESI(help="base type of modules that can be run",is_abstract=1,tid_vn="mode")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string mode; //NESI(help="name of mode to run",req=1)
    virtual void main( void ) = 0;
  };
  typedef shared_ptr< has_main_t > p_has_main_t;

}

#endif /* _HAS_MAIN_H_ */
