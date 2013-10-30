#ifndef _HAS_MAIN_H_
#define _HAS_MAIN_H_

#include<string>

namespace boda
{
  using std::string;
  struct has_main_t // NESI(help="base type of modules that can be run",is_abstract=1,tid_vn="mode")
  {
    string mode; //NESI(help="name of mode to run",req=1)
    virtual void main( void ) = 0;
  };
  typedef shared_ptr< has_main_t > p_has_main_t;

}

#endif /* _HAS_MAIN_H_ */
