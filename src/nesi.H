#ifndef _NESI_H_
#define _NESI_H_
#include<cstdint>
#include<boost/shared_ptr.hpp>
using boost::shared_ptr;

namespace boda {
  
#include"nesi_decls.H"

// generic init for shared_ptr types
  typedef shared_ptr< void > p_void;

  void p_init( void * init_arg, void * o, void * d )
  {
    tinfo_t * const pt = (tinfo_t *)( init_arg );
    p_void * const pv = (p_void *)( o );
    pv->reset( pt->newt() ); 
    pt->init( pt->init_arg, pv->get(), d );
  }

  void vect_init( void * init_arg, void * o, void * d )
  {
    tinfo_t * const pt = (tinfo_t *)( init_arg );
    // o is pointer to vect_t, suitable for passing to vect_push_back_t
    // assume d is iterable, yields dis
    for(;;)
    {
      void * di = 0; // derived from iteration over d
      void * rpv = pt->vect_push_back( o );
      pt->init( pt->init_arg, rpv, di );
    }
  }


  void nesi_struct_init( void * init_arg, void * o, void * d )
  {
    cinfo_t * const pc = (cinfo_t *)( init_arg );
    for( uint32_t i = 0; i < pc->num_vars; ++i )
    {
      void * di = 0; // derived from lookup into d using pc->vars[i].vname
      tinfo_t * const pt = pc->vars[i].tinfo;
      void * rpv = pc->get_field( o, i );
      pt->init( pt->init_arg, rpv, di );
    }
  }

  void nesi_init( void );
}

#endif /* _NESI_H_ */
