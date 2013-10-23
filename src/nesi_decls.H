#ifndef _NESI_DECLS_H_
#define _NESI_DECLS_H_

//typedef void make_p_t( void * po ); // po is (p_t *), calls po->reset( new t )
typedef void init_t( void * init_arg, void * o, void * d ); // o is (t *), calls o->init( init_arg, ... )
typedef void * newt_t( void ); // returns new t
typedef void * vect_push_back_t( void * v ); // v is (vect_t *), calls v->push_back( t() ), return & v->back()

init_t nesi_struct_init;
init_t p_init;
init_t vect_init;

struct tinfo_t
{
  char const * tname;
  //make_p_t make_p;
  void * init_arg; // passed to init. for pointers/vects, the tinfo_t of the pointed-to type. for structs, the cinfo_t
  init_t * init;
  newt_t * newt;
  vect_push_back_t * vect_push_back; 
};
struct vinfo_t
{
  char const * help;
  char const * vname;
  tinfo_t * tinfo;
};
typedef void * get_field_t( void * const o, uint32_t const ix );
struct cinfo_t
{
  char const * help;
  get_field_t * get_field;  
  uint32_t num_vars;
  vinfo_t const * vars;
};

#endif /* _NESI_DECLS_H_ */
