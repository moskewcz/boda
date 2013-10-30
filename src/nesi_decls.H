#ifndef _NESI_DECLS_H_
#define _NESI_DECLS_H_

// NOTE: this header is designed to be included within namespace boda,
// so it has no namspace boda {} of its own, nor does it (or can it)
// include other files itself. in particular, the files of type
// "gen/X.cc.nesi_gen.cc" which are included inside the boda namespace
// block of the corresponding X.cc file include this file. they also
// lack namespace blocks or other includes. while this method seems
// okay for now it may want or need to be changed at some point.

typedef void * make_p_t( void * init_arg, void * o, void * d ); // po is (p_t *), calls po->reset( new t )
typedef void * make_p_base_t( void * o ); // po is (p_t *), calls po->reset( new t )
typedef void init_t( void * init_arg, void * o, void * d ); // o is (t *), calls o->init( init_arg, ... )
typedef void * vect_push_back_t( void * v ); // v is (vect_t *), calls v->push_back( t() ), return & v->back()

init_t nesi_struct_init;
init_t vect_init;
init_t p_init;

make_p_t nesi_struct_make_p;
make_p_t vect_make_p;
make_p_t p_make_p;

vect_push_back_t vect_vect_push_back;
vect_push_back_t p_vect_push_back;

struct tinfo_t
{
  char const * tname;
  //make_p_t make_p;
  void * init_arg; // passed to init. for pointers/vects, the tinfo_t of the pointed-to type. for structs, the cinfo_t
  init_t * init;
  make_p_t * make_p; // for structs, may return pointer-to-derived
  vect_push_back_t * vect_push_back; 
};
struct vinfo_t
{
  char const * help;
  char const * default_val;
  bool req;
  char const * vname;
  tinfo_t * tinfo;
  bool exists( void ) const { return help; }
};
typedef void * get_field_t( void * const o, uint32_t const ix );
struct cinfo_t
{
  char const * cname;
  char const * help;
  get_field_t * get_field;  
  vinfo_t const * vars;
  make_p_base_t * make_p_base; // always allocates an actual object of this type (not a derived type)
  uint32_t tid_vix;
  char const * const tid_str;
  cinfo_t const * const * derived;
  cinfo_t const * const * bases;
  bool exists( void ) const { return help; }
};

#endif /* _NESI_DECLS_H_ */
