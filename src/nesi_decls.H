#ifndef _NESI_DECLS_H_
#define _NESI_DECLS_H_

// NOTE: this header is designed to be included within namespace boda,
// so it has no namspace boda {} of its own, nor does it (or can it)
// include other files itself. in particular, the files of type
// "gen/X.cc.nesi_gen.cc" which are included inside the boda namespace
// block of the corresponding X.cc file include this file. they also
// lack namespace blocks or other includes. while this method seems
// okay for now it may want or need to be changed at some point.

struct nesi;
struct tinfo_t;
struct nesi_dump_buf_t;

typedef void * cast_nesi_to_cname_t( nesi * p );
typedef nesi * cast_cname_to_nesi_t( void * p );

struct sstr_t;
struct lexp_t; typedef shared_ptr< lexp_t > p_lexp_t; 
struct lexp_name_val_map_t;
typedef lexp_name_val_map_t nesi_init_arg_t;

typedef void * make_p_t( nesi_init_arg_t * nia, tinfo_t const * tinfo, void * o ); // po is (p_t *), calls po->reset( new t )
typedef void make_p_nesi_t( void * p ); // p is (p_nesi *), calls p->reset( new t )
// v is (p_nesi *), dv is (p_cname *), does pd=dyn_sp_cas(p); assert(pd); return pd.get;
typedef void * set_p_cname_from_p_nesi_t( void * v, void * dv ); 
typedef void init_t( nesi_init_arg_t * nia, tinfo_t const * tinfo, void * o ); // o is (t *), calls o->init( init_arg, ... )
typedef void * vect_push_back_t( void * v ); // v is (vect_t *), calls v->push_back( t() ), return & v->back()
typedef bool nesi_dump_t( tinfo_t const * tinfo, void * o, nesi_dump_buf_t * ndb );
typedef void nesi_help_t( tinfo_t const * tinfo, void * o, void * os, 
			  bool const show_all, void * help_args, uint32_t help_ix );

init_t nesi_struct_init;
init_t vect_init;
init_t p_init;

make_p_t nesi_struct_make_p;
make_p_t vect_make_p;
make_p_t p_make_p;

vect_push_back_t vect_vect_push_back;
vect_push_back_t p_vect_push_back;

nesi_dump_t nesi_struct_nesi_dump;
nesi_dump_t vect_nesi_dump;
nesi_dump_t p_nesi_dump;

nesi_help_t nesi_struct_nesi_help;
nesi_help_t p_nesi_help;
nesi_help_t vect_nesi_help;
nesi_help_t leaf_type_nesi_help;

struct tinfo_t
{
  uint32_t sz_bytes;
  char const * tname;
  void * init_arg; // for pointers/vects, the tinfo_t of the pointed-to type. for structs, the cinfo_t
  init_t * init;
  make_p_t * make_p; // for structs, may return pointer-to-derived
  vect_push_back_t * vect_push_back; 
  nesi_dump_t * nesi_dump;
  bool no_init_okay;
  nesi_help_t * nesi_help;
};

struct vinfo_t
{
  char const * help;
  char const * default_val;
  bool req;
  char const * vname;
  tinfo_t * tinfo;
  bool hide;
  bool exists( void ) const { return help; }
};
typedef void * get_field_t( void * const o, uint32_t const ix );
struct cinfo_t
{
  tinfo_t const * tinfo; // back ref to the tinfo for this nesi struct (i.e. the tinfo with tinfo->init_arg == this)
  char const * cname;
  char const * help;
  get_field_t * get_field;  
  vinfo_t const * vars;
  make_p_nesi_t * make_p_nesi; // note: always allocates an actual object of this type (not a derived type)
  set_p_cname_from_p_nesi_t * set_p_cname_from_p_nesi;
  uint32_t tid_vix;
  char const * const tid_str;
  cinfo_t const * const * derived;
  cinfo_t const * const * bases;
  cast_cname_to_nesi_t * cast_cname_to_nesi;
  cast_nesi_to_cname_t * cast_nesi_to_cname;
  bool hide; // if true, hide class in nesi_struct_help
  bool exists( void ) const { return help; }
};

#endif /* _NESI_DECLS_H_ */
