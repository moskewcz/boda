#include"nesi.H"
#include<cassert>
#include<vector>
#include<boost/shared_ptr.hpp>
using boost::shared_ptr;
using std::vector;


#include<string>
using std::string;

#include"lexp.H"
#include"str_util.H"

namespace boda 
{

  typedef shared_ptr< void > p_void;
  typedef vector< char > vect_char;

  typedef shared_ptr< p_void > p_p_void;  
  typedef vector< vector< char > > vect_vect_char;
  typedef shared_ptr< vect_char > p_vect_char;
  typedef vector< p_void > vect_p_void;

  // generic init for shared_ptr types
  void p_init( void * init_arg, void * o, void * d )
  {
    tinfo_t * const pt = (tinfo_t *)( init_arg );
    void * v = pt->make_p( o );
    pt->init( pt->init_arg, v, d );
  }

  void vect_init( void * init_arg, void * o, void * d )
  {
    tinfo_t * const pt = (tinfo_t *)( init_arg );
    lexp_t * l = (lexp_t *)d;
    for( vect_lexp_nv_t::iterator i = l->kids.begin(); i != l->kids.end(); ++i ) {
      void * rpv = pt->vect_push_back( o );
      // note: for vector initialization, i->n (the name of the name/value pair) is ignored.
      pt->init( pt->init_arg, rpv, i->v.get() ); 
    }
  }

  void nesi_struct_init( void * init_arg, void * o, void * d )
  {
    cinfo_t * const pc = (cinfo_t *)( init_arg );
    lexp_t * l = (lexp_t *)d;
    lexp_name_val_map_t nvm;
    if( l->leaf_val.exists() ) {
      rt_err( "invalid attempt to use string as name/value list. string was:" + str(*l) );
    }
    for( vect_lexp_nv_t::iterator i = l->kids.begin(); i != l->kids.end(); ++i ) {
      bool const did_ins = nvm.insert( std::make_pair( i->n, i->v ) ).second;
      if( did_ins ) { rt_err( "invalid duplicate name "+i->n.str()+"in name/value list" ); }
    }
    sstr_t ss_vname;
    for( uint32_t i = 0; i < pc->num_vars; ++i )
    {
      vinfo_t const * const vi = &pc->vars[i];
      tinfo_t * const pt = vi->tinfo;
      void * rpv = pc->get_field( o, i );
      p_lexp_t di;
      ss_vname.borrow_from_string( vi->vname );
      lexp_name_val_map_t::const_iterator nvmi = nvm.find( ss_vname );
      if( nvmi != nvm.end() ) { di = nvmi->second; }
      if( !di && vi->default_val ) { di = parse_lexp( vi->default_val ); }
      // FIXME: distinguish types that are 'okay' without init (nesi struct, pointer, vector), 
      // and those that are not (all others / basic types, maybe with expections (i.e. string) ).
      if( !di && vi->req ) { rt_err( "missing required value" ); } 
      pt->init( pt->init_arg, rpv, di.get() );
    }
  }

  // not legal, but works, since for all vect<T> and shared_ptr<T>
  // these operations are the same bitwise (zeroing, bitwise copies,
  // etc...). note: we could code-generate per-type versions if we
  // needed/wanted to.
  void * vect_vect_push_back( void * v ) {
    vect_vect_char * vv = (vect_vect_char *)v;
    vv->push_back( vect_char() );
    return &vv->back();
  }
  void * vect_make_p( void * v ) { 
    p_vect_char * const pvc = (p_vect_char *)( v );
    pvc->reset( new vect_char );
    return pvc->get();
  }
  void * p_vect_push_back( void * v )
  {
    vect_p_void * vpv = (vect_p_void *)v;
    vpv->push_back( p_void() );
    return &vpv->back();
  }
  void * p_make_p( void * v ) {
    p_p_void * const ppv = (p_p_void *)( v );
    ppv->reset( new p_void );
    return ppv->get();
  }


  void nesi_string_init( void * init_arg, void * o, void * d )
  {
    string * v = (string *)o;
    lexp_t * l = (lexp_t *)d;
    if( !l->leaf_val.exists() ) {
      rt_err( "invalid attempt to use name/value list as string value. list was:" + str(*l) );
    }
    *v = l->leaf_val.str();
  }

  template< typename T >
  void * has_def_ctor_vect_push_back_t( void * v )
  {
    vector< T > * vv = (vector< T > *)v;
    vv->push_back( T() );
    return &vv->back();
  }
  template< typename T >
  void * has_def_ctor_make_p( void * v ) {
    shared_ptr< T > * const p = (shared_ptr< T > *)( v );
    p->reset( new T );
    return p->get();
  }

  make_p_t * string_make_p = &has_def_ctor_make_p< string >;
  make_p_t * string_vect_push_back = &has_def_ctor_vect_push_back_t< string >;

  void nesi_init( void ) { }




#include"gen/nesi.cc.nesi_gen.cc"

}
