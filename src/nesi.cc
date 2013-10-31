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
    void * v = pt->make_p( init_arg, o, d );
    pt->init( pt->init_arg, v, d );
  }

  void vect_init( void * init_arg, void * o, void * d )
  {
    tinfo_t * const pt = (tinfo_t *)( init_arg );
    lexp_t * l = (lexp_t *)d;
    if( l->leaf_val.exists() ) {
      rt_err( "invalid attempt to use string as name/value list for vector init. string was:" + str(*l) );
    }
    for( vect_lexp_nv_t::iterator i = l->kids.begin(); i != l->kids.end(); ++i ) {
      void * rpv = pt->vect_push_back( o );
      // note: for vector initialization, i->n (the name of the name/value pair) is ignored.
      pt->init( pt->init_arg, rpv, i->v.get() ); 
    }
  }
  void populate_nvm_from_lexp( lexp_t * const l, lexp_name_val_map_t & nvm )
  {
    if( l->leaf_val.exists() ) {
      rt_err( "invalid attempt to use string as name/value list. string was:" + str(*l) );
    }
    for( vect_lexp_nv_t::iterator i = l->kids.begin(); i != l->kids.end(); ++i ) {
      bool const did_ins = nvm.insert( std::make_pair( i->n, i->v ) ).second;
      if( !did_ins ) { rt_err( "invalid duplicate name '"+i->n.str()+"' in name/value list" ); }
    }
  }

  void init_var_from_nvm( lexp_name_val_map_t & nvm, vinfo_t const * const vi, void * rpv )
  {
    tinfo_t * const pt = vi->tinfo;
    sstr_t ss_vname;
    ss_vname.borrow_from_string( vi->vname );
    lexp_name_val_map_t::const_iterator nvmi = nvm.find( ss_vname );
    p_lexp_t di;
    if( nvmi != nvm.end() ) { di = nvmi->second; }
    if( !di && vi->default_val ) { di = parse_lexp( vi->default_val ); }
    if( !di && vi->req ) { rt_err( strprintf( "missing required value for var '%s'", vi->vname ) ); } 
    // FIXME: distinguish types that are 'okay' without init (nesi struct, pointer, vector), 
    // and those that are not (all others / basic types, maybe with expections (i.e. string) ).
    if( di ) { pt->init( pt->init_arg, rpv, di.get() ); }
  }

  // assumes o is a (`ci->cname` *). adjusts ci and o such that:
  // o is a (`ci->cname` *) and that o is the most derived legal pointer to the object.
  void make_most_derived( cinfo_t const * & ci, void * & o )
  {
    nesi * const no = ci->cast_cname_to_nesi( o );
    ci = no->get_cinfo();
    o = ci->cast_nesi_to_cname(no);
    assert_st( o ); // dynamic cast should never fail since get_cinfo() told us o was of the req'd type
  }

  // assumes o is a (`vc->cname` *)
  void nesi_struct_init_rec( cinfo_t const * ci, void * o, lexp_name_val_map_t & nvm ) 
  {
    for( cinfo_t const * const * bci = ci->bases; *bci; ++bci ) { // handle bases
      nesi_struct_init_rec( *bci, (*bci)->cast_nesi_to_cname( ci->cast_cname_to_nesi( o ) ), nvm );
    }
    for( uint32_t i = 0; ci->vars[i].exists(); ++i ) {
      init_var_from_nvm( nvm, &ci->vars[i], ci->get_field( o, i ) );
    }    
  }

  void nesi_struct_init( void * init_arg, void * o, void * d )
  {
    lexp_t * l = (lexp_t *)d;
    lexp_name_val_map_t nvm;
    populate_nvm_from_lexp( l, nvm );

    cinfo_t const * ci = (cinfo_t const *)( init_arg );
    make_most_derived( ci, o );
    nesi_struct_init_rec( ci, o, nvm );
  }


  cinfo_t const * get_derived_by_tid( cinfo_t const * const pc, char const * tid_str )
  {
    cinfo_t const * const * dci;
    for( dci = pc->derived; *dci; ++dci ) { 
      if( !strcmp(tid_str,(*dci)->tid_str) ) { return *dci; }
      if( (*dci)->tid_vix == uint32_t_const_max ) { // if no change in tid_vix, recurse
	cinfo_t const * ret = get_derived_by_tid( *dci, tid_str );
	if( ret ) { return ret; }
      }
    }
    return 0;
  }

  void nesi_struct_make_p_rec( p_nesi * pn, cinfo_t const * ci, lexp_name_val_map_t & nvm )
  {
    while( ci->tid_vix != uint32_t_const_max )
    {
      string tid_str;
      vinfo_t const * const vi = &ci->vars[ci->tid_vix];
      assert( !strcmp(vi->tinfo->tname,"string") ); // tid var must be of type string
      assert( vi->req ); // tid var must be required
      init_var_from_nvm( nvm, vi, &tid_str );
      cinfo_t const * const dci = get_derived_by_tid( ci, tid_str.c_str() );
      if( !dci ) {
	rt_err( strprintf( "type id str of '%s' did not match any derived class of %s\n", 
			   str(tid_str).c_str(), str(ci->cname).c_str() ) );
      }
      ci = dci;
    }
    ci->make_p_nesi( pn );
  }

  void * nesi_struct_make_p( void const * init_arg, void * o, void * d )
  {
    lexp_t * l = (lexp_t *)d;
    lexp_name_val_map_t nvm;
    populate_nvm_from_lexp( l, nvm );

    cinfo_t const * const ci = (cinfo_t *)( init_arg );
    p_nesi pn;
    nesi_struct_make_p_rec( &pn, ci, nvm );
    return ci->set_p_cname_from_p_nesi( &pn, o );
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
  void * vect_make_p( void const * init_arg, void * o, void * d ) { 
    p_vect_char * const pvc = (p_vect_char *)( o );
    pvc->reset( new vect_char );
    return pvc->get();
  }
  void * p_vect_push_back( void * v )
  {
    vect_p_void * vpv = (vect_p_void *)v;
    vpv->push_back( p_void() );
    return &vpv->back();
  }
  void * p_make_p( void const * init_arg, void * o, void * d ) {
    p_p_void * const ppv = (p_p_void *)( o );
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
  void * has_def_ctor_make_p( void const * init_arg, void * o, void * d ) {
    shared_ptr< T > * const p = (shared_ptr< T > *)( o );
    p->reset( new T );
    return p->get();
  }

  make_p_t * string_make_p = &has_def_ctor_make_p< string >;
  vect_push_back_t * string_vect_push_back = &has_def_ctor_vect_push_back_t< string >;

#include"gen/nesi.cc.nesi_gen.cc"

}
