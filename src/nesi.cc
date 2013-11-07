#include"nesi.H"
#include<cassert>
#include<vector>
#include<boost/shared_ptr.hpp>
using boost::shared_ptr;
using std::vector;

#include<boost/lexical_cast.hpp>


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

  void p_init( tinfo_t const * tinfo, void * o, void * d )
  {
    if( !d ) { return; } // no_value_init --> null pointer
    tinfo_t * const pt = (tinfo_t *)( tinfo->init_arg );
    void * v = pt->make_p( pt, o, d );
    pt->init( pt, v, d );
  }
  bool p_nesi_dump( tinfo_t const * tinfo, void * o, void * os ) {
    if( !bool( *((p_void *)( o )) ) ) { return 1;}
    tinfo_t * const pt = (tinfo_t *)( tinfo->init_arg );
    return pt->nesi_dump( pt, ((p_void *)o)->get(), os );
  }

  void vect_init( tinfo_t const * tinfo, void * o, void * d )
  {
    if( !d ) { return; } // no_value_init --> empty vector
    tinfo_t * const pt = (tinfo_t *)( tinfo->init_arg );
    lexp_t * l = (lexp_t *)d;
    if( l->leaf_val.exists() ) {
      rt_err( "invalid attempt to use string as name/value list for vector init. string was:" + str(*l) );
    }
    ++l->use_cnt;
    for( vect_lexp_nv_t::iterator i = l->kids.begin(); i != l->kids.end(); ++i ) {
      void * rpv = pt->vect_push_back( o );
      // note: for vector initialization, i->n (the name of the name/value pair) is ignored.
      pt->init( pt, rpv, i->v.get() ); 
    }
  }
  bool vect_nesi_dump( tinfo_t const * tinfo, void * o, void * os_ ) {
    bool at_default = 1;
    vect_char & vc = *(vect_char *)( o );
    string * os = (string *)os_;
    *os += "(";
    tinfo_t * const pt = (tinfo_t *)( tinfo->init_arg );
    // FIXME: deal with at_default items (force print? not possible for pointers? need new iface? implicit default str?)
    uint32_t sz = vc.size() / pt->sz_bytes;
    assert_st( sz * pt->sz_bytes == vc.size() );
    char * cur = &((vect_char *)o)->front();
    for( uint32_t i = 0; i < sz; ++i, cur += pt->sz_bytes )
    {
      if( at_default ) {
	at_default = 0; // non-empty vector is always not at default
      } else {
	*os += ",";
      }
      *os += "_=";
      //printf( "pt->tname=%s\n", str(pt->tname).c_str() );
      bool const li_at_default = pt->nesi_dump( pt, cur, os );
      if( li_at_default ) { // for lists, we want to print all items, even if they are at default.
	//os << pt->
      }       
    }
    *os += ")";
    return at_default;
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
    if( !di ) { assert_st( pt->no_init_okay ); } // nesi_gen.py should have checked to prevent this
    pt->init( pt, rpv, di.get() ); // note: di.get() may be null, yielding type-specific no-value init 
  }

  // assumes o is a (`ci->cname` *). adjusts ci and o such that:
  // o is a (`ci->cname` *) and that o is the most derived legal pointer to the object.
  void make_most_derived( cinfo_t const * & ci, void * & o )
  {
    nesi * const no = ci->cast_cname_to_nesi( o ); // note: drop const
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

  void nesi_struct_init( tinfo_t const * tinfo, void * o, void * d )
  {
    lexp_name_val_map_t nvm;
    lexp_t * l = (lexp_t *)d;
    if( l ) { // no_value_init has same semantics as empty list init
      ++l->use_cnt;
      populate_nvm_from_lexp( l, nvm );
    }
    cinfo_t const * ci = (cinfo_t const *)( tinfo->init_arg );
    make_most_derived( ci, o );
    nesi_struct_init_rec( ci, o, nvm );
  }


  void nesi_struct_hier_help( cinfo_t const * const ci, std::ostream & os, string & prefix )
  {
    if( ci->hide ) { return; } // skip if class is hidden. note: will ignore any derived classes as well.
    uint32_t const orig_prefix_sz = prefix.size();
    if( !ci->tid_str ) { 
      if( !prefix.empty() ) { return; } // type can't be created, and we're not at the top: do nothing
      os << ci->help << std::endl;
    } else {
      prefix += ci->tid_str;
      os << prefix << "   ----   " << ci->help << std::endl;
    }
    if( ci->tid_vix != uint32_t_const_max )  {
      if( !prefix.empty() ) { prefix += ","; }
      prefix += string(ci->vars[ci->tid_vix].vname)+"=";
    } else {
      
    }
    for( cinfo_t const * const * dci = ci->derived; *dci; ++dci ) { 
      nesi_struct_hier_help( *dci, os, prefix );
    }
    prefix.resize( orig_prefix_sz );
  }

  void nesi_struct_nesi_dump_rec( bool * const at_default, string * const os, 
				  cinfo_t const * ci, void * o )
  {
    for( cinfo_t const * const * bci = ci->bases; *bci; ++bci ) { // handle bases
      nesi_struct_nesi_dump_rec( at_default, os, 
				 *bci, (*bci)->cast_nesi_to_cname( ci->cast_cname_to_nesi( o ) ) );
    }
    for( uint32_t i = 0; ci->vars[i].exists(); ++i ) {
      vinfo_t const * vi = &ci->vars[i];
      //printf("vname=%s\n",vi->vname);
      uint32_t orig_os_sz = os->size();
      if( !*at_default ) { *os +=  ","; } // if we're printed a field already, add a comma
      *os += string(vi->vname) + "=";
      uint32_t const os_val_b = os->size();
      bool const var_at_default = vi->tinfo->nesi_dump( vi->tinfo, ci->get_field( o, i ), os );
      if( (!var_at_default) && // printed field, and ( no default or not equal to default ). keep new part of os.
	  ( (!vi->default_val) || strncmp(vi->default_val, &(*os)[os_val_b], os->size()-os_val_b ) ) ) {
	*at_default = 0;
      } else { // otherwise, remove anything we added to os for this field
	os->resize( orig_os_sz );
      }
    }
  }

  bool nesi_struct_nesi_dump( tinfo_t const * tinfo, void * o, void * os_ ) {
    cinfo_t const * ci = (cinfo_t const *)( tinfo->init_arg );
    make_most_derived( ci, o ); 
    string * os = (string *)os_;
    *os += "(";
    bool at_default = 1;
    nesi_struct_nesi_dump_rec( &at_default, os, ci, o );    
    *os += ")";
    return at_default;
  }

  std::ostream & operator<<(std::ostream & top_ostream, nesi const & v)
  {
    string os; // build result in a string buffer
    cinfo_t const * ci = v.get_cinfo();
    nesi_struct_nesi_dump( ci->tinfo, ci->cast_nesi_to_cname((nesi *)&v), &os );
    return top_ostream << os;
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
      init_var_from_nvm( nvm, vi, &tid_str ); // FIXME? increments use_cnt of lexp leaf used for tid_str ...
      cinfo_t const * const dci = get_derived_by_tid( ci, tid_str.c_str() );
      if( !dci ) {
	rt_err( strprintf( "type id str of '%s' did not match any derived class of %s\n", 
			   str(tid_str).c_str(), str(ci->cname).c_str() ) );
      }
      ci = dci;
    }
    ci->make_p_nesi( pn );
  }

  void * nesi_struct_make_p( tinfo_t const * tinfo, void * o, void * d )
  {
    lexp_t * l = (lexp_t *)d;
    lexp_name_val_map_t nvm;
    populate_nvm_from_lexp( l, nvm );

    cinfo_t const * const ci = (cinfo_t *)( tinfo->init_arg );
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
  void * vect_make_p( tinfo_t const * tinfo, void * o, void * d ) { 
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
  void * p_make_p( tinfo_t const * tinfo, void * o, void * d ) {
    p_p_void * const ppv = (p_p_void *)( o );
    ppv->reset( new p_void );
    return ppv->get();
  }

  // base type methods

  // shared base type functions
  template< typename T >
  void nesi_lexcast_init( tinfo_t const * tinfo, void * o, void * d )
  { 
    T * v = (T *)o;
    assert( d ); // no_value_init disabled
    //if( !d ) { *v = 0; return; } // no_value_init = 0
    char const * tstr = (char const *)tinfo->init_arg;
    lexp_t * l = (lexp_t *)d;
    if( !l->leaf_val.exists() ) {
      rt_err( "invalid attempt to use name/value list as "+string(tstr)+" value. list was:" + str(*l) );
    }
    ++l->use_cnt;
    string const s = l->leaf_val.str();  
    try { *v = boost::lexical_cast< T >( s ); }
    catch( boost::bad_lexical_cast & e ) { rt_err( strprintf("can't convert '%s' to %s.", s.c_str(), tstr ) ); }
  }
  // note: always prints/clear pending
  template< typename T >
  bool with_op_left_shift_nesi_dump( tinfo_t const * tinfo, void * o, void * os_ ) {
    T * v = (T *)o;
    string * os = (string *)os_;
    std::ostringstream oss;
    oss << *v;
    *os += oss.str();
    return 0;
  }
  template< typename T >
  void * has_def_ctor_vect_push_back_t( void * v )
  {
    vector< T > * vv = (vector< T > *)v;
    vv->push_back( T() );
    return &vv->back();
  }
  template< typename T >
  void * has_def_ctor_make_p( tinfo_t const * tinfo, void * o, void * d ) {
    shared_ptr< T > * const p = (shared_ptr< T > *)( o );
    p->reset( new T );
    return p->get();
  }

  // string
  void nesi_string_init( tinfo_t const * tinfo, void * o, void * d )
  {
    assert( d ); // no_value_init disabled
    //if( !d ) { return; } // no_value_init --> empty string
    string * v = (string *)o;
    lexp_t * l = (lexp_t *)d;
    if( !l->leaf_val.exists() ) {
      rt_err( "invalid attempt to use name/value list as string value. list was:" + str(*l) );
    }
    ++l->use_cnt;
    *v = l->leaf_val.str();
  }
  make_p_t * string_make_p = &has_def_ctor_make_p< string >;
  vect_push_back_t * string_vect_push_back = &has_def_ctor_vect_push_back_t< string >;
  nesi_dump_t * string_nesi_dump = &with_op_left_shift_nesi_dump< string >;
  void *string_init_arg = (void *)"string";

  // uint64_t  
  init_t * nesi_uint64_t_init = &nesi_lexcast_init< uint64_t >;
  make_p_t * uint64_t_make_p = &has_def_ctor_make_p< uint64_t >;
  vect_push_back_t * uint64_t_vect_push_back = &has_def_ctor_vect_push_back_t< uint64_t >;
  nesi_dump_t * uint64_t_nesi_dump = &with_op_left_shift_nesi_dump< uint64_t >;
  void *uint64_t_init_arg = (void *)"uint64_t (64-bit unsigned integer)";

  // double  
  init_t * nesi_double_init = &nesi_lexcast_init< double >;
  make_p_t * double_make_p = &has_def_ctor_make_p< double >;
  vect_push_back_t * double_vect_push_back = &has_def_ctor_vect_push_back_t< double >;
  nesi_dump_t * double_nesi_dump = &with_op_left_shift_nesi_dump< double >;
  void *double_init_arg = (void *)"double (double precision floating point number)";

#include"gen/nesi.cc.nesi_gen.cc"

}
