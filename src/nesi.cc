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

  struct ndb_frame_t {
    char const * key;
    uint32_t os_ix;
    uint32_t xml_ix;
  };
  typedef vector< ndb_frame_t > vect_ndb_frame_t;

  struct nesi_dump_buf_t {
    string os;   // lexp-format output (always created)
    bool xml;
    string xos;  // xml-format output (optional)
    vect_ndb_frame_t frames;

    void begin_list( void ) { os += "("; }
    void add_sep_to_list( void ) { os += ","; }
    void end_list( void ) { os += ")"; }
    
    nesi_dump_buf_t( void ) : xml(0) { }
    void set_key( char const * key_ ) { } // key of next key/value pair
    void write_key_value_if_not_default( ) { }
  };
  

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
  bool p_nesi_dump( tinfo_t const * tinfo, void * o, nesi_dump_buf_t * ndb ) {
    if( !bool( *((p_void *)( o )) ) ) { return 1;}
    tinfo_t * const pt = (tinfo_t *)( tinfo->init_arg );
    return pt->nesi_dump( pt, ((p_void *)o)->get(), ndb );
  }
  void p_nesi_help( tinfo_t const * tinfo, void * o, void * os, 
		    bool const show_all, void * help_args, uint32_t help_ix ) {
    tinfo_t * const pt = (tinfo_t *)( tinfo->init_arg ); 
    void * fo = o ? ((p_void *)o)->get() : 0; // note: fo is NULL is o is NULL, or if o points to NULL
    return pt->nesi_help( pt, fo, os, show_all, help_args, help_ix ); // as per above, fo may be null
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
  bool vect_nesi_dump( tinfo_t const * tinfo, void * o, nesi_dump_buf_t * ndb ) {
    bool at_default = 1;
    vect_char & vc = *(vect_char *)( o );
    ndb->begin_list();
    tinfo_t * const pt = (tinfo_t *)( tinfo->init_arg );
    // FIXME: deal with at_default items (force print? not possible for pointers? need new iface? implicit default str?)
    uint32_t sz = vc.size() / pt->sz_bytes;
    assert_st( sz * pt->sz_bytes == vc.size() );
    char * cur = &vc.front();
    for( uint32_t i = 0; i < sz; ++i, cur += pt->sz_bytes )
    {
      if( at_default ) {
	at_default = 0; // non-empty vector is always not at default
      } else {
	ndb->add_sep_to_list();
      }
      ndb->os += "_=";
      //printf( "pt->tname=%s\n", str(pt->tname).c_str() );
      bool const li_at_default = pt->nesi_dump( pt, cur, ndb );
      if( li_at_default ) { // for lists, we want to print all items, even if they are at default.
	//os << pt->
      }       
    }
    ndb->end_list();
    return at_default;
  }
  void vect_nesi_help( tinfo_t const * tinfo, void * o, void * os, 
		    bool const show_all, void * help_args, uint32_t help_ix ) {
    tinfo_t * const pt = (tinfo_t *)( tinfo->init_arg );
    void * fo = 0;
    if( o ) { // note: use first elem, or NULL if vect is empty or NULL itself
      vect_char & vc = *(vect_char *)( o );
      if( vc.size() ) { fo = (&vc.front()); }
    } 
    return pt->nesi_help( pt, fo, os, show_all, help_args, help_ix ); // note: as per above, fo may be null
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
  void make_most_derived( cinfo_t const * & ci, void * & o, bool const null_okay = 0 )
  {
    if( !o ) { if( null_okay ) { return; } assert_st(0); }
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


  void nesi_struct_hier_help( cinfo_t const * const ci, string * os, string & prefix, bool const show_all )
  {
    if( (!show_all) && ci->hide ) { return; } // skip if class is hidden. note: will ignore any derived classes as well.
    *os += prefix;
    //if( !prefix.empty() ) { return; } // type can't be created, and we're not at the top: do nothing
    if( ci->tid_str ) { *os += string(ci->tid_str) + ":  "; }
    *os += string(ci->help);
    if( ci->tid_vix != uint32_t_const_max )  {
      *os += ". subtypes when "+string(ci->vars[ci->tid_vix].vname)+"=mode_name:";
    } else { *os += ". subtypes:"; }
    *os += string("\n");
    uint32_t const orig_prefix_sz = prefix.size();
    prefix += "|   ";
    for( cinfo_t const * const * dci = ci->derived; *dci; ++dci ) { 
      nesi_struct_hier_help( *dci, os, prefix, show_all );
    }
    prefix.resize( orig_prefix_sz );
  }

  void nesi_struct_vars_help_rec( string * const os, cinfo_t const * ci, bool const show_all )
  {
    for( cinfo_t const * const * bci = ci->bases; *bci; ++bci ) { // handle bases
      nesi_struct_vars_help_rec( os, *bci, show_all );
    }
    for( uint32_t i = 0; ci->vars[i].exists(); ++i ) {
      vinfo_t const * vi = &ci->vars[i];
      char const * hidden = "";
      if( vi->hide ) { if( !show_all ) { continue; } hidden = "(hidden)"; } // skip if hidden
      string req_opt_def = "(optional)";
      if( vi->req ) { req_opt_def = "(required)"; assert_st( !vi->default_val ); }
      else if( vi->default_val ) { req_opt_def = strprintf("(default='%s')",vi->default_val ); }
      *os += strprintf( "  %s: %s%s type=%s  --  %s\n", vi->vname, 
			hidden, req_opt_def.c_str(), vi->tinfo->tname, vi->help );
    }
  }

  vinfo_t const * nesi_struct_find_var( cinfo_t const * ci, void * o, string const & vname, void * & ret_field ) {
    for( cinfo_t const * const * bci = ci->bases; *bci; ++bci ) { // handle bases
      void * bo = o; // may be null
      if( o ) { bo = (*bci)->cast_nesi_to_cname( ci->cast_cname_to_nesi( o ) ); }
      vinfo_t const * ret = nesi_struct_find_var( *bci, bo, vname, ret_field );
      if( ret ) { return ret; }
    }
    for( uint32_t i = 0; ci->vars[i].exists(); ++i ) {
      vinfo_t const * vi = &ci->vars[i];
      if( vi->vname == vname ) { 
	if( o ) { ret_field = ci->get_field( o, i ); }
	return vi; 
      }
    }    
    return 0;
  }


  void nesi_struct_nesi_help( tinfo_t const * tinfo, void * o, void * os_, 
			      bool const show_all, void * help_args_, uint32_t help_ix ) {
    cinfo_t const * ci = (cinfo_t const *)( tinfo->init_arg );
    make_most_derived( ci, o, 1 ); // null ok here
    string * os = (string *)os_; 
    vect_string * help_args = (vect_string *)help_args_; 
    assert_st( help_args );
    if( help_ix < help_args->size() ) { // use a help-arg to decend
      void * fo = 0;
      vinfo_t const * vi = nesi_struct_find_var( ci, o, help_args->at(help_ix), fo ); // may or may not set fo
      if( !vi ) { rt_err( strprintf("struct '%s' has no field '%s', so help cannot be provided for it.",
				    ci->cname, help_args->at(help_ix).c_str() ) ); }
      *os += strprintf( "DECENDING TO DEATILED HELP FOR field '%s' of type=%s of struct '%s'\n",
			vi->vname, vi->tinfo->tname, ci->cname );
      assert_st( vi->tinfo->nesi_help );
      vi->tinfo->nesi_help( vi->tinfo, fo, os, show_all, help_args, help_ix + 1 );
    } else { // leaf case, emit help for this struct
      assert_st( help_ix == help_args->size() );
      string prefix;
      *os += strprintf( "\nTYPE INFO:\n" );
      nesi_struct_hier_help( ci, os, prefix, show_all );
      *os += strprintf( "\nFIELDS:\n" );
      nesi_struct_vars_help_rec( os, ci, show_all );
    }
  } 

  void nesi_struct_nesi_dump_rec( bool * const at_default, nesi_dump_buf_t * ndb,
				  cinfo_t const * ci, void * o )
  {
    for( cinfo_t const * const * bci = ci->bases; *bci; ++bci ) { // handle bases
      nesi_struct_nesi_dump_rec( at_default, ndb, 
				 *bci, (*bci)->cast_nesi_to_cname( ci->cast_cname_to_nesi( o ) ) );
    }
    for( uint32_t i = 0; ci->vars[i].exists(); ++i ) {
      vinfo_t const * vi = &ci->vars[i];
      //printf("vname=%s\n",vi->vname);
      uint32_t orig_os_sz = ndb->os.size();
      if( !*at_default ) { ndb->add_sep_to_list(); } // if we're printed a field already, add a comma
      ndb->os += string(vi->vname) + "=";
      uint32_t const os_val_b = ndb->os.size();
      bool const var_at_default = vi->tinfo->nesi_dump( vi->tinfo, ci->get_field( o, i ), ndb );
      if( (!var_at_default) && // printed field, and ( no default or not equal to default ). keep new part of os.
	  ( (!vi->default_val) || strncmp(vi->default_val, &(ndb->os)[os_val_b], ndb->os.size()-os_val_b ) ) ) {
	*at_default = 0;
      } else { // otherwise, remove anything we added to os for this field
	ndb->os.resize( orig_os_sz );
      }
    }
  }

  bool nesi_struct_nesi_dump( tinfo_t const * tinfo, void * o, nesi_dump_buf_t * ndb ) {
    cinfo_t const * ci = (cinfo_t const *)( tinfo->init_arg );
    make_most_derived( ci, o ); 
    ndb->begin_list();
    bool at_default = 1;
    nesi_struct_nesi_dump_rec( &at_default, ndb, ci, o );    
    ndb->end_list();
    return at_default;
  }

  std::ostream & operator<<(std::ostream & top_ostream, nesi const & v)
  {
    nesi_dump_buf_t ndb;
    cinfo_t const * ci = v.get_cinfo();
    nesi_struct_nesi_dump( ci->tinfo, ci->cast_nesi_to_cname((nesi *)&v), &ndb );
    return top_ostream << ndb.os;
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
  bool with_op_left_shift_nesi_dump( tinfo_t const * tinfo, void * o, nesi_dump_buf_t * ndb ) {
    T * v = (T *)o;
    std::ostringstream oss;
    oss << *v;
    ndb->os += oss.str();
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
  void leaf_type_nesi_help( tinfo_t const * tinfo, void * o, void * os_, 
			    bool const show_all, void * help_args_, uint32_t help_ix ) {
    string * os = (string *)os_; 
    vect_string * help_args = (vect_string *)help_args_; 
    assert_st( help_args );
    if( help_ix < help_args->size() ) { // use a help-arg to decend
      rt_err( strprintf("leaf type '%s' has no fields at all, certainly not '%s', so help cannot be provided for it.",
			tinfo->tname, help_args->at(help_ix).c_str() ) );
    } else { // leaf case, emit help for this leaf type
      assert_st( help_ix == help_args->size() );
      *os += strprintf( "LEAF TYPE INFO: %s\n", (char const *)tinfo->init_arg );
    }
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
