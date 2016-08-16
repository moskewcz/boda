// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"nesi.H"
#include<cassert>
#include<vector>
#include<memory>
using std::shared_ptr;
using std::vector;

#include<boost/lexical_cast.hpp>
#include<boost/algorithm/string.hpp>


#include<string>
using std::string;

#include"lexp.H"
#include"str_util.H"
#include"xml_util.H"
#include"geom_prim.H"
#include"cnn_op.H"
#include"ext/half.hpp"

namespace boda 
{
  void nesi_init_and_check_unused_from_nia(  nesi_init_arg_t * nia, tinfo_t const * ti, void * o ) {
    assert_st( nia->l );
    ti->init( nia, ti, o );
    vect_string path;
    lexp_check_unused( nia->l.get(), path );
  }

  void nesi_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * parent, tinfo_t const * ti,  void * o ) {
    lexp_name_val_map_t nvm( lexp, parent );
    nesi_init_and_check_unused_from_nia( &nvm, ti, o );
  }

  struct xml_attr_t {
    string n;
    string v;
    xml_attr_t( string const & n_, string const & v_ ) : n(n_), v(v_) { }
    void print( std::ostream & os );
  };
  typedef vector< xml_attr_t > vect_xml_attr_t;
  void xml_attr_t::print( std::ostream & os ) {
    os << n << "=\"" << xml_escape(v) << "\"";
  }

  struct xml_elem_t;
  typedef shared_ptr< xml_elem_t > p_xml_elem_t;  
  typedef vector< p_xml_elem_t > vect_p_xml_elem_t;
  struct xml_elem_t {
    string name;
    vect_xml_attr_t attrs;
    vect_p_xml_elem_t elems;
    bool empty( void ) const { return attrs.empty() && elems.empty(); }
    void print( std::ostream & os, string & prefix );
    xml_elem_t( string const & name_ ) : name(name_) { }
  };
  
  void xml_elem_t::print( std::ostream & os, string & prefix ) {
    os << prefix << "<" << name;
    for( vect_xml_attr_t::iterator i = attrs.begin(); i != attrs.end(); ++i ) {
      os << " "; i->print( os );
    }
    if( elems.empty() ) { os << " />\n"; }
    else {
      os << ">\n";
      uint32_t const prefix_sz_orig = prefix.size();
      prefix += "\t";
      for( vect_p_xml_elem_t::iterator i = elems.begin(); i != elems.end(); ++i ) {
	(*i)->print( os, prefix );
      }
      prefix.resize( prefix_sz_orig );
      os << prefix << "</" << name<< ">\n"; 
    }
  }

  struct nesi_frame_t { 
    bool at_default;
    p_xml_elem_t parent_xn;
    uint32_t os_sz_orig;
    uint32_t os_sz_val_begin;
    nesi_frame_t( void ) : at_default(1), os_sz_orig(uint32_t_const_max), os_sz_val_begin(uint32_t_const_max) { }
    void clear( void ) { parent_xn.reset(); os_sz_orig = uint32_t_const_max; os_sz_val_begin = uint32_t_const_max; }
    bool is_clear( void ) const { 
      return (!parent_xn) && (os_sz_orig == uint32_t_const_max) && (os_sz_val_begin == uint32_t_const_max); 
    }
  };

  struct nesi_dump_buf_t {
    // outputs
    string os;   // lexp-format output (always created). appended to by each dump
    p_xml_elem_t xn; // xml rep of dumped nesi object. filled in by each dump if not null
    bool is_non_leaf; // set by each dump call if the value was a struct or list.

    void begin_list( void ) { os += "("; }
    void end_list( void ) { os += ")"; is_non_leaf = 1;}
    void new_list_val( nesi_frame_t & nf, string const & name ) {
      assert( nf.is_clear() );
      nf.os_sz_orig = os.size();
      if( !nf.at_default ) { os += ","; } // if we're printed a field already, add a comma
      os += name + "=";
      nf.os_sz_val_begin = os.size();
      if( xn ) { nf.parent_xn = xn; xn.reset( new xml_elem_t( name ) ); }
    }
    void commit_list_val( nesi_frame_t & nf ) {
      nf.at_default = 0;
      if( xn ) { 
	if( !is_non_leaf ) {
	  assert_st( xn->empty() );
	  nf.parent_xn->attrs.push_back( xml_attr_t( xn->name, string( os, nf.os_sz_val_begin ) ) );
	} else {
	  nf.parent_xn->elems.push_back( xn ); 
	}
	xn = nf.parent_xn; 
      }
      nf.clear();
    }
    void abort_list_val( nesi_frame_t & nf ) {
      os.resize( nf.os_sz_orig );	
      if( xn ) { xn = nf.parent_xn; }
      nf.clear();
    }
    nesi_dump_buf_t( void ) { }
  };
  

  typedef shared_ptr< void > p_void;
  typedef vector< char > vect_char;

  typedef shared_ptr< p_void > p_p_void;  
  typedef vector< vector< char > > vect_vect_char;
  typedef shared_ptr< vect_char > p_vect_char;
  typedef vector< p_void > vect_p_void;

  void p_init( nesi_init_arg_t * nia, tinfo_t const * tinfo, void * o )
  {
    if( !nia->l ) { return; } // no_value_init --> null pointer
    tinfo_t * const pt = (tinfo_t *)( tinfo->init_arg );
    void * v = pt->make_p( nia, pt, o );
    pt->init( nia, pt, v );
  }
  bool p_nesi_dump( tinfo_t const * tinfo, void * o, nesi_dump_buf_t * ndb ) {
    if( !bool( *((p_void *)( o )) ) ) { return 1;}
    tinfo_t * const pt = (tinfo_t *)( tinfo->init_arg );
    return pt->nesi_dump( pt, ((p_void *)o)->get(), ndb );
  }

  string const deep_help_str( "-" );

  void p_nesi_help( tinfo_t const * tinfo, void * o, string * os, string & prefix,
		    bool const show_all, vect_string * help_args, uint32_t help_ix ) {
    tinfo_t * const pt = (tinfo_t *)( tinfo->init_arg ); 
    void * fo = o ? ((p_void *)o)->get() : 0; // note: fo is NULL is o is NULL, or if o points to NULL
    return pt->nesi_help( pt, fo, os, prefix, show_all, help_args, help_ix ); // as per above, fo may be null
  }

  void vect_init( nesi_init_arg_t * nia, tinfo_t const * tinfo, void * o )
  {
    if( !nia->l ) { return; } // no_value_init --> empty vector
    tinfo_t * const pt = (tinfo_t *)( tinfo->init_arg );
    lexp_t * l = nia->l.get();
    if( l->leaf_val.exists() ) {
      rt_err( "invalid attempt to use string as name/value list for vector init. string was:" + str(*l) );
    }
    ++l->use_cnt;
    for( vect_lexp_nv_t::iterator i = l->kids.begin(); i != l->kids.end(); ++i ) {
      void * rpv = pt->vect_push_back( o );
      // note: for vector initialization, i->n (the name of the name/value pair) is ignored.
      lexp_name_val_map_t nvm( i->v, nia );
      try { pt->init( &nvm, pt, rpv ); }
      catch( rt_exception & rte ) {
	rte.err_msg = "list elem " + str(i-l->kids.begin()) + ": " + rte.err_msg;
	throw;
      }
    }
  }
  bool vect_nesi_dump( tinfo_t const * tinfo, void * o, nesi_dump_buf_t * ndb ) {
    nesi_frame_t nf;
    vect_char & vc = *(vect_char *)( o );
    ndb->begin_list();
    tinfo_t * const pt = (tinfo_t *)( tinfo->init_arg );
    uint32_t sz = vc.size() / pt->sz_bytes;
    assert_st( sz * pt->sz_bytes == vc.size() );
    char * cur = &vc.front();
    for( uint32_t i = 0; i < sz; ++i, cur += pt->sz_bytes )
    {
      //printf( "pt->tname=%s\n", str(pt->tname).c_str() );
      ndb->new_list_val( nf, "li_"+str(i) );
      // note: ret value of dump ignored; for lists, we want to print all items, even if they are at default.
      pt->nesi_dump( pt, cur, ndb ); 
      ndb->commit_list_val( nf );
    }
    ndb->end_list();
    return nf.at_default;
  }

  //  note: we could use a help arg here as a list index. this seems
  //  slightly more correct and complete, but also not to useful and
  //  confusing. hmm.
  void vect_nesi_help( tinfo_t const * tinfo, void * o, string * os, string & prefix,
		    bool const show_all, vect_string * help_args, uint32_t help_ix ) {
    tinfo_t * const pt = (tinfo_t *)( tinfo->init_arg );
    void * fo = 0;
    if( o ) { // note: use first elem, or NULL if vect is empty or NULL itself
      vect_char & vc = *(vect_char *)( o );
      if( vc.size() ) { fo = (&vc.front()); }
    } 
    return pt->nesi_help( pt, fo, os, prefix, show_all, help_args, help_ix ); // note: as per above, fo may be null
  }

  void init_var_from_nvm( nesi_init_arg_t * nia, vinfo_t const * const vi, void * rpv )
  {
    tinfo_t * const pt = vi->tinfo;
    nesi_init_arg_t * found_scope = 0;
    p_lexp_t di = nia->find( vi->vname, &found_scope );
    if( !di && vi->default_val ) { di = parse_lexp( vi->default_val ); found_scope = nia; }
    if( !di && vi->req ) { rt_err( strprintf( "missing required value for var '%s'", vi->vname ) ); } 
    if( !di ) { assert_st( pt->no_init_okay ); } // nesi_gen.py should have checked to prevent this
    else { assert_st( found_scope ); }
    // note: if pt->no_init_okay, then di.get() may be null, yielding type-specific no-value init 
    try { 
      lexp_name_val_map_t nvm( di, found_scope ); // note lexical (non-dynamic) scoping here
      pt->init( &nvm, pt, rpv ); 
    } 
    catch( rt_exception & rte ) {
      rte.err_msg = "var '" + str(vi->vname) + "': " + rte.err_msg;
      throw;
    }
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
  void nesi_struct_init_rec( nesi_init_arg_t * nia, cinfo_t const * ci, void * o ) 
  {
    for( cinfo_t const * const * bci = ci->bases; *bci; ++bci ) { // handle bases
      nesi_struct_init_rec( nia, *bci, (*bci)->cast_nesi_to_cname( ci->cast_cname_to_nesi( o ) ) );
    }
    for( uint32_t i = 0; ci->vars[i].exists(); ++i ) {
      init_var_from_nvm( nia, &ci->vars[i], ci->get_field( o, i ) );
    }    
  }

  void nesi_struct_init( nesi_init_arg_t * nia, tinfo_t const * tinfo, void * o )
  {
    if( nia->l ) { nia->init_nvm(); ++nia->l->use_cnt; }
    else { } // for no_value_init case we leave nia empty (with an nvm_init=0, but that doesn't matter)
    cinfo_t const * ci = (cinfo_t const *)( tinfo->init_arg );
    make_most_derived( ci, o );
    nesi_struct_init_rec( nia, ci, o );
  }


  void nesi_struct_hier_help( cinfo_t const * const ci, string * os, string & prefix, bool const show_all )
  {
    if( (!show_all) && ci->hide ) { return; } // skip if class is hidden. note: will ignore any derived classes as well.
    *os += prefix;
    //if( !prefix.empty() ) { return; } // type can't be created, and we're not at the top: do nothing
    if( ci->tid_str ) { *os += string(ci->tid_str) + ":  "; }
    if( ci->tid_vix != uint32_t_const_max )  {
      *os += ">" + string(ci->help) + " when "+string(ci->vars[ci->tid_vix].vname)+"=mode_name:";
    } else if( *(ci->derived) ) { *os += ">" + string(ci->help) + "; has subtypes:"; }
    else { *os += string(ci->help); }
    *os += string("\n");
    {
      string_scoped_prefixer_t ssp( prefix, "|   " );
      for( cinfo_t const * const * dci = ci->derived; *dci; ++dci ) { 
	nesi_struct_hier_help( *dci, os, prefix, show_all );
      }
    }
  }

  void nesi_struct_vars_help_rec( void * o, string * const os, string & prefix, cinfo_t const * ci, bool const show_all,
				  vect_string * help_args, uint32_t help_ix ) 
  {
    for( cinfo_t const * const * bci = ci->bases; *bci; ++bci ) { // handle bases
      nesi_struct_vars_help_rec( o, os, prefix, *bci, show_all, help_args, help_ix );
    }
    for( uint32_t i = 0; ci->vars[i].exists(); ++i ) {
      vinfo_t const * vi = &ci->vars[i];
      char const * hidden = "";
      if( vi->hide ) { if( !show_all ) { continue; } hidden = "(hidden)"; } // skip if hidden
      string req_opt_def = "(optional)";
      if( vi->req ) { req_opt_def = "(required)"; assert_st( !vi->default_val ); }
      else if( vi->default_val ) { req_opt_def = strprintf("(default='%s')",vi->default_val ); }
      *os += strprintf( "%s  %s: %s%s type=%s  --  %s\n", prefix.c_str(), vi->vname, 
			hidden, req_opt_def.c_str(), vi->tinfo->tname, vi->help );
      // for 'deep help' case, descend to all fields
      if( (help_ix < help_args->size()) && (help_args->at(help_ix) == deep_help_str) ) { 
	string_scoped_prefixer_t ssp( prefix, "  |   " );
	void * fo = 0;
	if( o ) { fo = ci->get_field( o, i ); }
	vi->tinfo->nesi_help( vi->tinfo, fo, os, prefix, show_all, help_args, help_ix );
      }
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


  void nesi_struct_nesi_help( tinfo_t const * tinfo, void * o, string * os, string & prefix,
			      bool const show_all, vect_string * help_args, uint32_t help_ix ) {
    cinfo_t const * ci = (cinfo_t const *)( tinfo->init_arg );
    make_most_derived( ci, o, 1 ); // null ok here
    assert_st( help_args );
    assert_st( help_ix <= help_args->size() );
    if( (help_ix < help_args->size()) && (help_args->at(help_ix) != deep_help_str) ) { // use a help-arg to decend
      void * fo = 0;
      vinfo_t const * vi = nesi_struct_find_var( ci, o, help_args->at(help_ix), fo ); // may or may not set fo
      if( !vi ) { 
	*os += strprintf("struct '%s' has no field '%s', so help cannot be provided for it.\n",
			 ci->cname, help_args->at(help_ix).c_str() );
      } else {
	*os += strprintf( "%sDESCENDING TO DETAILED HELP FOR field '%s' of type=%s of struct '%s'\n",
			  prefix.c_str(), vi->vname, vi->tinfo->tname, ci->cname );
	assert_st( vi->tinfo->nesi_help );
	vi->tinfo->nesi_help( vi->tinfo, fo, os, prefix, show_all, help_args, help_ix + 1 );
      }
    } else { // leaf or deep case, emit help for this struct
      *os += strprintf( "%sTYPE INFO:\n", prefix.c_str() );
      nesi_struct_hier_help( ci, os, prefix, show_all );
      *os += strprintf( "%sFIELDS:\n", prefix.c_str() );
      nesi_struct_vars_help_rec( o, os, prefix, ci, show_all, help_args, help_ix );
    }
  } 

  void nesi_struct_nesi_dump_rec( nesi_frame_t & nf, nesi_dump_buf_t * ndb, cinfo_t const * ci, void * o )
  {
    for( cinfo_t const * const * bci = ci->bases; *bci; ++bci ) { // handle bases
      nesi_struct_nesi_dump_rec( nf, ndb, *bci, (*bci)->cast_nesi_to_cname( ci->cast_cname_to_nesi( o ) ) );
    }
    for( uint32_t i = 0; ci->vars[i].exists(); ++i ) {
      vinfo_t const * vi = &ci->vars[i];
      //printf("vname=%s\n",vi->vname);
      ndb->new_list_val( nf, vi->vname );
      bool const var_at_default = vi->tinfo->nesi_dump( vi->tinfo, ci->get_field( o, i ), ndb );
      if( (!var_at_default) && // printed field, and ( no default or not equal to default ). keep new part of os.
	  ( (!vi->default_val) || strncmp(vi->default_val, &(ndb->os)[nf.os_sz_val_begin], ndb->os.size()-nf.os_sz_val_begin ) ) ) {
	ndb->commit_list_val( nf );
      } else { // otherwise, remove anything we added to os for this field
	ndb->abort_list_val( nf );
      }
    }
  }

  bool nesi_struct_nesi_dump( tinfo_t const * tinfo, void * o, nesi_dump_buf_t * ndb ) {
    cinfo_t const * ci = (cinfo_t const *)( tinfo->init_arg );
    make_most_derived( ci, o ); 
    ndb->begin_list();
    nesi_frame_t nf;
    nesi_struct_nesi_dump_rec( nf, ndb, ci, o );    
    ndb->end_list();
    return nf.at_default;
  }

  std::ostream & operator <<(std::ostream & top_ostream, nesi const & v)
  {
    nesi_dump_buf_t ndb;
    cinfo_t const * ci = v.get_cinfo();
    nesi_struct_nesi_dump( ci->tinfo, ci->cast_nesi_to_cname((nesi *)&v), &ndb );
    return top_ostream << ndb.os;
  }

  void nesi_dump_xml(std::ostream & top_ostream, nesi const & v, char const * const root_elem_name )
  {
    nesi_dump_buf_t ndb;
    ndb.xn.reset( new xml_elem_t( root_elem_name ) );
    cinfo_t const * ci = v.get_cinfo();
    nesi_struct_nesi_dump( ci->tinfo, ci->cast_nesi_to_cname((nesi *)&v), &ndb );
    string prefix;
    ndb.xn->print( top_ostream, prefix );
  }

  cinfo_t const * get_derived_by_tid( cinfo_t const * const pc, char const * tid_str )
  {
    cinfo_t const * const * dci;
    for( dci = pc->derived; *dci; ++dci ) { 
      if( (*dci)->tid_str && !strcmp(tid_str,(*dci)->tid_str) ) { return *dci; }
      if( (*dci)->tid_vix == uint32_t_const_max ) { // if no change in tid_vix, recurse
	cinfo_t const * ret = get_derived_by_tid( *dci, tid_str );
	if( ret ) { return ret; }
      }
    }
    return 0;
  }

  void nesi_struct_make_p_rec( nesi_init_arg_t * nia, p_nesi * pn, cinfo_t const * ci )
  {
    while( ci->tid_vix != uint32_t_const_max )
    {
      string tid_str;
      vinfo_t const * const vi = &ci->vars[ci->tid_vix];
      assert( !strcmp(vi->tinfo->tname,"string") ); // tid var must be of type string
      assert( vi->req ); // tid var must be required
      init_var_from_nvm( nia, vi, &tid_str ); // FIXME? increments use_cnt of lexp leaf used for tid_str ...
      cinfo_t const * const dci = get_derived_by_tid( ci, tid_str.c_str() );
      if( !dci ) {
	rt_err( strprintf( "type id str of '%s' did not match any derived class of %s\n", 
			   str(tid_str).c_str(), str(ci->cname).c_str() ) );
      }
      ci = dci;
    }
    ci->make_p_nesi( pn );
  }

  void * nesi_struct_make_p( nesi_init_arg_t * nia, tinfo_t const * tinfo, void * o )
  {
    assert_st( nia->l );
    nia->init_nvm();
    cinfo_t const * const ci = (cinfo_t *)( tinfo->init_arg );
    p_nesi pn;
    nesi_struct_make_p_rec( nia, &pn, ci );
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
  void * vect_make_p( nesi_init_arg_t * nia, tinfo_t const * tinfo, void * o ) { 
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
  void * p_make_p( nesi_init_arg_t * nia, tinfo_t const * tinfo, void * o ) {
    p_p_void * const ppv = (p_p_void *)( o );
    ppv->reset( new p_void );
    return ppv->get();
  }

  // base type methods

  // shared base type functions
  lexp_t * nesi_leaf_val_init_helper( char const * & tstr, nesi_init_arg_t * nia, tinfo_t const * tinfo )
  { 
    assert( nia->l ); // no_value_init disabled
    //if( !d ) { *v = 0; return; } // no_value_init = 0
    tstr = (char const *)tinfo->init_arg;
    lexp_t * l = nia->l.get();
    if( !l->leaf_val.exists() ) {
      rt_err( "invalid attempt to use name/value list as "+string(tstr)+" value. list was:" + str(*l) );
    }
    ++l->use_cnt;
    return l;
  }

  template< typename T >
  void nesi_lexcast_init( nesi_init_arg_t * nia, tinfo_t const * tinfo, void * o )
  { 
    T * v = (T *)o;
    char const * tstr = 0;
    string const s = nesi_leaf_val_init_helper(tstr,nia,tinfo)->leaf_val.str();  
    try { *v = boost::lexical_cast< T >( s ); }
    catch( boost::bad_lexical_cast & e ) { rt_err( strprintf("can't convert '%s' to %s.", s.c_str(), tstr ) ); }
  }
  template< typename T >
  void nesi_str_parts_init( nesi_init_arg_t * nia, tinfo_t const * tinfo, void * o )
  { 
    T * v = (T *)o;
    char const * tstr = 0;
    string s = nesi_leaf_val_init_helper(tstr,nia,tinfo)->leaf_val.str();  
    vect_string parts = split_space_tab_colon( s );
    try { v->read_from_line_parts( parts, 0 ); }
    catch( boost::bad_lexical_cast & e ) { rt_err( strprintf("can't convert '%s' to %s.", s.c_str(), tstr ) ); }
  }

  // note: always prints/clear pending
  template< typename T >
  bool with_op_left_shift_nesi_dump( tinfo_t const * tinfo, void * o, nesi_dump_buf_t * ndb ) {
    T * v = (T *)o;
    std::ostringstream oss;
    oss << *v;
    ndb->os += oss.str();
    ndb->is_non_leaf = 0;
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
  void * has_def_ctor_make_p( nesi_init_arg_t * nia, tinfo_t const * tinfo, void * o ) {
    shared_ptr< T > * const p = (shared_ptr< T > *)( o );
    p->reset( new T );
    return p->get();
  }
  void leaf_type_nesi_help( tinfo_t const * tinfo, void * o, string * os, string & prefix,
			    bool const show_all, vect_string * help_args, uint32_t help_ix ) {
    assert_st( help_args );
    if( help_ix < help_args->size() ) { // use a help-arg to decend
      if( help_args->at(help_ix) != deep_help_str ) { // the 'deep help' case on leaf types doesn't make sense
	*os += strprintf("%sleaf type '%s' has no fields at all, certainly not '%s', so help cannot be provided for it.\n",
			 prefix.c_str(), tinfo->tname, help_args->at(help_ix).c_str() );
      }
    } else { // leaf case, emit help for this leaf type
      assert_st( help_ix == help_args->size() );
      *os += strprintf( "%sLEAF TYPE INFO: %s\n", prefix.c_str(), (char const *)tinfo->init_arg );
    }
  }

  // string
  void nesi_string_init( nesi_init_arg_t * nia, tinfo_t const * tinfo, void * o )
  {
    assert_st( nia->l ); // no_value_init disabled
    //if( !d ) { return; } // no_value_init --> empty string
    string * v = (string *)o;
    lexp_t * l = nia->l.get();
    if( !l->leaf_val.exists() ) {
      //rt_err( "invalid attempt to use name/value list as string value. list was:" + str(*l) ); // too strong?
      l->deep_inc_use_cnt();
      *v = l->src.str();
    } else {
      ++l->use_cnt;
      *v = l->leaf_val.str();
    }

  }
  make_p_t * string_make_p = &has_def_ctor_make_p< string >;
  vect_push_back_t * string_vect_push_back = &has_def_ctor_vect_push_back_t< string >;
  nesi_dump_t * string_nesi_dump = &with_op_left_shift_nesi_dump< string >;
  void *string_init_arg = (void *)"string";

  // filename_t 
  void nesi_filename_t_init( nesi_init_arg_t * nia, tinfo_t const * tinfo, void * o ) // note: tinfo unused
  {
    filename_t * v = (filename_t *)o;
    nesi_string_init( nia, 0, &v->in ); // note: tinfo unused 
    assert( v->exp.empty() );
    assert_st( nia );
    str_format_from_nvm( v->exp, v->in, *nia );
  }
  make_p_t * filename_t_make_p = &has_def_ctor_make_p< filename_t >;
  vect_push_back_t * filename_t_vect_push_back = &has_def_ctor_vect_push_back_t< filename_t >;
  bool filename_t_nesi_dump( tinfo_t const * tinfo, void * o, nesi_dump_buf_t * ndb ) {
    filename_t * v = (filename_t *)o;
    return string_nesi_dump( tinfo, &v->in, ndb ); // well, it happens the first field, but let's be clear ;)
  }
  void *filename_t_init_arg = (void *)"filename_t";

  string nesi_filename_t_expand( nesi_init_arg_t * nia, string const & fmt ) {
    string ret;
    str_format_from_nvm( ret, fmt, *nia );
    return ret;
  }
    
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

  // uint32_t  
  init_t * nesi_uint32_t_init = &nesi_lexcast_init< uint32_t >;
  make_p_t * uint32_t_make_p = &has_def_ctor_make_p< uint32_t >;
  vect_push_back_t * uint32_t_vect_push_back = &has_def_ctor_vect_push_back_t< uint32_t >;
  nesi_dump_t * uint32_t_nesi_dump = &with_op_left_shift_nesi_dump< uint32_t >;
  void *uint32_t_init_arg = (void *)"uint32_t (32-bit unsigned integer)";

  // int32_t  
  init_t * nesi_int32_t_init = &nesi_lexcast_init< int32_t >;
  make_p_t * int32_t_make_p = &has_def_ctor_make_p< int32_t >;
  vect_push_back_t * int32_t_vect_push_back = &has_def_ctor_vect_push_back_t< int32_t >;
  nesi_dump_t * int32_t_nesi_dump = &with_op_left_shift_nesi_dump< int32_t >;
  void *int32_t_init_arg = (void *)"int32_t (32-bit signed integer)";

  // uint8_t
  init_t * nesi_uint8_t_init = &nesi_lexcast_init< uint8_t >;
  make_p_t * uint8_t_make_p = &has_def_ctor_make_p< uint8_t >;
  vect_push_back_t * uint8_t_vect_push_back = &has_def_ctor_vect_push_back_t< uint8_t >;
  nesi_dump_t * uint8_t_nesi_dump = &with_op_left_shift_nesi_dump< uint8_t >;
  void *uint8_t_init_arg = (void *)"uint8_t (8-bit unsigned integer)";

  // u32_pt_t
  init_t * nesi_u32_pt_t_init = &nesi_str_parts_init< u32_pt_t >;
  make_p_t * u32_pt_t_make_p = &has_def_ctor_make_p< u32_pt_t >;
  vect_push_back_t * u32_pt_t_vect_push_back = &has_def_ctor_vect_push_back_t< u32_pt_t >;
  nesi_dump_t * u32_pt_t_nesi_dump = &with_op_left_shift_nesi_dump< u32_pt_t >;
  void *u32_pt_t_init_arg = (void *)"u32_pt_t (pair of 32-bit unsigned integers, i.e. an unsigned point)";

  // u32_box_t
  init_t * nesi_u32_box_t_init = &nesi_str_parts_init< u32_box_t >;
  make_p_t * u32_box_t_make_p = &has_def_ctor_make_p< u32_box_t >;
  vect_push_back_t * u32_box_t_vect_push_back = &has_def_ctor_vect_push_back_t< u32_box_t >;
  nesi_dump_t * u32_box_t_nesi_dump = &with_op_left_shift_nesi_dump< u32_box_t >;
  void *u32_box_t_init_arg = (void *)"u32_box_t (pair of u32_pt_t, i.e. an unsigned box)";

  extern tinfo_t tinfo_uint32_t;
  extern tinfo_t tinfo_string;
  void nesi_init_checked( void * ret, tinfo_t const * const & tinfo,
                          p_lexp_t const & lexp, nesi_init_arg_t * const & nia, string const & err_prefix ) {
    lexp_name_val_map_t nvm( lexp, nia );
    try { tinfo->init( &nvm, tinfo, ret ); }
    catch( rt_exception & rte ) { rte.err_msg = err_prefix + rte.err_msg; throw; }
  }

  // dims_t
  void nesi_dims_t_init( nesi_init_arg_t * nia, tinfo_t const * tinfo, void * o ) {
    dims_t * v = (dims_t *)o;
    if( !nia->l ) { return; } // no_value_init --> empty vector
    lexp_t * l = nia->l.get();
    if( l->leaf_val.exists() ) {
      char const * const tstr = (char const *)tinfo->init_arg;
      rt_err( "invalid attempt to use string as name/value list for "+string(tstr)+" init. string was:" + str(*l) );
    }
    ++l->use_cnt;
    string tn = "float"; // FIXME_TNDA: user-input dims_t's have a hard-coded default type of float for now ...
    for( vect_lexp_nv_t::iterator i = l->kids.begin(); i != l->kids.end(); ++i ) {
      if( i->n.str() == "__tn__" ) { nesi_init_checked( &tn, &tinfo_string, i->v, nia, "dims_t __tn__: " ); }
      else {
        uint32_t dim_v = 0;
        nesi_init_checked( &dim_v, &tinfo_uint32_t, i->v, nia, "dims_t dim " + i->n.str() + ": " );
        v->add_dims( i->n.str(), dim_v );
      }
    }
    v->tn = tn;
    if( !v->empty() ) { v->calc_strides(); } // if not null/empty, calc strides. FIXME: strides not really handled here, obv.
  }
  make_p_t * dims_t_make_p = &has_def_ctor_make_p< dims_t >;
  vect_push_back_t * dims_t_vect_push_back = &has_def_ctor_vect_push_back_t< dims_t >;
  nesi_dump_t * dims_t_nesi_dump = &with_op_left_shift_nesi_dump< dims_t >;
  void *dims_t_init_arg = (void *)"dims_t (N-D Array Dimentions)";

  // FIXME: broken for uint8_t, as per the usual (bad?) reasons. need a wrapper to treat uint8_t as numeric ...
  struct nda_elems_init_t {
    vect_string const & parts;
    nda_elems_init_t( vect_string const & parts_ ) : parts(parts_) {}
    template< typename T > void op( nda_t & nda ) const { 
      assert_st( parts.size() == nda.elems_sz() );
      T * const elems = static_cast<T *>(nda.rp_elems());
      for( uint32_t i = 0; i != nda.elems_sz(); ++i ) {
        try { elems[i] = boost::lexical_cast< T >( parts[i] ); }
        catch( boost::bad_lexical_cast & e ) { rt_err( strprintf("can't convert '%s' to %s.", 
                                                                 parts[i].c_str(), nda.dims.get_tn().c_str() ) ); }
      }
    }
  };

  // nda_t's are dims_t's with values. however, it's not clear if we can or should make nda_t a 'real' NESI type. so,
  // like dims_t, we manually parse nda_t's with the following code. we treat nda_t's as having three fields: tn, dims,
  // and v. dims is parsed as a dims_t using the above function. tn is like the '__tn__' pseudo-dim used by the dims_t
  // parsing code, and will override the tn of any parsed dims_t (FIXME: we should check that __tn__ was not used inside
  // the dims_t in that case). v is a space-or-colon seperated list of values. wrinkles. for now, we require the v
  // field, but if we wish to allow specifciation of dims-only nda_t's, we could make it optional. if the v field is
  // specified, the dims field is optional. if the dims field is ommited, the tn field is required; otherwise it is
  // optional. when ommited, the dims will be set to a dims_t with a single dim named 'v' of length == to the number of
  // elems given in the 'v' field.
  extern tinfo_t tinfo_dims_t;
  void nesi_nda_t_init( nesi_init_arg_t * nia, tinfo_t const * tinfo, void * o ) {
    nda_t * v = (nda_t *)o;
    assert_st( v->dims.empty() ); // should be freshly constructed
    if( !nia->l ) { return; } // no_value_init --> empty vector
    lexp_t * l = nia->l.get();
    if( l->leaf_val.exists() ) {
      char const * const tstr = (char const *)tinfo->init_arg;
      rt_err( "invalid attempt to use string as name/value list for "+string(tstr)+" init. string was:" + str(*l) );
    }
    ++l->use_cnt;
    p_lexp_t tn,dims,vv;
    for( vect_lexp_nv_t::iterator i = l->kids.begin(); i != l->kids.end(); ++i ) { // gather fields, error check for extras
      if( 0 ) {}
      else if( i->n.str() == "tn" ) { tn = i->v; }
      else if( i->n.str() == "dims" ) { dims = i->v; }
      else if( i->n.str() == "v" ) { vv = i->v; }
      else { rt_err( "nda_t has no field named "+i->n.str()+". valid fields are 'tn' (string; type name), 'dims' (dims_t), and 'v' (string; space-or-colon-seperated list of values" ); }
    }
    if( (!dims) && (!tn) ) { rt_err( "nda_t has neither 'tn' nor 'dims' field, can't determine type; set at least one." );}
    string tn_str;
    if( dims ) {
      lexp_name_val_map_t nvm( dims, nia );
      try { nesi_dims_t_init( &nvm, &tinfo_dims_t, &v->dims ); }
      catch( rt_exception & rte ) { rte.err_msg = "nda_t dims: " + rte.err_msg; throw; }
      if( v->dims.empty() ) { rt_err( "nda_t 'dims' field specified empty dims." ); }
    } // set v->dims
    if( tn ) { nesi_init_checked( &tn_str, &tinfo_string, tn, nia, "nda_t tn: " ); } // note: lexp->str can't fail ...
    else { assert_st( dims ); tn_str = v->dims.tn; }
    assert_st( !tn_str.empty() ); // FIXME: user could set null tn; better error check here?
    // pase values
    //if( !vv ) { rt_err( "nda_t has no 'v' field; currently, only non-empty nda_t's may be specified." ); }
    if( vv ) {
      string v_str;
      nesi_init_checked( &v_str, &tinfo_string, vv, nia, "nda_t v: " );
      vect_string parts = split_space_tab_colon( v_str );
      // init dims if needed; otherwise override dims.tn if tn was specificed (note: could set unconditionally to tn_str)
      if( !dims ) { v->dims = dims_t( vect_uint32_t{uint32_t(parts.size())}, vect_string{"v"}, tn_str ); }
      else if( tn ) { v->dims.tn = tn_str; }  // note: if !tn, tn_str came from v->dims in the first place
      // check length agreement
      if( v->dims.dims_prod() != parts.size() ) {
        assert_st( dims ); // disagreement not possible otherwise
        rt_err( strprintf( "nda_t 'v' field has %s elements/parts, but user-specified 'dims' say there should be %s.",
                           str(parts.size()).c_str(), str(v->dims.dims_prod()).c_str() ) );
      }
      v->nesi_init_private_alloc(); // just calls alloc(), but has a scary/descriptive name
      nda_dispatch( *v, nda_elems_init_t( parts ) ); // actually fill in values
    } else {
      // note: data pointer left null
      
      // instead of an error here, we could maybe default the dims to scalar (i.e. one 'v' dim of sz 1). but do we ever really want that case?
      if( !dims ) { rt_err( "nda_t has neither 'v' nor 'dims' field, can't determine size; set at least one." );}
      else if( tn ) { v->dims.tn = tn_str; }  // note: if !tn, tn_str came from v->dims in the first place
    }
  }
  make_p_t * nda_t_make_p = &has_def_ctor_make_p< nda_t >;
  vect_push_back_t * nda_t_vect_push_back = &has_def_ctor_vect_push_back_t< nda_t >;
  nesi_dump_t * nda_t_nesi_dump = &with_op_left_shift_nesi_dump< nda_t >;
  void *nda_t_init_arg = (void *)"nda_t (N-D Array)";

  // FIXME: we've used templates here to reduce code duplication for the various map_str_T type. however, we still don't
  // have 'full' NESI support for generic map_str_T NESI types, and this causes a few issues -- not the least of which
  // that we need to declare every map type here. we make some attempt to move in the right direction by having the
  // tinfo->init_arg for each of these types point to the value's tinfo, and use it to get the error string for the type
  // as well as to create the values with make_p. however, lacking a 'generic' must_insert(), we need the type T to
  // declare a fully-correctly-typed pointer to the map, as well as a fully-correctly-typed shared pointer.

  // map_str_T
  template< typename T > void nesi_map_str_T_init( nesi_init_arg_t * nia, tinfo_t const * tinfo, void * o ) {
    map< string, T > * v = (map< string, T > *)o;
    if( !nia->l ) { return; } // no_value_init --> empty vector
    lexp_t * l = nia->l.get();

    tinfo_t * const vt = (tinfo_t *)( tinfo->init_arg );

    if( l->leaf_val.exists() ) {
      char const * const tstr = (char const *)vt->init_arg;
      rt_err( "invalid attempt to use string as name/value list for map str -> "+string(tstr)+" init. string was:" + str(*l) );
    }
    ++l->use_cnt;
    for( vect_lexp_nv_t::iterator i = l->kids.begin(); i != l->kids.end(); ++i ) {
      shared_ptr< T > p_vv;
      void * vv = vt->make_p( nia, vt, &p_vv );
      // note: for vector initialization, i->n (the name of the name/value pair) is ignored.
      lexp_name_val_map_t nvm( i->v, nia );
      try { 
	vt->init( &nvm, vt, vv );
      }
      catch( rt_exception & rte ) {
	rte.err_msg = "map elem with key " + i->n.str() + ": " + rte.err_msg;
	throw;
      }
      must_insert( *v, i->n.str(), *p_vv );
    }
  }

  // map_str_p_nda_t
  init_t * nesi_map_str_p_nda_t_init = &nesi_map_str_T_init< p_nda_t >;
  make_p_t * map_str_p_nda_t_make_p = &has_def_ctor_make_p< map_str_p_nda_t >;
  vect_push_back_t * map_str_p_nda_t_vect_push_back = &has_def_ctor_vect_push_back_t< map_str_p_nda_t >;
  nesi_dump_t * map_str_p_nda_t_nesi_dump = &with_op_left_shift_nesi_dump< map_str_p_nda_t >;
  extern tinfo_t tinfo_p_nda_t;
  void *map_str_p_nda_t_init_arg = (void *)&tinfo_p_nda_t;

  // map_str_op_tune_t
  init_t * nesi_map_str_op_tune_t_init = &nesi_map_str_T_init< op_tune_t >;
  make_p_t * map_str_op_tune_t_make_p = &has_def_ctor_make_p< map_str_op_tune_t >;
  vect_push_back_t * map_str_op_tune_t_vect_push_back = &has_def_ctor_vect_push_back_t< map_str_op_tune_t >;
  nesi_dump_t * map_str_op_tune_t_nesi_dump = &with_op_left_shift_nesi_dump< map_str_op_tune_t >;
  extern tinfo_t tinfo_op_tune_t;
  void *map_str_op_tune_t_init_arg = (void *)&tinfo_op_tune_t;

  // map_str_dims_t
  init_t * nesi_map_str_dims_t_init = &nesi_map_str_T_init< dims_t >;
  make_p_t * map_str_dims_t_make_p = &has_def_ctor_make_p< map_str_dims_t >;
  vect_push_back_t * map_str_dims_t_vect_push_back = &has_def_ctor_vect_push_back_t< map_str_dims_t >;
  nesi_dump_t * map_str_dims_t_nesi_dump = &with_op_left_shift_nesi_dump< map_str_dims_t >;
  extern tinfo_t tinfo_dims_t;
  void *map_str_dims_t_init_arg = (void *)&tinfo_dims_t;

  // map_str_uint32_t
  init_t * nesi_map_str_uint32_t_init = &nesi_map_str_T_init< uint32_t >;
  make_p_t * map_str_uint32_t_make_p = &has_def_ctor_make_p< map_str_uint32_t >;
  vect_push_back_t * map_str_uint32_t_vect_push_back = &has_def_ctor_vect_push_back_t< map_str_uint32_t >;
  nesi_dump_t * map_str_uint32_t_nesi_dump = &with_op_left_shift_nesi_dump< map_str_uint32_t >;
  extern tinfo_t tinfo_uint32_t;
  void *map_str_uint32_t_init_arg = (void *)&tinfo_uint32_t;

  // map_str_double_t
  init_t * nesi_map_str_double_init = &nesi_map_str_T_init< double >;
  make_p_t * map_str_double_make_p = &has_def_ctor_make_p< map_str_double >;
  vect_push_back_t * map_str_double_vect_push_back = &has_def_ctor_vect_push_back_t< map_str_double >;
  nesi_dump_t * map_str_double_nesi_dump = &with_op_left_shift_nesi_dump< map_str_double >;
  extern tinfo_t tinfo_double;
  void *map_str_double_init_arg = (void *)&tinfo_double; // "map_str_double (key-value map from string to double)";

  init_t * nesi_map_str_str_init = &nesi_map_str_T_init< string >;
  make_p_t * map_str_str_make_p = &has_def_ctor_make_p< map_str_str >;
  vect_push_back_t * map_str_str_vect_push_back = &has_def_ctor_vect_push_back_t< map_str_str >;
  nesi_dump_t * map_str_str_nesi_dump = &with_op_left_shift_nesi_dump< map_str_str >;
  extern tinfo_t tinfo_string;
  void *map_str_str_init_arg = (void *)&tinfo_string;


// note: a special case in nesi_gen.py makes 'base' NESI types be associated with nesi.cc (and
// thus go into the nesi.cc.nesi_gen.cc file)
#include"gen/nesi.cc.nesi_gen.cc"

}
