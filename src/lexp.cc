// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"test_base.H"
#include"lexp.H"
#include"str_util.H"
#include"xml_util.H"
#include"has_main.H"

#include<boost/filesystem.hpp>

namespace boda {
  using std::string;
  using namespace pugi;
  
  using boost::filesystem::path;

  string const xml_inc_prefix = "XML_INC_";

  string apply_rel_path_of( string const & ref_fn, string const & targ_fn ) {
    return ( path( ref_fn ).parent_path() / targ_fn ).string();
  }

  // error format strings
  string const le_unparsed_data( "unparsed data remaining (expected end of input)" );
  string const le_end_after_list_item( "unexpected end of input (expecting ',' to continue list or ')' to end list)" );
  string const le_end_missing_cp( "unexpected end of input (expecting more close parens)" );
  string const le_end_in_escape( "unexpected end of input after escape char '\\' (expected char)" );
  string const le_end_in_name( "unexpected end of input (expecting '=' to end name)" );
  string const le_bad_name_char( "invalid name character in name" );
  string const le_empty_name( "invalid empty name (no chars before '=' in name)" );
  string const le_bad_char_after_list_item( "expected ',' or ')' after list item" );

  bool sstr_t::operator < ( sstr_t const & o ) const { 
    if( sz() != o.sz() ) { return sz() < o.sz(); }
    return memcmp( base.get()+b, o.base.get()+o.b, sz() ) < 0; // no embedded nulls, so memcmp is okay to use
  }

  void sstr_t::set_from_string( string const &s ) {
    b = 0; e = s.size(); // set b/e
    base = p_uint8_t( (uint8_t *)malloc( s.size() ), free ); // allocate space
    memcpy( base.get(), &s[0], sz() ); // copy string data
  }
  void sstr_t::set_from_cstr( char const * s ) {
    b = 0; e = strlen(s); // set b/e
    base = p_uint8_t( (uint8_t *)malloc( e ), free ); // allocate space
    memcpy( base.get(), s, sz() ); // copy string data
  }
  void no_free( void * ) {}
  void sstr_t::borrow_from_string( string const &s ) {
    b = 0; e = s.size(); // set b/e
    base = p_uint8_t( (uint8_t *)&s[0], no_free ); // borrow data
  }
  void sstr_t::borrow_from_cstr( char const * s ) {
    b = 0; e = strlen(s); // set b/e
    base = p_uint8_t( (uint8_t *)s, no_free ); // borrow data
  }

  p_lexp_t parse_lexp_leaf_str( char const * const s )
  {
    p_lexp_t ret( new lexp_t( sstr_t() ) );
    ret->leaf_val.set_from_cstr( s );
    return ret;
  }

  // for testing/debugging (particularly regressing testing), create
  // some simple statistics of a lexp.
  struct lexp_stats_t
  {
    uint32_t num_leaf;
    uint32_t num_list;
    uint32_t num_kids;
    void set_to_zeros( void ) { num_leaf=0; num_list=0; num_kids=0; } // no ctor so we can use {} init; --std=c++-sigh 
    bool is_eq( lexp_stats_t const & o ) { return num_leaf==o.num_leaf && num_list==o.num_list && num_kids==o.num_kids; }
  };
  std::ostream & operator << ( std::ostream & os, lexp_stats_t const & v)  {
    return os << strprintf( "num_leaf=%s num_list=%s num_kids=%s", str(v.num_leaf).c_str(), str(v.num_list).c_str(), str(v.num_kids).c_str() );
  }

  void lexp_t::get_stats( lexp_stats_t & lex_stats ) {
    if( leaf_val.exists() ) { ++lex_stats.num_leaf; assert_st( kids.empty() ); }
    else { 
      ++lex_stats.num_list;
      for( vect_lexp_nv_t::iterator i = kids.begin(); i != kids.end(); ++i ) {
	++lex_stats.num_kids;
	i->v->get_stats( lex_stats );
      }
    }
  }

#if 0 // unused
  void lexp_t::deep_inc_use_cnt( void ) {
    ++use_cnt;
    for( vect_lexp_nv_t::iterator i = kids.begin(); i != kids.end(); ++i ) {
      i->v->deep_inc_use_cnt();
    }
  }
  void lexp_t::add_all_kids_from( p_lexp_t lexp ) {
    kids.insert( kids.end(), lexp->kids.begin(), lexp->kids.end() );
  }
  void lexp_t::expand_include_xml( void ) { 
    // TODO: add support if desired
  }
#endif

  void lexp_check_unused( lexp_t * l, vect_string & path )
  {
    if( !l->use_cnt ) { rt_err( strprintf( "unused input: %s:%s", join(path,"->").c_str(), str(*l).c_str() ) ); } 
    for( vect_lexp_nv_t::iterator i = l->kids.begin(); i != l->kids.end(); ++i ) {
      path.push_back( i->n.str() );
      lexp_check_unused( i->v.get(), path );
      path.pop_back();
    }
  }

  void lexp_name_val_map_t::dump( void ) {
    printf("nvm contents (nvm_init=%s):\n",str(nvm_init).c_str());
    for( map<sstr_t,p_lexp_t>::const_iterator i = nvm.begin(); i != nvm.end(); ++i ) {
      printf( "%s=%s\n", str(i->first).c_str(), str((*i->second)).c_str() );
    }
    if( parent ) { printf("parent:\n"); parent->dump(); }
  }

  bool lexp_name_val_map_t::insert_leaf( char const * n, char const * v, bool const inc_use_cnt ) {
    init_nvm();
    sstr_t ss_n;
    ss_n.set_from_cstr( n );
    bool const did_ins = nvm.insert( std::make_pair( ss_n, parse_lexp_leaf_str( v ) ) ).second;
    if( did_ins ) { // put new value in l as well
      lexp_nv_t kid;
      kid.n.set_from_cstr( n );
      kid.v = parse_lexp_leaf_str( v );
      if( inc_use_cnt ) { ++kid.v->use_cnt; }
      l->kids.push_back( kid );
    }
    return did_ins;
  }

  void lexp_name_val_map_t::init_nvm( void ) {
    assert_st( l );
    if( l->leaf_val.exists() ) { // should not be a leaf
      rt_err( "invalid attempt to use string as name/value list. string was:" + str(*l) );
    }
    if( nvm_init ) { return; }
    for( vect_lexp_nv_t::iterator i = l->kids.begin(); i != l->kids.end(); ++i ) {
      bool const did_ins = nvm.insert( std::make_pair( i->n, i->v ) ).second;
      if( !did_ins ) { rt_err( "invalid duplicate name '"+i->n.str()+"' in name/value list" ); }
    }
    nvm_init = 1;
  }

  p_lexp_t lexp_name_val_map_t::find( char const * n, lexp_name_val_map_t * * const found_scope ) {
    assert_st( found_scope ); // who doesn't want to know the found scope? nobody. could be optional, though.
    if( nvm_init ) { // skip uninited nvms (from vectors, leafs, etc)
      sstr_t ss_vname;
      ss_vname.borrow_from_string( n );
      std::map< sstr_t, p_lexp_t >::const_iterator nvmi = nvm.find( ss_vname );
      if( nvmi != nvm.end() ) {	*found_scope = this; return nvmi->second; } // if found at this scope, return value
    }
    if( parent ) { return parent->find( n, found_scope ); } // try parent scope if it exists
    return p_lexp_t(); // ... otherwise return not found (and don't set found_scope).
  }
  // for testing/debugging, re-create the exact 'basic' format input
  // string from the tree of name/value lists. note: we sort-of assume
  // the tree was built from a single string here (and thus that .src
  // is always valid and in 'basic' format), but generilizations are
  // possilbe: src_has_trailing_comma would need to be a variable, not
  // a function (set by the creator properly), and leaves would either
  // need to provide a 'basic raw' version of thier value (i.e. by
  // escaping needed chars from thier cooked value), or the following
  // would need to create a default 'raw' version itself (again by
  // escaping as needed). note that there isn't really a cannonical
  // way to escape a value, but probably only escaping as needed would
  // be best.
  std::ostream & operator <<(std::ostream & os, sstr_t const & v) {
    return os.write( (char const *)v.base.get()+v.b, v.sz() );
  }
  std::ostream & operator <<(std::ostream & os, lexp_nv_t const & v) {
    return os << v.n << "=" << (*v.v);
  }
  // if we want to re-create the input string exactly
  // *without* just printing src, and this node is a non-empty list,
  // we need to know if there was an optional trailing
  // comma. wonderful!
  bool lexp_t::src_has_trailing_comma( void ) const
  {
    if( !src.exists() ) { return 0; } // if created from, say, XML. assume no training comma
    assert( src.sz() > 1 );
    if( !kids.size() ) { return 0; }
    assert( src.sz() > 2 );
    assert( src.base.get()[src.e-1] == ')' );
    return src.base.get()[src.e-2] == ',';
  }

  // because we are pretty lenient in requiring escapes, it's a little
  // tricky to 'minimally' escape leaf values. there certinaly isn't a
  // cannonical escaped form, but this will 'often' yield similar
  // escapes to what the user might have originally used. in
  // particular, when no escapes are needed, none will be used.
  string lexp_escape( string const & s )
  {
    string ret;
    uint32_t paren_depth = 0;
    for (string::const_iterator i = s.begin(); i != s.end(); ++i) {
      switch (*i)
      {
      case '(': ++paren_depth; break;
      case ')': if(!paren_depth) { ret.push_back('\\'); } else { --paren_depth; } break;
      case ',': if(!paren_depth) { ret.push_back('\\'); } break;
      case '\\': ret.push_back('\\'); break;
      }
      ret.push_back( *i );
    }
    return ret;
  }

  std::ostream & operator <<(std::ostream & os, lexp_t const & v) {
    // leaf case. note: we print the original 'raw' value here if availible
    if( v.leaf_val.exists() ) { 
      if( v.src.exists() ) { return os << v.src; } else { return os << lexp_escape( v.leaf_val.str() ); } 
    } else { // otherwise, list case
      os << "(";
      for (vect_lexp_nv_t::const_iterator i = v.kids.begin(); i != v.kids.end(); ++i) 
      { 
	if( i != v.kids.begin() ) { os << ','; }
	os << (*i);
      }
      return os << (v.src_has_trailing_comma() ? ",)" : ")" );
    }
  }

  struct lex_parse_t
  {
    sstr_t const s;
    uint32_t paren_depth;
    lex_parse_t( sstr_t const & s_ ) : s(s_), paren_depth(0), m_cur_off(0) { set_cur_c(); }
    // parse iface (uses token/char level functions internally)
    p_lexp_t parse_lexp( void );
    void err_if_data_left( void );
  protected:
    // internal parsing level function
    sstr_t parse_name_eq( void );
    void parse_leaf( p_lexp_t ret );
    void parse_list( p_lexp_t ret );

    string err_str( string const & msg, uint32_t const off );
    void err( string const & msg );
    // token/char iface
    uint32_t cur_off( void ) const { return m_cur_off; }
    uint32_t cur_c( void ) const { return m_cur_c; }
    void next_c( void ) { assert_st( !at_end() ); ++m_cur_off; set_cur_c(); }

    // lexer state; used only internally by token/char level access functions
    uint32_t m_cur_off; 
    char m_cur_c; 
    bool at_end( void ) const { return cur_off() == s.end_off(); }
    void set_cur_c( void ) { m_cur_c = ( at_end() ? 0 : s.base.get()[cur_off()] ); }

    friend struct lexp_test_run_t;
  };

  void lex_parse_t::err( string const & msg ) { rt_err( err_str( msg, cur_off() ) ); }
  string lex_parse_t::err_str( string const & msg, uint32_t const off )
  {
    uint32_t const max_ccs = 35; // max context chars
    uint32_t const cb = ( off >= (s.b + max_ccs) ) ? (off - max_ccs) : s.b;
    uint32_t const ce = ( (off + max_ccs) <= s.e ) ? (off + max_ccs) : s.e;
    uint32_t const eo = (off - cb);
    
    return strprintf( "at offset %s=%s: %s in context:\n%s%s%s\n%s^",
		       str(off).c_str(),
		       cur_c() ? strprintf("'%c'",cur_c()).c_str() : "END", // offending char or END for end of input
		       msg.c_str(), // error message
		       ( cb == s.b ) ? "'" : "...'", // mark if start with truncated with ...
		       string( s.base.get()+cb, s.base.get()+ce ).c_str(), // context around error
		       ( ce == s.e ) ? "'" : "'...", // mark if end was truncated with ...
		       string(eo+( ( cb == s.b ) ? 1 : 4 ),' ').c_str() // spaces to offset '^' to error char
		      );
  }

  void lex_parse_t::err_if_data_left( void ) {
    if( cur_c() != 0 ) { err( le_unparsed_data ); }
  }

  p_lexp_t lex_parse_t::parse_lexp( void )
  {
    p_lexp_t ret( new lexp_t( s ) );
    ret->src.b = cur_off(); // note: final ret->s.e will be <= current val (== s.e), but is currently unknown
    if( cur_c() == '(' ) { parse_list( ret ); }
    else { parse_leaf( ret ); }
    ret->src.shrink_end_off( cur_off() );
    // for bad input, this assert is too strong, and downstream errors will catch whatever the issue is
    //assert_st( (cur_c() == 0) || (cur_c() == ',') || (cur_c() == ')') ); // end-of-lexp postcondition
    return ret;
  };

  // when the first char of a lexp_t is '(', we parse it as a list
  // notes: empty lists are allowed (i.e. first char is ')'). trailing
  // commas are allowed (i.e. a list can end with ",)")
  void lex_parse_t::parse_list( p_lexp_t ret )
  {
    assert_st( cur_c() == '(' ); // start of list precondition
    next_c(); // consume '('
    while( cur_c() != ')' ) {
      lexp_nv_t kid;
      kid.n = parse_name_eq();      
      kid.v = parse_lexp();
      if( startswith( kid.n.str(), xml_inc_prefix ) ) {
	if( !kid.v->leaf_val.exists() ) { // FIXME: better error?
	  rt_err( "invalid attempt to use name/value list as filename for XML include. list was:" + str(*kid.v) );
	}
	kid.n.set_from_string( string( kid.n.str(), xml_inc_prefix.size(), string::npos ) );
	kid.v = parse_lexp_xml_file( kid.v->leaf_val.str() );
      }
      ret->kids.push_back( kid );
      if( cur_c() == 0 ) { err( le_end_after_list_item ); }
      else if( cur_c() == ',' ) { next_c(); }
      else if( cur_c() == ')' ) { } // note: will exit loop now.
      else { // error, end-of-lexp postcondition didn't hold (but was not checked there so we can emit the error here)
	err( le_bad_char_after_list_item ); // note: this can (only?) happen if the kid.v lexp was a list. 
      } 
    }
    assert_st( cur_c() == ')' ); // end of list postcondition
    next_c(); // consume ')'
  }

  // when the first char of lexp_t is anything other than '(', we
  // parse a leaf string value.  it ends with the first ',' or ')' we
  // see at the paren_depth==0. for convenience, we allow and ignore
  // all '=', matched '(' and ')', as well as and ',' that are inside
  // parens. values that cannot follow these requirements must use
  // escapes for at least the relevant special chars
  void lex_parse_t::parse_leaf( p_lexp_t ret )
  {
    bool had_escape = 0; // if no escapes, ret->leaf_val can share data with ret->src, ...
    string cooked; // ... bue if there are escapes, we need to make a local copy for the leaf value.
    uint32_t paren_depth = 0;
    while( 1 ) {
      if( cur_c() == 0 ) { // end of input always ends scope
	if( paren_depth ) { err( le_end_missing_cp ); }
	break; // end scope
      }
      else if( cur_c() == ')' ) { // sometimes ends scope
	if( !paren_depth ) { break; } // unmatched ')' ends leaf, ')' is unconsumed
	--paren_depth;
      }
      else if( cur_c() == '(' ) { 
	assert_st( cur_off() != ret->src.b ); // should not be at first char or we missed a list
	++paren_depth; 
      }
      else if( cur_c() == '\\' ) { // escape char
	if( !had_escape ) { // first escape. copy part of leaf consumed so far to cooked.
	  had_escape = 1; 
	  cooked = string( ret->src.base.get()+ret->src.b, ret->src.base.get()+cur_off() ); 
	} 
	next_c(); // consume '\\'
	if( cur_c() == 0 ) { err( le_end_in_escape ); }
      } 
      else if( cur_c() == ',' ) { if( !paren_depth ) { break; } } // un-()ed ',' ends leaf
      if( had_escape ) { cooked.push_back( cur_c() ); }
      next_c(); // (possibly escaped/ignored special) char is part of leaf value, consume it
    }
    if( had_escape ) { ret->leaf_val.set_from_string( cooked ); }
    else { ret->leaf_val = ret->src; ret->leaf_val.shrink_end_off( cur_off() ); }
  }

  // names must be non-empty.  we do not allow eof, '=', '(', ')',
  // ',', or '\\' inside names (keys). note that since '\\' is
  // forbidden, escapes are not possible. '=' must end the key, thus
  // eof before a '=' is also an error. generally, names must be valid
  // C identifiers anyway, so these restrictions should be good/okay.
  sstr_t lex_parse_t::parse_name_eq( void )
  {
    sstr_t ret( s );
    ret.b = cur_off();
    while( 1 ) {
      if( (cur_c() == 0) ) { err( le_end_in_name ); }
      else if( (cur_c() == '(') || (cur_c() == ')' ) || (cur_c() == '\\' ) || (cur_c() == ',' ) ) {
	err( le_bad_name_char );
      }
      else if( cur_c() == '=' ) { // '=' always ends a name
	if( ret.b == cur_off() ) { err( le_empty_name ); }
	ret.shrink_end_off( cur_off() );
	next_c(); // consume '='
	return ret;
      } 
      else { next_c(); } // char is part of name
    }
  }

  p_lexp_t parse_lexp( string const & s )
  {
    sstr_t sstr;
    sstr.set_from_string( s );
    lex_parse_t lex_parse( sstr );
    p_lexp_t ret = lex_parse.parse_lexp();
    lex_parse.err_if_data_left();
    return ret;
  }

  // note: there is plenty of room to optimization memory/speed here
  // if we share data with the pugixml nodes. this should be possible
  // using sstr_t::borrow_from_pchar(). this function is TODO, but
  // similar to borrow from string, but, we need some trickery to keep
  // the xml doc in memory ... a custom deleter that holds a
  // shared_ptr to the xml doc should do it.
  p_lexp_t parse_lexp_list_xml( xml_node const & xn, string const & xml_fn )
  {
    p_lexp_t ret( new lexp_t( sstr_t() ) );
    lexp_nv_t kid;
    for( xml_attribute attr: xn.attributes() ) { // attributes become leaf values
      if( startswith( attr.name(), xml_inc_prefix ) ) {
	kid.n.set_from_string( string( attr.name(), xml_inc_prefix.size(), string::npos ) );
	string const xml_inc_fn( apply_rel_path_of( xml_fn, attr.value() ) );
	kid.v = parse_lexp_xml_file( xml_inc_fn );
      } else {
	kid.n.set_from_string( attr.name() );
	kid.v = parse_lexp_leaf_str( attr.value() );
      }
      ret->kids.push_back( kid );
    }
    for( xml_node xn_i: xn.children() ) { // child elements become list values
      kid.n.set_from_string( xn_i.name() );
      kid.v = parse_lexp_list_xml( xn_i, xml_fn );
      ret->kids.push_back( kid );
    }
    return ret;
  }

  p_lexp_t make_list_lexp_from_one_key_val( std::string const & k, std::string const & v ) {
    p_lexp_t ret( new lexp_t( sstr_t() ) );
    ret->add_key_val( k, v );
    return ret;
  }

  void lexp_t::add_key_val( std::string const & k, std::string const & v ) {
    lexp_nv_t kid;
    if( startswith( k, xml_inc_prefix ) ) {
      kid.n.set_from_string( string( k, xml_inc_prefix.size(), string::npos ) );
      kid.v = parse_lexp_xml_file( v );
    } else {
      kid.n.set_from_string( k );
      try { kid.v = parse_lexp( v ); }
      catch( rt_exception & rte ) {
	rte.err_msg = "parsing value '" + k + "': " + rte.err_msg;
	throw;
      }
    }
    kids.push_back( kid );
  }

  p_lexp_t parse_lexp_xml_file( string const & fn )
  {
    vect_string fn_parts = split(fn,':');
    assert_st( !fn_parts.empty() );
    string const & xml_fn = fn_parts[0];
    xml_document doc;
    xml_node xn = xml_file_get_root( doc, xml_fn );
    for (vect_string::const_iterator i = fn_parts.begin() + 1; i != fn_parts.end(); ++i) { // decend path if given
      xn = xml_must_decend( xml_fn.c_str(), xn, (*i).c_str() );
    }
    p_lexp_t ret = parse_lexp_list_xml( xn, xml_fn );
    //printf( "*ret=%s\n", str(*ret).c_str() );
    return ret;
  }

  void str_format_from_nvm( string & out, string const & fmt, lexp_name_val_map_t & nvm )
  {
    // expand refs. note: refs do not inc the use count of the lexp they use (as this seems sensible).
    for( string::const_iterator i = fmt.begin(); i != fmt.end(); ++i ) {
      if( *i == '%' ) {
	++i; if( i == fmt.end() ) { rt_err( "end of string after '%' in filename, expected '(' or '%'." ); }
	if( *i == '%' ) { out.push_back( *i ); } // escaped '%'
	else if( *i != '(' ) { rt_err( "'" + string(1,*i) + "' after '%' in filename, expected '(' or '%'." ); }
	else { // saw '(', process ref
	  string ref;
	  while( 1 ) {
	    ++i;
	    if( i == fmt.end() ) { rt_err( "end of string after '%(' in filename, expected ')' to terminate ref" ); }
	    if( *i == ')' ) { break; }
	    ref.push_back( *i );
	  }
	  // expand ref recursively
	  nesi_init_arg_t * found_scope = 0;
	  p_lexp_t di = nvm.find( ref.c_str(), &found_scope );
	  if( !di ) { rt_err( "unable to expand ref '" + ref + "' in filename, ref not found" );  }
	  if( !di->leaf_val.exists() ) { 
	    rt_err( "invalid attempt to use name/value list as filename ref '" + ref + "' value. list was:" + str(*di) );
	  }
	  ++di->use_cnt;
	  string const nest_fmt = di->leaf_val.str();
	  lexp_name_val_map_t nest_nvm( di, found_scope ); // note lexical (non-dynamic) scoping here
	  str_format_from_nvm( out, nest_fmt, nest_nvm );
	}
      } else { out.push_back( *i ); } // not a '%'
    }
  }

  void str_format_from_nvm_str( string & out, string const & fmt, string const & nvm_str ) {
    lexp_name_val_map_t nvm( parse_lexp( nvm_str ) );
    nvm.init_nvm();
    str_format_from_nvm( out, fmt, nvm );
  }

  struct lexp_test_t
  {
    string name;
    string desc;
    string in;
    string const * err_fmt;
    uint32_t const err_off;
    lexp_stats_t lexp_stats;
  };

  string const si = "(foo=baz,biz=boo,bing=(f1=21,faz=na),hap=(baz=234,fin=12))";
  lexp_test_t lexp_tests[] = {
    { "junk_end", "", si + ")sdf", &le_unparsed_data, 58 },
    { "ext_cp_end", "extra close paren (looks like junk at end)", "baz)foo,bar,bing)", &le_unparsed_data, 3 },
    { "end_li_bad_char", "bad char (not ',' or ')' after list item", "(foo=()d)", &le_bad_char_after_list_item, 7 },
    { "end_li_val", "end after list item (with val after=)", "(foo=bar,baz=(foo=da", &le_end_after_list_item, 20 },
    { "end_li", "end after list item (no val after=)", "(foo=bar,baz=(foo=", &le_end_after_list_item, 18 },
    { "end_no_cp", "end missing cp", "foo(", &le_end_missing_cp, 4 },
    { "end_no_cp_nest", " missing cp (nested)", "bar=foo(", &le_end_missing_cp, 8 },
    { "end_in_escape_1", "", "\\", &le_end_in_escape, 1 },
    { "end_in_escape_2", "", "(foo=fv\\", &le_end_in_escape, 8 },
    { "end_in_escape_3", "", "(foo=bar,var=\\", &le_end_in_escape, 14 },
    { "end_in_name_1", "", "(foo", &le_end_in_name, 4 },
    { "end_in_name_2", "", "(foo=bar,baz", &le_end_in_name, 12},
    { "invalid_name_char_bs", "", "(fo\\", &le_bad_name_char, 3},
    { "invalid_name_char_comma", "", "(fo,", &le_bad_name_char, 3},
    { "invalid_name_char_op", "", "(fo(", &le_bad_name_char, 3},
    { "invalid_name_char_cp", "", "(fo)", &le_bad_name_char, 3},
    { "invalid_empty_name_1", "", "(=", &le_empty_name, 1},
    { "invalid_empty_name_2", "", "(foo=bar,var=(=))", &le_empty_name, 14},

    { "si", "an nice 'lil input", si, 0, 0, {6,3,8} },
    { "value_with_nested_commas", "", "baz(foo,bar,bing)", 0, 0, {1,0,0} },
    { "odd_leaf_1", "odd-but-okay leaf value (maybe user error, maybe should be illegal)", "bar=foo()", 0, 0, {1,0,0} },
    { "odd_leaf_2_esc_lp", "", "bar=foo\\(", 0, 0, {1,0,0} },
    { "odd_leaf_3_nested", "", "(foo=bar(=))", 0,0, {1,1,1} },
    { "off_leaf_4_nested_esc_lp", "", "(biz=bar=foo\\()", 0, 0, {1,1,1} },
  };

  struct lexp_test_run_t : public test_run_t, public virtual nesi, public has_main_t // NESI(help="NESI wrapper for low-level lexp tests; use test_lexp() global func to run w/o NESI", bases=["has_main_t"], type_id="test_lexp", hide=1 )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    void test_print( void ) {
      printf( "tix=%s lexp_tests[tix].desc=%s\n", str(tix).c_str(), str(lexp_tests[tix].desc).c_str() );
    }

    void test_lexp_run( void )
    {
      // note: global parse_lexp() function is inlined here for customization / error generation
      lexp_test_t const & lt = lexp_tests[tix];
      sstr_t sstr;
      sstr.set_from_string( lt.in );
      lex_parse_t lex_parse( sstr );
      p_lexp_t test_ret;
      try {
	p_lexp_t ret = lex_parse.parse_lexp();
	lex_parse.err_if_data_left();
	test_ret = ret;
      } catch( rt_exception const & rte ) {
	assert_st( !test_ret );
	if( !lt.err_fmt ) { test_fail_err( rte.err_msg ); } // expected no error, but got one
	else { 	// check if error is correct one
	  string const exp_err_msg = "error: " + lex_parse.err_str( *lt.err_fmt, lt.err_off );
	  if( rte.err_msg != exp_err_msg ) { test_fail_wrong_err( 
	      strprintf( "  %s\nexpected:\n  %s\n", str(rte.err_msg).c_str(), str(exp_err_msg).c_str() ) ); }
	}
      }
      if( test_ret ) {
	if( lt.err_fmt ) { test_fail_no_err( *lt.err_fmt ); }
	else { 
	  // no error expected, no error occured. check that the string reps and stats agree
	  lexp_stats_t lexp_stats;
	  lexp_stats.set_to_zeros();
	  test_ret->get_stats( lexp_stats );
	  if( !lexp_stats.is_eq( lt.lexp_stats ) ) {
	    test_fail_wrong_res( strprintf( "lexp_stats=%s != lt.lexp_stats=%s", 
					    str(lexp_stats).c_str(), str(lt.lexp_stats).c_str() ) );
	  }
	  string const lexp_to_str( str( *test_ret ) );
	  if( lexp_to_str != lt.in ) { test_fail_wrong_res( strprintf( "lexp_to_str=%s != lt.in=%s\n", 
								       str(lexp_to_str).c_str(), str(lt.in).c_str() ) );
	  }	
	}
      }
    }
    void main( nesi_init_arg_t * nia ) {
      num_fail = 0;
      for( tix = 0; tix < ( sizeof( lexp_tests ) / sizeof( lexp_test_t ) ); ++tix ) { test_lexp_run(); }
      if( num_fail ) { printf( "test_lexp num_fail=%s\n", str(num_fail).c_str() ); }
    }
  };
  // if NESI is borked, and lexp is to blame, you could run the lexp tests with this function
  void test_lexp( void ) { lexp_test_run_t().main( 0 ); } 

#include"gen/lexp.cc.nesi_gen.cc"

}
