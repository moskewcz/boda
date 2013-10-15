#include"boda_tu_base.H"
#include"lexp.H"
#include"str_util.H"

namespace boda {
  using std::string;

  // error format strings
  string const le_unparsed_data( "unparsed data remaining (expected end of input)" );
  string const le_end_after_list_item( "unexpected end of input (expecting ',' to continue list or ')' to end list)" );
  string const le_end_missing_cp( "unexpected end of input (expecting more close parens)" );
  string const le_end_in_escape( "unexpected end of input after escape char '\\' (expected char)" );
  string const le_end_in_name( "unexpected end of input (expecting '=' to end name)" );
  string const le_bad_name_char( "invalid name character in name" );
  string const le_empty_name( "invalid empty name (no chars before '=' in name)" );

  void sstr_t::set_from_string( string const &s )
  {
    b = 0; e = s.size(); // set b/e
    base = p_uint8_t( (uint8_t *)malloc( s.size() ), free ); // allocate space
    memcpy( base.get(), &s[0], sz() ); // copy string data
  }

  // for testing/debugging (particularly regressing testing), create
  // some simple statistics of a lexp.
  void lexp_nv_t::get_stats( lexp_stats_t & lex_stats ) {

  }
  void lexp_t::get_stats( lexp_stats_t & lex_stats ) {

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
  std::ostream & operator<<(std::ostream & os, sstr_t const & v) {
    return os.write( (char const *)v.base.get()+v.b, v.sz() );
  }
  std::ostream & operator<<(std::ostream & os, lexp_nv_t const & v) {
    return os << v.n << "=" << (*v.v);
  }
  // if we want to re-create the input string exactly
  // *without* just printing src, and this node is a non-empty list,
  // we need to know if there was an optional trailing
  // comma. wonderful!
  bool lexp_t::src_has_trailing_comma( void ) const
  {
    assert( src.sz() > 1 );
    if( !kids.size() ) { return 0; }
    assert( src.sz() > 2 );
    assert( src.base.get()[src.e-1] == ')' );
    return src.base.get()[src.e-2] == ',';
  }
  std::ostream & operator<<(std::ostream & os, lexp_t const & v) {
    if( v.leaf_val.exists() ) { return os << v.src; } // leaf case. note: we print the 'raw' value here.
    else { // otherwise, list case
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

    string err_str( string const & msg );
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
  };

  void lex_parse_t::err( string const & msg ) { rt_err( err_str( msg ) ); }
  string lex_parse_t::err_str( string const & msg )
  {
    uint32_t const max_ccs = 35; // max context chars
    uint32_t const cb = ( cur_off() >= (s.b + max_ccs) ) ? (cur_off() - max_ccs) : s.b;
    uint32_t const ce = ( (cur_off() + max_ccs) <= s.e ) ? (cur_off() + max_ccs) : s.e;
    uint32_t const eo = (cur_off() - cb);
    
    return strprintf( "at offset %s=%s: %s in context:\n%s%s%s\n%s^",
		       str(cur_off()).c_str(),
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
    assert_st( (cur_c() == 0) || (cur_c() == ',') || (cur_c() == ')') ); // end-of-lexp postcondition
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
      ret->kids.push_back( kid );
      if( cur_c() == 0 ) { err( le_end_after_list_item ); }
      else if( (cur_c() == ',') ) { next_c(); }
      else { assert_st( cur_c() == ')' ); } // must hold due to parse_lexp() postcondition. note: will exit loop now.
    }
    assert_st( cur_c() == ')' ); // end of list precondition
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

  struct lexp_stats_t
  {
    uint32_t num_leaf;
    uint32_t num_list;
    uint32_t num_kids;
    lexp_stats_t( void ) : num_leaf(0), num_list(0), num_kids(0) { }
  };

  struct lexp_test_t
  {
    string desc;
    string in;
    string const * err_fmt;
    uint32_t const err_off;
    lexp_stats_t exp;
  };
  string const si1 = "(foo=baz,biz=boo,bing=(f1=21,faz=na),hap=(baz=234,fin=12))";
  lexp_test_t lexp_tests[] = {
    { "si1+junk", si1 + ")sdf", &le_unparsed_data, 0 },
#if 0
   &le_end_after_list_item( "unexpected end of input (expecting ',' to continue list or ')' to end list)" );
   &le_end_missing_cp( "unexpected end of input (expecting more close parens)" );
   &le_end_in_escape( "unexpected end of input after escape char '\\' (expected char)" );
   &le_end_in_name( "unexpected end of input (expecting '=' to end name)" );
   &le_bad_name_char( "invalid name character in name" );
   &le_empty_name( "invalid empty name (no chars before '=' in name)" );
#endif
  };

  void test_fail_no_err( string const & msg )
  {
    printf( "test failed: no error. expected error was:\n  %s\n", str(msg).c_str() );
  }

  void test_lexp_run( lexp_test_t const & lt )
  {
    p_lexp_t test_ret;
    try {
      test_ret = parse_lexp( lt.in );
    } catch( rt_exception const & rte ) {
      
    }
    if( test_ret ) {
      if( lt.err_fmt ) { test_fail_no_err( *lt.err_fmt ); }
      else {
	lexp_stats_t lexp_stats;
	test_ret->get_stats( lexp_stats );
	
      }
    }

  }

  void test_lexp( void )
  {
    for( uint32_t i = 0; i < ( sizeof( lexp_tests ) / sizeof( lexp_test_t ) ); ++i ) {
      printf( "i=%s lexp_tests[i].desc=%s\n", str(i).c_str(), str(lexp_tests[i].desc).c_str() );
      test_lexp_run( lexp_tests[i] );
    }

  }




}
