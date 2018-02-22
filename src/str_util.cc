// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include<stdarg.h>
#include<assert.h>
#include<malloc.h>
#include<boost/lexical_cast.hpp>
#include<boost/algorithm/string.hpp>
#include<boost/algorithm/hex.hpp>
#include<cxxabi.h>

namespace boda
{
  using std::string;
  
  string hex( std::string const & s ) { string ret; boost::algorithm::hex( s.begin(), s.end(), std::back_inserter( ret ) ); return ret; }
  string unhex( std::string const & s ) { string ret; boost::algorithm::unhex( s.begin(), s.end(), std::back_inserter( ret ) );return ret;}

  string cxx_demangle( string const & sym ) {
    int dm_ret = 0;
    std::unique_ptr< char > dm_fn( abi::__cxa_demangle( sym.c_str(), 0, 0, &dm_ret ) );
    if( dm_ret == 0 ) { return string( dm_fn.get() ); } // demangled okay, return copy of demangled result
    return sym; // can't demangle, return copy of input 
  }

  string get_part_before( string const & s, string const & to_find ) {
    size_t const ix = s.find( to_find );
    if( ix == string::npos ) { return s; }
    else { return string( s, 0, ix ); }
  }
  string get_part_after( string const & s, string const & to_find ) {
    size_t const ix = s.find( to_find );
    if( ix == string::npos ) { return string(); }
    else { return string( s, ix+to_find.size(), string::npos ); }
  }

  double lc_str_d( char const * const s )
  { 
    try { return boost::lexical_cast< double >( s ); }
    catch( boost::bad_lexical_cast & e ) { rt_err( strprintf("can't convert '%s' to double.", s ) ); }
  }
  uint32_t lc_str_u32( char const * const s )
  { 
    try { return boost::lexical_cast< uint32_t >( s ); }
    catch( boost::bad_lexical_cast & e ) { rt_err( strprintf("can't convert '%s' to uint32_t.", s ) ); }
  }
  uint64_t lc_str_u64( char const * const s )
  { 
    try { return boost::lexical_cast< uint64_t >( s ); }
    catch( boost::bad_lexical_cast & e ) { rt_err( strprintf("can't convert '%s' to uint64_t.", s ) ); }
  }
  int32_t lc_str_i32( char const * const s )
  { 
    try { return boost::lexical_cast< int32_t >( s ); }
    catch( boost::bad_lexical_cast & e ) { rt_err( strprintf("can't convert '%s' to int32_t.", s ) ); }
  }
  int64_t lc_str_i64( char const * const s )
  { 
    try { return boost::lexical_cast< int64_t >( s ); }
    catch( boost::bad_lexical_cast & e ) { rt_err( strprintf("can't convert '%s' to int64_t.", s ) ); }
  }
  double lc_str_d( string const & s ) { return lc_str_d( s.c_str() ); } 
  uint32_t lc_str_u32( string const & s ) { return lc_str_u32( s.c_str() ); }
  uint64_t lc_str_u64( string const & s ) { return lc_str_u64( s.c_str() ); }
  int32_t lc_str_i32( string const & s ) { return lc_str_i32( s.c_str() ); } 
  int64_t lc_str_i64( string const & s ) { return lc_str_i64( s.c_str() ); }

  string join( vect_string const & vs, string const & sep ) {
    string ret;
    for( vect_string::const_iterator i = vs.begin(); i != vs.end(); ++i ) {
      if( i != vs.begin() ) { ret += sep; }
      ret += *i;
    }
    return ret;
  }

  // note: not reversable (i.e. just sanitization, not escaping)
  string as_pyid_fixme( string const & s ) {
    string ret;
    for (string::const_iterator i = s.begin(); i != s.end(); ++i) {
      switch (*i) {
      case '-': ret += "_"; break;
      case '.': ret += "_"; break;
      default: ret.push_back( *i );
      }
    }
    return ret;
  }
  string as_py_str_list( vect_string const & vs ) { 
    vect_string vs_id;
    for( vect_string::const_iterator i = vs.begin(); i != vs.end(); ++i ) { vs_id.push_back( "\""+(*i)+"\"" ); }
    return "[ " + join(vs_id,", ") + " ]"; 
  }

  // for split(), # elements output (or size of returned vector for ret-flavor) is always 1 + (count of sep in s)
  void split( vect_string & out, std::string const & s, char const sep ) {
    string::const_iterator b = s.begin();
    for( string::const_iterator i = s.begin(); i != s.end(); ++i) {
      if( (*i) == sep ) {
	out.push_back( string( b, i ) );
	b = i+1;
      }
    }
    out.push_back( string( b, s.end() ) );
  }
  vect_string split( std::string const & s, char const sep ) {
    vect_string ret;
    split( ret, s, sep );
    return ret;
  }

  // note: leading or trailing space will create empty strings in the front or back of the output vector.
  vect_string split_ws( std::string const & s ) {
    vect_string parts;
    boost::algorithm::split( parts, s, boost::algorithm::is_space(), boost::algorithm::token_compress_on );
    return parts;
  }

  vect_string split_space_tab_colon( std::string const & s ) {
    vect_string parts;
    boost::algorithm::split( parts, s, boost::algorithm::is_any_of("\t :"), boost::algorithm::token_compress_on );
    return parts;
  }

  string strip_ws( std::string const & s ) { return boost::algorithm::trim_copy( s ); }
  void strip_ws_inplace( std::string & s ) { return boost::algorithm::trim( s ); }


  string strprintf( char const * const format, ... )
  {
    va_list ap;
    char *s = 0;
    va_start( ap, format );
    int const va_ret = vasprintf( &s, format, ap );
    assert( va_ret > 0 );
    va_end( ap );
    assert( s );
    string ret( s );
    free(s);
    return ret;
  }

  // FIXME: factor out dupe junk around vasprintf() calls? can't do without ret more string copy overhead? hmm.
  filename_t filename_t_printf( filename_t const & fn, ... ) // note: can't use gcc format-string checking here
  {
    filename_t ret;
    va_list ap;
    char *s = 0;
    int va_ret = 0;

    va_start( ap, fn );
    va_ret = vasprintf( &s, fn.in.c_str(), ap );
    assert( va_ret > 0 );
    va_end( ap );
    assert( s );
    ret.in = string( s );
    free(s); s = 0; va_ret = 0;

    va_start( ap, fn );
    va_ret = vasprintf( &s, fn.exp.c_str(), ap );
    assert( va_ret > 0 );
    va_end( ap );
    assert( s );
    ret.exp = string( s );
    free(s); s = 0;

    return ret;
  }

  void printstr( string const & str )
  {
    printf( "%s", str.c_str() );
  }

  string replace_chars_with_char( string const & s, string const & chars_to_find, char const rep_with ) {
    string ret;
    for (string::const_iterator i = s.begin(); i != s.end(); ++i) {
      if( chars_to_find.find(*i) != string::npos ) { ret.push_back( rep_with ); }
      else { ret.push_back( *i ); }
    }
    return ret;
  }

  bool maybe_replace_str_with_str( string & s, string const & find_s, string const & rep_s ) {
    string ret;
    size_t const fp = s.find( find_s, 0 );
    if( fp == string::npos ) { return 0; }
    s = s.substr( 0, fp ) + rep_s + s.substr( fp+find_s.size(), string::npos );
    return 1;
  }
  
  string strip_ending_chars( string const & s, string const & chars_to_strip ) { 
    string ret;
    size_t epos = s.size();
    while( epos && ( chars_to_strip.find( s[epos - 1] ) != string::npos ) ) { --epos; }
    return string( s, 0, epos );
  }

  string xml_escape( string const & str )
  {
    string ret;
    for (string::const_iterator i = str.begin(); i != str.end(); ++i) {
      switch (*i)
      {
      case '&': ret += "&amp;"; break;
      case '<': ret += "&lt;"; break;
      case '>': ret += "&gt;"; break;
      case '"': ret += "&quot;"; break;
      default: ret.push_back( *i );
      }
    }
    return ret;
  }

  string shell_escape( string const & str )
  {
    string ret;
    for (string::const_iterator i = str.begin(); i != str.end(); ++i) {
      switch (*i)
      {
      case '\'': ret += "'\\''"; break;
      default: ret.push_back( *i );
      }
    }
    return "'" + ret + "'";
  }

  // pretty-printing support (with units); ported from boda's python prettyprint.py
  // string pad( uint32_t const & v, string const & s ) {  } // TODO
  string pp_val_part( double const & v, bool const & force ) { 
    if( v < 10.0 ) { return strprintf( "%.2f", v ); }
    if( v < 100.0 ) { return strprintf( "%.1f", v ); }
    if( (v < 1000.0) or force ) { return strprintf( "%.0f", v ); }
    return "***"; // too big to render
  }
  string pp_val( double const & orig_v ) { 
    if( std::isnan(orig_v) ) { return "NAN"; }
    double v = orig_v;
    int32_t exp = 0; // engineering exponent step: orig_v = v 10^(3*exp)
    assert_st( v >= 0.0 );
    while( v < 1.0 ) { 
      v *= 1000.0; --exp; 
      if( exp < -4 ) { return str(orig_v); } // too small, give up
    }
    // while pp_val_part returns its 'too-big' return (i.e. "***" currently)
    string ret;
    while( 1 ) {
      ret = pp_val_part(v,0);
      if( ret != pp_val_part(1e6,exp==5) ) { break; }
      v /= 1000.0; ++exp;
    }
    if( exp < 0 ) { return ret+"munp"[- 1 - exp]; }
    if( exp == 0 ) { return ret; } // no size suffix
    assert_st( exp <= 5 ); // should have forced earlier otherwise
    return ret+"KMGTP"[exp - 1];
  }
  string pp_secs( double const & v, bool const & verbose ) { return pp_val(v) + (verbose ? " SECS" : "s"); }
  string pp_flops( double const & v, bool const & verbose ) { return pp_val(v) + (verbose ? " FLOPS" : "F"); }
  string pp_bytes( double const & v, bool const & verbose ) { return pp_val(v) + (verbose ? " BYTES" : "B"); }

  string pp_bps( double const & v, bool const & verbose ) { return pp_val(v) + (verbose ? " BYTES/SEC" : "B/s"); }
  string pp_fpb( double const & v, bool const & verbose ) { return pp_val(v) + (verbose ? " FLOPS/BYTE" : "F/B"); }
  string pp_fps( double const & v, bool const & verbose ) { return pp_val(v) + (verbose ? " FLOPS/SEC" : "F/s"); }
  string pp_fpspw( double const & v, bool const & verbose ) { return pp_val(v) + (verbose ? " FLOPS/SEC/WATT" : "F/s/W"); }
  string pp_joules( double const & v, bool const & verbose ) { return pp_val(v) + (verbose ? " JOULES" : "J"); }

}

