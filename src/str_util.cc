#include"str_util.H"
#include<stdarg.h>
#include<assert.h>
#include<malloc.h>

namespace boda
{
  using std::string;

  string join( vect_string const & vs, string const & sep ) {
    string ret;
    for( vect_string::const_iterator i = vs.begin(); i != vs.end(); ++i ) {
      if( i != vs.begin() ) { ret += sep; }
      ret += *i;
    }
    return ret;
  }

  // size of return value is always 1 + (count of sep in s)
  vect_string split( std::string const & s, char const sep ) {
    vect_string ret;
    string::const_iterator b = s.begin();
    for( string::const_iterator i = s.begin(); i != s.end(); ++i) {
      if( (*i) == sep ) {
	ret.push_back( string( b, i ) );
	b = i+1;
      }
    }
    ret.push_back( string( b, s.end() ) );
    return ret;
  }

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

  void printstr( string const & str )
  {
    printf( "%s", str.c_str() );
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
}

