#ifndef MWM_str_util
#define MWM_str_util

#include<string>
#include<vector>
#include<exception>
#include<sstream>
#include<stdio.h>
#include<string.h>

namespace boda
{

  double lc_str_d( char const * const s );
  uint32_t lc_str_u32( char const * const s );
  double lc_str_d( std::string const & s );
  uint32_t lc_str_u32( std::string const & s );

  typedef std::vector< std::string > vect_string;

  std::string strprintf( char const * const format, ... ) __attribute__( (format(__printf__, 1, 2) ) );
  void printstr( std::string const & str );

  std::string xml_escape( std::string const & str );

  inline bool endswith( std::string const & s, std::string const & suf )
  {
    if( s.size() < suf.size() )
    {
      return false;
    }
    size_t const off( s.size() - suf.size() );
    for( size_t i = 0; i < suf.size(); i++ )
    {
      if( suf[i] != s[i+off] )
      {
	return false;
      }
    }
    return true;
  }
  inline bool startswith( std::string const & s, std::string const & pref )
  {
    if( s.size() < pref.size() )
    {
      return false;
    }
    for( size_t i = 0; i < pref.size(); i++ )
    {
      if( pref[i] != s[i] )
      {
	return false;
      }
    }
    return true;
  }

  std::string join( vect_string const & vs, std::string const & sep );
  vect_string split( std::string const & s, char const sep );
  
  inline void strip_suffs( std::string & s, std::string const & suff )
  {
    while( endswith(s,suff) ) { s = s.substr( 0, s.size() - suff.size() ); }
  }


  template< typename T >
  inline std::string str(T const & i)	// convert T i to string
  {
    std::stringstream s;
    s << i;
    return s.str();
  }

  class str_exception : public std::exception
  {
  public:
    char const * const msg;
    str_exception( char const * const msg_ ) : msg(strdup(msg_)) { }
    str_exception( std::string const & msg_ ) : msg(strdup(msg_.c_str())) { }
    virtual char const * what() const throw() { return msg; }
  };
}

#endif // MWM_str_util
