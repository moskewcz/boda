#include"str_util.H"
#include<stdarg.h>
#include<assert.h>
#include<malloc.h>

namespace boda
{

  std::string strprintf( char const * const format, ... )
  {
    va_list ap;
    char *s = 0;
    va_start( ap, format );
    int const va_ret = vasprintf( &s, format, ap );
    assert( va_ret > 0 );
    va_end( ap );
    assert( s );
    std::string ret( s );
    free(s);
    return ret;
  }

  void printstr( std::string const & str )
  {
    printf( "%s", str.c_str() );
  }


}

