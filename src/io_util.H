#ifndef _IO_UTIL_H_
#define _IO_UTIL_H_

#include"boda_base.H"
#include"timers.H"
#include<boost/algorithm/string.hpp>

namespace boda 
{
  template< typename T >
  void read_text_file( vector< T > & out, std::string const & fn )
  {
    timer_t t("read_text_file");
    p_ifstream in = ifs_open( fn );  
    string line;
    while( !ifs_getline( fn, in, line ) )
    {
      vect_string parts;
      boost::algorithm::split( parts, line, boost::algorithm::is_space(), boost::algorithm::token_compress_on );
      if( (parts.size() == 1) && parts[0].empty() ) { continue; } // skip ws-only lines
      T t;
      t.read_from_line_parts( parts, 0 );
      out.push_back( t );
    }
  }
}
#endif /* _IO_UTIL_H_ */
