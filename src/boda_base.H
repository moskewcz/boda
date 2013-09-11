#include<cassert>
#include<boost/shared_ptr.hpp>
#include<fstream>
#include<string>
#include<vector>

namespace boda
{
  using boost::shared_ptr;
  typedef shared_ptr< std::ifstream > p_ifstream;
  p_ifstream ifs_open( std::string const & fn );

  typedef std::vector< void * > vect_rp_void;
  std::string stacktrace_str( void );

}
