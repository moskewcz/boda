#include<cassert>
#include<boost/shared_ptr.hpp>
#include<fstream>
#include<string>

namespace boda
{
  using boost::shared_ptr;
  typedef shared_ptr< std::ifstream > p_ifstream;
  p_ifstream ifs_open( std::string const & fn );
}
