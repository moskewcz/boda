#include"boda_tu_base.H"

namespace boda 
{

  using namespace std;

// opens a ifstream that will raise expections for all errors (not
// including eof). note: this function itself will raise if the open()
// fails.
  p_ifstream ifs_open( std::string const & fn )
  {
    p_ifstream ret( new ifstream );
    ret->exceptions( ifstream::failbit | ifstream::badbit );
    ret->open( fn.c_str() );
    return ret;
  }
}
