#include"boda_tu_base.H"
#include"str_util.H"
#include"geom_prim.H"
#include<cassert>
#include<string>
#include<map>
#include<vector>
#include<boost/algorithm/string.hpp>
#include<boost/lexical_cast.hpp>

namespace boda 
{
  using namespace std;
  using namespace boost;
  double lc_str_d( string const & s ) 
  { 
    try 
    {
      return lexical_cast< double >( s ); 
    }
    catch( bad_lexical_cast & e )
    {
      cout << "can't convert '" << s << "' to double." << endl;
      assert(0);
    }
  }
  uint32_t lc_str_u32( string const & s ) 
  { 
    try 
    {
      return lexical_cast< uint32_t >( s ); 
    }
    catch( bad_lexical_cast & e )
    {
      cout << "can't convert '" << s << "' to uint32_t." << endl;
      assert(0);
    }
  }

  typedef vector< string > vect_string;
  typedef map< string, uint32_t > str_uint32_t_map_t;
  string id_from_image_fn( string const & image )
  {
    const size_t lastDot = image.find_last_of('.');	
    string id = image.substr(0, lastDot);
    const size_t lastSlash = id.find_last_of("/\\");
    if (lastSlash != string::npos)
      id = id.substr(lastSlash + 1);
    return id;
  }

  void read_results_file( string const & fn )
  {
    p_ifstream in = ifs_open( fn );
  
    while( !in->eof() )
    {
      string line;
      getline(*in, line);
      vect_string parts;
      split( parts, line, is_space(), token_compress_on );
      if( (parts.size() == 1) && parts[0].empty() ) { continue; } // skip ws-only lines
      assert( parts.size() == 6 );
      string const id = parts[0];

      double const score = lc_str_d( parts[1] );
      uint32_t const left = lc_str_u32( parts[2] ) - 1;
      uint32_t const top = lc_str_u32( parts[3] ) - 1;
      uint32_t const right = lc_str_u32( parts[4] ) - 1;
      uint32_t const bottom = lc_str_u32( parts[5] ) - 1;
      assert( bottom > top );
      assert( right > left );
      printf( "score=%s\n", str(score).c_str() );

    }
  }
}
