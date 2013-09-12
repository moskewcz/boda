#include"boda_tu_base.H"
#include"str_util.H"
#include"geom_prim.H"
#include<cassert>
#include<string>
#include<map>
#include<vector>
#include<boost/algorithm/string.hpp>
#include<boost/lexical_cast.hpp>
#include<boost/filesystem.hpp>
#include"pugixml.hpp"

namespace boda 
{
  using namespace std;
  using namespace boost;
  using filesystem::path;
  using filesystem::filesystem_error;
  using namespace pugi;

  double lc_str_d( string const & s ) 
  { 
    try {
      return lexical_cast< double >( s ); 
    }
    catch( bad_lexical_cast & e ) {
      cout << "can't convert '" << s << "' to double." << endl;
      assert(0);
    }
  }
  uint32_t lc_str_u32( string const & s ) 
  { 
    try {
      return lexical_cast< uint32_t >( s ); 
    }
    catch( bad_lexical_cast & e ) {
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

  // clears line and reads one line from in. returns true if at EOF. 
  // note: calls rt_err() if a complete line cannot be read.
  bool ifs_getline( std::string const &fn, p_ifstream in, string & line )
  {
    line.clear();
    // the file should initially be good (including if we just
    // opened it).  note the eof is not set until trying to read
    // past the end. after each line is read, we check for eof, and
    // if we're not at eof, we check that the stream is still good
    // for more reading.
    assert_st( in->good() ); 
    getline(*in, line);
    if( in->eof() ) { 
      if( !line.empty() ) { rt_err( "reading "+fn+": incomplete (no newline) line at EOF:'" + line + "'" ); } 
      return 1;
    }
    else {
      if( !in->good() ) { rt_err( "reading "+fn+ " unknown failure" ); }
      return 0;
    }
  }

  void read_results_file( string const & fn )
  {
    p_ifstream in = ifs_open( fn );  
    string line;
    while( !ifs_getline( fn, in, line ) )
    {
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

  void read_pascal_annotations_for_id( string const & fn, string const & id )
  {
    path const pfn = fn;
    path const ann_dir = pfn.parent_path() / ".." / ".." / "Annotations";
    ensure_is_dir( ann_dir );
    path const ann_file = ann_dir / (id + ".xml");
    ensure_is_regular_file( ann_file );
    
    xml_document doc;
    xml_parse_result result = doc.load_file( ann_file.c_str() );

    if( !result ) { 
      rt_err( strprintf( "loading xml file '%s' failed: %s", ann_file.c_str(), result.description() ) );
    }
    //std::cout << "Load result: " << result.description() << ", mesh name: " << doc.child("annotation").child("object").child_value("name") << std::endl;

  }

  void read_image_list_file( string const & fn )
  {
    p_ifstream in = ifs_open( fn );  
    string line;
    while( !ifs_getline( fn, in, line ) )
    {
      vect_string parts;
      split( parts, line, is_space(), token_compress_on );
      if( (parts.size() == 1) && parts[0].empty() ) { continue; } // skip ws-only lines
      if( parts.size() != 2 ) { 
	rt_err( strprintf( "invalid line in image list file '%s': num of parts != 2 after space "
			   "splitting. line was:\n%s", fn.c_str(), line.c_str() ) ); }
      string const id = parts[0];
      string const pn = parts[1];
      if( (pn != "1") && (pn != "-1") && (pn != "0") ) {
	rt_err( strprintf( "invalid type string in image list file '%s': saw '%s', expected '1', '-1', or '0'.",
			   fn.c_str(), pn.c_str() ) );
      }
      //printf( "id=%s pn=%s\n", id.c_str(), pn.c_str() );
      // load annotations for this image
      read_pascal_annotations_for_id( fn, id );

    }
  }
}
