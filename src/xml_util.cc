#include"boda_tu_base.H"
#include"xml_util.H"
#include"str_util.H"

namespace boda {
  using namespace pugi;
  using namespace std;

  xml_node xml_must_decend( char const * const fn, xml_node const & node, char const * const child_name )
  {
    xml_node ret = node.child(child_name);
    if( !ret ) { 
      rt_err( strprintf( "error: parsing xml file: '%s': expected to find child named '%s' from %s",
			 fn, child_name, (node==node.root()) ? "document root" : 
			 ("node with name '" + string(node.name()) + "'").c_str() ) ); }
    return ret;
  }

}
