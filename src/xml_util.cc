// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"xml_util.H"
#include"str_util.H"

namespace boda {
  using namespace pugi;

  xml_node xml_file_get_root( xml_document & doc, string const & xml_fn ) {
    ensure_is_regular_file( xml_fn );
    xml_parse_result result = doc.load_file( xml_fn.c_str() );
    if( !result ) { 
      rt_err( strprintf( "loading xml file '%s' failed: %s", xml_fn.c_str(), result.description() ) );
    }    
    xml_node xn = doc.first_child();
    assert_st( !xn.empty() ); // doc should have a child (the root)
    assert_st( xn.next_sibling().empty() ); // doc should have exactly one root elem
    return xn;
  }


  xml_node xml_must_decend( char const * const fn, xml_node const & node, char const * const child_name )
  {
    xml_node ret = node.child(child_name);
    if( !ret ) { 
      rt_err( strprintf( "error: parsing xml file: '%s': expected to find child named '%s' from %s",
			 fn, child_name, (node==node.root()) ? "document root" : 
			 ("node with name '" + string(node.name()) + "'").c_str() ) ); }
    return ret;
  }

  xml_attribute xml_must_get_attr( char const * const fn, xml_node const & node, char const * const attr_name )
  {
    xml_attribute ret = node.attribute(attr_name);
    if( !ret ) { 
      rt_err( strprintf( "error: parsing xml file: '%s': expected to find attribute named '%s' from %s",
			 fn, attr_name, (node==node.root()) ? "document root" : 
			 ("node with name '" + string(node.name()) + "'").c_str() ) ); }
    return ret;
  }

}
