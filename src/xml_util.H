// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _XML_UTIL_H_
#define _XML_UTIL_H_

#include"ext/pugixml.hpp"

namespace boda {
  pugi::xml_node xml_file_get_root( pugi::xml_document & doc, std::string const & xml_fn );
  pugi::xml_node xml_must_decend( char const * const fn, pugi::xml_node const & node, char const * const child_name );
}

#endif /* _XML_UTIL_H_ */
