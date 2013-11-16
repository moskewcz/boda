#ifndef _XML_UTIL_H_
#define _XML_UTIL_H_

#include"pugixml.hpp"

namespace boda {
  pugi::xml_node xml_must_decend( char const * const fn, pugi::xml_node const & node, char const * const child_name );
}

#endif /* _XML_UTIL_H_ */
