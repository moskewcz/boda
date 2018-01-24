// Copyright (c) 2018, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef FONT_UTIL_H
#define FONT_UTIL_H

#include "boda_base.H"


namespace boda 
{
  struct font_render_t : virtual public nesi // NESI(help="modes that can be used for rendering fonts",is_abstract=1,tid_vn="be")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string be; //NESI(req=1,help="name of back-end to use for font rendering")
    virtual p_uint8_t render_char( int8_t const & c, int & w, int & h ) = 0;
  };
  typedef shared_ptr< font_render_t > p_font_render_t; 

}

#endif /* FONT_UTIL_H */
