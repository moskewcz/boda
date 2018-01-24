// Copyright (c) 2018, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef FONT_UTIL_H
#define FONT_UTIL_H

#include "boda_base.H"


namespace boda 
{
  struct rendered_char_t {
    p_uint8_t bitmap;
    int w;
    int h;
    int xoff;
    int yoff;
    int advance_pels;
    int lsb_pels;
  };
  typedef shared_ptr< rendered_char_t > p_rendered_char_t; 

  struct font_render_t : virtual public nesi // NESI(help="modes that can be used for rendering fonts",is_abstract=1,tid_vn="be")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string be; //NESI(req=1,help="name of back-end to use for font rendering")
    virtual p_rendered_char_t render_char( int8_t const & c ) = 0;
    virtual int kern_advance_pels( int8_t const & c1, int8_t const & c2 ) = 0;
  };
  typedef shared_ptr< font_render_t > p_font_render_t; 

}

#endif /* FONT_UTIL_H */
