// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"font-util.H"
#include"has_main.H"

namespace boda
{

  struct test_font_util_t : public virtual nesi, public has_main_t // NESI(help="test of stb_truetype/font-rendering", bases=["has_main_t"], type_id="test-font-util" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    virtual void main( nesi_init_arg_t * nia ) {
      printf( "test-font-util main() begins.\n" );
    }
  };

#include"gen/font-util.cc.nesi_gen.cc"

}
