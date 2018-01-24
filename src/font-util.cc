// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"font-util.H"
#include"has_main.H"
#define STB_TRUETYPE_IMPLEMENTATION  // force following include to generate implementation
#include"ext/stb_truetype.h"


namespace boda
{

  struct test_font_util_t : public virtual nesi, public has_main_t // NESI(help="test of stb_truetype/font-rendering", bases=["has_main_t"], type_id="test-font-util" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t font_fn; //NESI(default="%(boda_dir)/fonts/DroidSansMono.ttf",help="ttf font filename")
    string to_render; //NESI(default="a",help="string to render")
    uint32_t render_scale; //NESI(default=20,help="render scale")
    
    virtual void main( nesi_init_arg_t * nia ) {
      printf( "test-font-util main() begins.\n" );
      p_string font_data = read_whole_fn( font_fn );
      uint8_t * const rp_font_data = (uint8_t * const)&font_data->at(0);
      int const font_offset = stbtt_GetFontOffsetForIndex(rp_font_data,0); // FIXME: doesn't take len of data, so presumably unsafe ...
      if( font_offset == -1 ) { rt_err( "stbtt_GetFontOffsetForIndex() failed" ); }
      stbtt_fontinfo font;
      int const ret = stbtt_InitFont( &font, rp_font_data, font_offset );
      if( ret == 0 ) { rt_err( "stbtt_InitFont() failed" ); }
      
      for( string::const_iterator c = to_render.begin(); c != to_render.end(); ++c ) {
        int w,h;
        unsigned char * const bitmap = stbtt_GetCodepointBitmap(&font, 0,stbtt_ScaleForPixelHeight(&font, render_scale), *c, &w, &h, 0,0);
        for (int j=0; j < h; ++j) {
          for (int i=0; i < w; ++i)
            putchar(" .:ioVM@"[bitmap[j*w+i]>>5]);
          putchar('\n');
        }
      }
    }
  };

#include"gen/font-util.cc.nesi_gen.cc"

}
