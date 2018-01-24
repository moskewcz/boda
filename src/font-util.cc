// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"font-util.H"
#include"has_main.H"
#define STB_TRUETYPE_IMPLEMENTATION  // force following include to generate implementation
#include"ext/stb_truetype.h"


namespace boda
{

  
  struct ttf_font_render_t : public virtual nesi, public font_render_t // NESI(help="stb_truetype-based ttf-font-rendering to bitmaps", bases=["font_render_t"], type_id="ttf" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t font_fn; //NESI(default="%(boda_dir)/fonts/DroidSansMono.ttf",help="ttf font filename")
    uint32_t render_scale; //NESI(default=20,help="render scale")

    p_string font_data;
    stbtt_fontinfo font;

    void lazy_init( void ) {
      if( font_data ) { return; }
      font_data = read_whole_fn( font_fn );
      uint8_t * const rp_font_data = (uint8_t * const)&font_data->at(0);
      int const font_offset = stbtt_GetFontOffsetForIndex(rp_font_data,0); // FIXME: doesn't take len of data, so presumably unsafe ...
      if( font_offset == -1 ) { rt_err( "stbtt_GetFontOffsetForIndex() failed" ); }
      int const ret = stbtt_InitFont( &font, rp_font_data, font_offset );
      if( ret == 0 ) { rt_err( "stbtt_InitFont() failed" ); }
    }

    virtual p_uint8_t render_char( int8_t const & c, int & w, int & h ) {
      lazy_init();
      return p_uint8_t( stbtt_GetCodepointBitmap(&font, 0,stbtt_ScaleForPixelHeight(&font, render_scale), c, &w, &h, 0,0), free );
    }
      
  };


  struct test_font_util_t : public virtual nesi, public has_main_t // NESI(help="test of stb_truetype/font-rendering", bases=["has_main_t"], type_id="test-font-util" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    p_font_render_t font_renderer; //NESI(default="(be=ttf)",help="font renderer to use for test")

    string to_render; //NESI(default="a",help="string to render")
    virtual void main( nesi_init_arg_t * nia ) {
      printf( "test-font-util main() begins.\n" );

      for( string::const_iterator c = to_render.begin(); c != to_render.end(); ++c ) {
        int w,h;
        p_uint8_t bitmap = font_renderer->render_char( *c, w, h );
        for (int j=0; j < h; ++j) {
          for (int i=0; i < w; ++i)
            putchar(" .:ioVM@"[bitmap.get()[j*w+i]>>5]);
          putchar('\n');
        }
      }
    }

  };

#include"gen/font-util.H.nesi_gen.cc"
#include"gen/font-util.cc.nesi_gen.cc"

}
