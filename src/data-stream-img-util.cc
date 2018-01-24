// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"img_io.H"
#include"str_util.H"
#include"data-stream.H"
#include"font-util.H"

namespace boda 
{
  struct data_stream_img_add_text_t : virtual public nesi, public data_stream_t  // NESI(help="add text to image in data stream (modify in-place)",
                                      // bases=["data_stream_t"], type_id="img-add-text")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    p_font_render_t font_renderer; //NESI(default="(be=ttf)",help="font renderer for annotating images")
    i32_pt_t text_pos; //NESI(req=1,help="text pt")
    string text_str; //NESI(req=1,help="text string")

    virtual void data_stream_init( nesi_init_arg_t * const nia ) { }
    virtual string get_pos_info_str( void ) { return strprintf( "img-add-text: text_pos=%s text_str=%s\n", str(text_pos).c_str(), str(text_str).c_str() ); }

    virtual data_block_t proc_block( data_block_t const & db ) {
      data_block_t ret = db;
      if( !ret.as_img ) { rt_err( "img-add-text: expected a data block with an image" ); }
      render_text_to_img( font_renderer, ret.as_img, text_pos, text_str );
      return ret;
    }

  };
  
#include"gen/data-stream-img-util.cc.nesi_gen.cc"    
}                        
                           
