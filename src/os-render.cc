// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"os-render.H"
#include"img_io.H"
#include"has_main.H"
#include"str_util.H"

namespace boda 
{

  struct data_to_img_pts_t : virtual public nesi, public data_stream_t // NESI(help="annotate data blocks (containing point cloud data) with image representations (in as_img field of data block). returns annotated data block.",
                           // bases=["data_stream_t"], type_id="add-img-pts")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    u32_pt_t disp_sz; //NESI(default="300:300",help="X/Y per-stream-image size")

    p_img_t frame_buf;
    
    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      frame_buf = make_shared< img_t >();
      frame_buf->set_sz_and_alloc_pels( disp_sz );
    }

    virtual data_block_t proc_block( data_block_t const & db ) {
      data_block_t ret = db;
      ret.as_img = data_block_to_img_inner( db );
      return ret;
    }

    p_img_t data_block_to_img_inner( data_block_t const & db ) {
      frame_buf->fill_with_pel( db.timestamp_ns );
      return frame_buf;
    }

    virtual string get_pos_info_str( void ) {      
      return strprintf( "data-to-img: disp_sz=%s", str(disp_sz).c_str() );
    }

    
  };

#include"gen/os-render.cc.nesi_gen.cc"

}
