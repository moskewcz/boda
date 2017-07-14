// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _RAW_VID_IO_H_
#define _RAW_VID_IO_H_

#include"boda_base.H"
#include"geom_prim.H"
#include"img_io.H"
#include"has_main.H"

namespace boda 
{

  struct raw_vid_io_t : virtual public nesi, public has_main_t // NESI(help="parse raw video into frames",
                           // bases=["has_main_t"], type_id="parse-raw-vid")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t fn; //NESI(default="vid.raw",help="input raw video filename")
    u32_pt_t frame_sz; //NESI(default="10 5",help="X/Y frame size")
    uint32_t padding; //NESI(default="0",help="frame padding")
    
    
    uint64_t num_to_read; //NESI(default=100,help="read this many records")

    uint64_t tot_num_read; // num frames read so far
    p_mapped_file_source fn_map;
    uint64_t fn_map_pos;
    uint32_t frame_stride;

    p_img_t frame_buf;
    
    p_img_t read_next_frame( void ) {
      assert_st( tot_num_read <= num_to_read );
      if( tot_num_read == num_to_read ) { return p_img_t(); }
      ++tot_num_read;
      frame_buf->fill_with_pel( grey_to_pel( (10 * tot_num_read) & 0xff ) );
      return frame_buf;
    }

    void raw_vid_init( void ) {
      tot_num_read = 0;
      fn_map = map_file_ro( fn );
      fn_map_pos = 0;
      frame_stride = frame_sz.dims_prod() + padding;
      frame_buf = make_shared< img_t >();
      frame_buf->set_sz_and_alloc_pels( frame_sz );
      // unsigned ret = lodepng::decode( lp_pels, lp_w, lp_h, (uint8_t *)mfile->data(), mfile->size() )
    }
    
    void main( nesi_init_arg_t * nia ) { 
      raw_vid_init();
      while( read_next_frame() ) { }
    }

  };

  uint64_t score_batch( p_nda_float_t const & out_batch, vect_uint32_t const & batch_labels_gt );

}


#endif /* _RAW_VID_IO_H_ */
