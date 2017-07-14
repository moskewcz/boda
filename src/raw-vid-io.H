// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _RAW_VID_IO_H_
#define _RAW_VID_IO_H_

#include"boda_base.H"
#include"geom_prim.H"
#include"img_io.H"
#include"has_main.H"
#include<boost/iostreams/device/mapped_file.hpp>

namespace boda 
{

  struct raw_vid_io_t : virtual public nesi, public has_main_t // NESI(help="parse raw video into frames",
                           // bases=["has_main_t"], type_id="parse-raw-vid")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t fn; //NESI(default="vid.raw",help="input raw video filename")
    u32_pt_t frame_sz; //NESI(default="10 5",help="X/Y frame size")
    uint32_t padding; //NESI(default="0",help="frame padding")
    uint32_t bytes_per_pel; //NESI(default="2",help="bytes per pel")

    uint64_t start_frame; //NESI(default=0,help="start at this frame")
    uint64_t skip_frames; //NESI(default=0,help="drop/skip this many frames after each returned frame (default 0, no skipped/dropped frames)")

    uint64_t num_to_read; //NESI(default=0,help="read this many records; zero for unlimited")

    uint64_t tot_num_read; // num frames read so far
    p_mapped_file_source fn_map;
    uint64_t fn_map_pos;
    uint32_t frame_stride;

    p_img_t frame_buf;
    
    p_img_t read_next_frame( void ) {
      if( num_to_read && (tot_num_read == num_to_read) ) { return p_img_t(); }
      uint64_t const frame_sz_bytes = frame_sz.dims_prod() * bytes_per_pel; // assumes 2 bytes per pel
      if( !( (frame_sz_bytes + fn_map_pos) < fn_map->size() ) ) { return p_img_t(); } // not enough bytes left for another frame
      uint16_t const * const rp_frame = (uint16_t const *)((uint8_t const *)fn_map->data() + fn_map_pos);
      // copy and convert frame data
      for( uint32_t y = 0; y < frame_sz.d[1]; ++y ) {
        uint32_t * const dest_data = frame_buf->get_row_addr( y );
        uint16_t const * const src_data = rp_frame + y*frame_sz.d[0];
        for( uint32_t x = 0; x < frame_sz.d[0]; ++x ) { dest_data[x] = grey_to_pel( src_data[x] >> 4 ); }
      }
      //frame_buf->fill_with_pel( grey_to_pel( (10 * tot_num_read) & 0xff ) );
      ++tot_num_read;
      fn_map_pos += frame_stride; // move to next raw frame
      for( uint32_t i = 0; i != skip_frames; ++i ) { fn_map_pos += frame_stride; } // skip frames if requested
      return frame_buf;
    }

    void raw_vid_init( void ) {
      tot_num_read = 0;
      fn_map = map_file_ro( fn );
      frame_stride = frame_sz.dims_prod()*bytes_per_pel + padding;
      frame_buf = make_shared< img_t >();
      frame_buf->set_sz_and_alloc_pels( frame_sz );
      fn_map_pos = start_frame * frame_stride;
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
