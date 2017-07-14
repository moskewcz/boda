// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _RAW_VID_IO_H_
#define _RAW_VID_IO_H_

#include"boda_base.H"
#include"has_main.H"

namespace boda 
{

  struct raw_vid_parse_t : virtual public nesi, public has_main_t // NESI(help="parse raw video into frames",
                           // bases=["has_main_t"], type_id="parse-raw-vid")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t fn; //NESI(default="vid.raw",help="input raw video filename")
    uint64_t num_to_read; //NESI(default=10,help="read this many records")

    uint64_t tot_num_read; // num read so far

    p_img_t read_next_frame( void ) {
      assert_st( tot_num_read <= num_to_read );
      if( tot_num_read == num_to_read ) { return p_img_t(); }
      return p_img_t();
    }

    void init( void ) {
      tot_num_read = 0;

    }
    
    void main( nesi_init_arg_t * nia ) { 
      init();
      while( read_next_frame() ) { }
    }

  };

  uint64_t score_batch( p_nda_float_t const & out_batch, vect_uint32_t const & batch_labels_gt );

}


#endif /* _RAW_VID_IO_H_ */
