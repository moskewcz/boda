// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"data-stream.H"
#include"nesi.H"

namespace boda 
{

  struct data_stream_ffmpeg_src_t : virtual public nesi, public data_stream_t // NESI(
                                    // help="parse file with ffmpeg (libavformat,...) output one block per raw video frame",
                                    // bases=["data_stream_t"], type_id="ffmpeg-src")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    virtual string get_pos_info_str( void ) { return string( "ffmpeg-src: pos info TODO" ); }

    virtual bool seek_to_block( uint64_t const & frame_ix ) { return false; }
    
    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
    }
    
    virtual data_block_t proc_block( data_block_t const & db ) {
      data_block_t ret = db;
      return ret;
    }
  };
  
#include"gen/data-stream-ffmpeg.cc.nesi_gen.cc"

}
