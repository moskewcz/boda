// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef DATA_STREAM_FILE_H
#define DATA_STREAM_FILE_H

#include"boda_base.H"
#include"data-stream.H"
#include"mapped-file-util.H"

namespace boda 
{

  struct data_stream_file_t : virtual public nesi, public data_stream_t // NESI(help="parse serialized data stream from file into data blocks",
                              // bases=["data_stream_t"], type_id="file-src", is_abstract=1)
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    filename_t fn; //NESI(req=1,help="input filename")
    p_dims_t out_dims; // NESI(help="set dims of output to this. size must match data size. if not set, will typically emit 'bytes' (1D array of uint8_t, with one dim named 'v'), but default may depend on specific file reader.")
    p_string tag; // NESI(help="if set, set tag field in output data blocks to this. the tag field is a: descritive tag of what this is: camera, camera-front, radar-intensity, ...")
    p_string meta; // NESI(help="if set, set meta field in output data blocks to this. the meta field is a: general class/type of data (i.e. how to set params for visualization, ...) image, lidar, radar, ... ")

    uint64_t tot_num_read; // num blocks read so far
    mapped_file_stream_reader mfsr;
    
    virtual string get_pos_info_str( void ) { return strprintf( "pos=%s tot_num_read=%s", str(mfsr.pos).c_str(), str(tot_num_read).c_str() ); }
    
    virtual void data_stream_init( nesi_init_arg_t * nia ) { // note: to be called explicity by derived classes if they override
      printf( "data_stream_init(): mode=%s fn.exp=%s\n", str(mode).c_str(), str(fn.exp).c_str() );
      mfsr.init( fn );
      tot_num_read = 0;
    }
    void data_stream_file_block_done_hook( data_block_t & ret ) {
      ret.frame_ix = tot_num_read;
      if( out_dims ) { assert_st( ret.nda ); ret.nda->reshape( *out_dims ); }
      if( tag.get() ) { ret.tag = *tag; }
      if( meta.get() ) { ret.meta = *meta; }
      ++tot_num_read;
      if( verbose ) { printf( "ret=%s\n", ret.info_str().c_str() ); }
    }
  };

  struct data_sink_file_t : virtual public nesi, public data_stream_t // NESI(help="base functionality to write stream of data blocks into file",
                            // bases=["data_stream_t"], type_id="file-sink", is_abstract=1)
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    filename_t fn; //NESI(req=1,help="output filename")
    p_ostream out;
    virtual string get_pos_info_str( void ) { return strprintf( "pos=<NOT_IMPL> tot_num_written=<NOT_IMPL>" ); }
    virtual void data_stream_init( nesi_init_arg_t * const nia ) { out = ofs_open( fn );  }
  };
  
}

#endif /* DATA_STREAM_FILE_H */
