// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"data-stream.H"
#include"stream_util.H"

namespace boda 
{

  struct data_stream_stream_src_t : virtual public nesi, public data_stream_t // NESI(help="read data blocks from a boda stream url",
                                     // bases=["data_stream_t"], type_id="stream-src")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string stream_url; //NESI(req=1,help="input stream url")
    zi_bool at_eof;
    virtual data_block_t proc_block( data_block_t const & db ) {
      data_block_t ret = db;
      if( at_eof.v ) { return ret; } // at end of stream
      return ret;
    }
    virtual void data_stream_init( nesi_init_arg_t * nia ) { }
    virtual string get_pos_info_str( void ) { return strprintf( "stream-src: at_eof=%s", str(uint32_t(at_eof.v)).c_str() ); }
  };
  
  struct data_stream_stream_sink_t : virtual public nesi, public data_stream_t // NESI(help="write data blocks to a boda stream url",
                                     // bases=["data_stream_t"], type_id="stream-sink")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string stream_url; //NESI(req=1,help="output stream url")
    zi_bool at_eof;
    virtual data_block_t proc_block( data_block_t const & db ) {
      return db;
    }
    virtual void data_stream_init( nesi_init_arg_t * nia ) { }
    virtual string get_pos_info_str( void ) { return strprintf( "stream-src: <no-state>" ); }
  };

#include"gen/data-stream-stream.cc.nesi_gen.cc"

}
