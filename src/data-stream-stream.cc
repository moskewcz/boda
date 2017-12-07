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
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    string stream_url; //NESI(req=1,help="input stream url")
    zi_bool at_eof;
    p_stream_t stream;
    virtual data_block_t proc_block( data_block_t const & db ) {
      data_block_t ret = db;
      if( at_eof.v ) { return ret; } // at end of stream
      while( 1 ) {
        try {
          bread( *stream, ret.nda );
          break;
        } catch( rt_exception const & rte ) {
          if( startswith( rte.what(), "error: socket-read-error" ) ) {
            printstr( rte.what() + string("\n") );
            printf( "-- reseting socket, waiting for new client --\n" );
            data_stream_init( 0 );
            continue;
          }
          throw;
        }
      }
      if( verbose ) { printf( "stream-src: ret.info_str()=%s\n", ret.info_str().c_str() ); }
      return ret;
    }
    virtual void data_stream_init( nesi_init_arg_t * nia ) {
       stream = make_stream_t( stream_url, 0 );
       stream->wait_for_worker();
    }
    virtual string get_pos_info_str( void ) { return strprintf( "stream-src: at_eof=%s", str(uint32_t(at_eof.v)).c_str() ); }
  };
  
  struct data_stream_stream_sink_t : virtual public nesi, public data_stream_t // NESI(help="write data blocks to a boda stream url",
                                     // bases=["data_stream_t"], type_id="stream-sink")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    string stream_url; //NESI(req=1,help="output stream url")
    zi_bool at_eof;
    p_stream_t stream;
    virtual data_block_t proc_block( data_block_t const & db ) {
      if( verbose ) { printf( "stream-sink: db.info_str()=%s\n", db.info_str().c_str() ); }
      bwrite( *stream, db.nda );
      return db;
    }
    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      stream = make_stream_t( stream_url, 1 );
    }
    virtual string get_pos_info_str( void ) { return strprintf( "stream-src: <no-state>" ); }
  };

#include"gen/data-stream-stream.cc.nesi_gen.cc"

}
