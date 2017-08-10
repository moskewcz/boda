// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"data-stream.H"
#include"data-stream-file.H"
#include"nesi.H"

namespace boda 
{

  struct data_stream_pcap_t : virtual public nesi, public data_stream_file_t // NESI(
                              // help="parse pcap file and output one block per packet/block in the file",
                              // bases=["data_stream_file_t"], type_id="pcap")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      data_stream_file_t::data_stream_init( nia );
    }
    
    virtual data_block_t read_next_block( void ) {
      data_block_t ret;
      return ret;
    }

  };

  struct data_sink_pcap_t : virtual public nesi, public data_sink_file_t // NESI(
                              // help="parse pcap file and output one block per packet/block in the file",
                              // bases=["data_sink_file_t"], type_id="pcap")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    virtual void data_sink_init( nesi_init_arg_t * const nia ) {
      data_sink_file_t::data_sink_init( nia );
    }

    virtual void consume_block( data_block_t const & db ) { }

    
  };

#include"gen/data-stream-pcap.cc.nesi_gen.cc"

}
