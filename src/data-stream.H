// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _DATA_STREAM_H_
#define _DATA_STREAM_H_

#include"boda_base.H"
#include"geom_prim.H"

namespace boda
{
  struct lexp_name_val_map_t;
  typedef lexp_name_val_map_t nesi_init_arg_t;

  struct data_block_t {
    p_uint8_t d; // pointer to data block, null if no data (usually a borrowed pointer from data_stream_t. TODO: add clone/copy iface?
    uint64_t sz; // size of data
    // metadata (optional in general)
    uint64_t frame_ix; // what frame is this? zero based. const_max is invalid.
    uint64_t timestamp_ns; // data-stream-defined timestamp in nanoseconds. may not start at zero.
    data_block_t( void ) : sz(uint64_t_const_max), frame_ix(uint64_t_const_max), timestamp_ns(uint64_t_const_max) { }
    bool valid( void ) const { return d.get(); } // if false, no fields are valid (in particular, sz/metadata is not valid)
  };
  typedef vector< data_block_t > vect_data_block_t; 
  typedef shared_ptr< data_block_t > p_data_block_t; 
  typedef vector< p_data_block_t > vect_p_data_block_t;

  
  struct data_stream_t : virtual public nesi // NESI(help="modes that can be used for reading streams of timestamped data blocks"
                                              // ,is_abstract=1,tid_vn="mode")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string mode; //NESI(help="type of datastream",req=1)
    virtual void data_stream_init( nesi_init_arg_t * const nia ) = 0;
    virtual data_block_t read_next_block( void ) = 0;

    // debugging iface(s) (may only have meaning for certain modes)
    virtual string get_pos_info_str( void ) = 0;
  };
  typedef vector< data_stream_t > vect_data_stream_t; 
  typedef shared_ptr< data_stream_t > p_data_stream_t; 
  typedef vector< p_data_stream_t > vect_p_data_stream_t;

  struct data_sink_t : virtual public nesi // NESI(help="modes that can be used for consuming timestamped data blocks"
                                           // ,is_abstract=1,tid_vn="mode")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string mode; //NESI(help="type of data sink",req=1)
    virtual void data_sink_init( nesi_init_arg_t * const nia ) = 0;
    virtual void consume_block( data_block_t const & db ) = 0;
  };
  typedef vector< data_sink_t > vect_data_sink_t; 
  typedef shared_ptr< data_sink_t > p_data_sink_t; 
  typedef vector< p_data_sink_t > vect_p_data_sink_t;

  struct multi_data_stream_t : virtual public nesi // NESI(help="modes that can be used for reading streams of timestamped data blocks"
                                              // ,is_abstract=1,tid_vn="mode")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string mode; //NESI(help="type of datastream",req=1)
    virtual uint32_t multi_data_stream_init( nesi_init_arg_t * const nia ) = 0; // return # of multiplexed streams
    virtual void multi_read_next_block( vect_data_block_t & dbs ) = 0;

    // debugging iface(s) (may only have meaning for certain modes)
    virtual string get_pos_info_str( void ) = 0;
  };
  typedef vector< multi_data_stream_t > vect_multi_data_stream_t; 
  typedef shared_ptr< multi_data_stream_t > p_multi_data_stream_t; 
  typedef vector< p_multi_data_stream_t > vect_p_multi_data_stream_t;

}

#endif /* _DATA_STREAM_H_ */
