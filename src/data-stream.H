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
    p_uint8_t d;
    uint64_t sz;
    uint64_t timestamp_ns;
    data_block_t( void ) : sz(uint64_t_const_max), timestamp_ns(uint64_t_const_max) { }
  };
  
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
  

}

#endif /* _DATA_STREAM_H_ */
