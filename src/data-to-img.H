// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _DATA_TO_IMG_H_
#define _DATA_TO_IMG_H_

#include"boda_base.H"

namespace boda
{
  struct img_t;
  typedef shared_ptr< img_t > p_img_t;
  struct data_block_t;
  
  struct lexp_name_val_map_t;
  typedef lexp_name_val_map_t nesi_init_arg_t;

  struct data_to_img_t : virtual public nesi // NESI(help="convert raw data blocks into images",is_abstract=1,tid_vn="mode")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string mode; //NESI(help="type of raw image",req=1)
    virtual void data_to_img_init( nesi_init_arg_t * const nia ) = 0;
    virtual p_img_t data_block_to_img( data_block_t const & db ) = 0;
    virtual string data_block_to_str( data_block_t const & db ) = 0;
    virtual p_nda_t data_block_to_nda( data_block_t const & db ) = 0;
    // debugging iface(s) (may only have meaning for certain modes)
    virtual void set_samp_pt( i32_pt_t const & samp_pt_ ) = 0;
  };
  typedef vector< data_to_img_t > vect_data_to_img_t; 
  typedef shared_ptr< data_to_img_t > p_data_to_img_t; 
  typedef vector< p_data_to_img_t > vect_p_data_to_img_t;
  
}

#endif /* _DATA_TO_IMG_H_ */
