// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _LMDB_CAFFE_IO_H_
#define _LMDB_CAFFE_IO_H_

#include"boda_base.H"
#include"lmdbif.H"
#include"caffepb.H"
#include"has_main.H"

namespace boda 
{

  struct lmdb_parse_datums_t : virtual public nesi, public has_main_t // NESI(help="parse caffe-style datums stored in an lmdb",
			       // bases=["has_main_t"], type_id="lmdb_parse_datums")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t db_fn; //NESI(default="%(datasets_dir)/imagenet_classification/ilsvrc12_val_lmdb",help="input lmdb dir filename")
    uint64_t num_to_read; //NESI(default=10,help="read this many records")

    lmdb_state_t lmdb;
    uint64_t tot_num_read; // num read so far

    p_datum_t read_next_datum( void ) {
      assert_st( tot_num_read <= num_to_read );
      if( tot_num_read == num_to_read ) { return p_datum_t(); }
      MDB_val key, data;      
      bool const ret = lmdb.cursor_next( &key, &data );
      if( !ret ) { return p_datum_t(); }
      ++tot_num_read;
      p_datum_t datum = parse_datum( data.mv_data, data.mv_size );
      return datum;
    }

    void read_batch_of_datums( p_nda_float_t & in_batch, vect_uint32_t & labels ) {
      assert_st( labels.empty() );
      MDB_val key, data;   
      assert_st( in_batch->dims.sz() == 4 );
      for( uint32_t i = 0; i != in_batch->dims.dims(0); ++i ) {
	if( tot_num_read == num_to_read ) { return; }
	bool const ret = lmdb.cursor_next( &key, &data );
	if( !ret ) { break; }
	labels.push_back( parse_datum_into( in_batch, i, data.mv_data, data.mv_size ) );
	++tot_num_read;
      }
    }

    void lmdb_open_and_start_read_pass( void ) {
      lmdb.env_open( db_fn.exp, MDB_RDONLY ); 
      lmdb.txn_begin( MDB_RDONLY );
      lmdb.cursor_open();
      tot_num_read = 0;
    }
    
    void main( nesi_init_arg_t * nia ) { 
      lmdb_open_and_start_read_pass();
      while( read_next_datum() ) { }
    }

  };

  uint64_t score_batch( p_nda_float_t const & out_batch, vect_uint32_t const & batch_labels_gt );

}


#endif /* _LMDB_CAFFE_IO_H_ */
