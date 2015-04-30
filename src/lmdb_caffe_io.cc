// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"lmdbif.H"
#include"caffepb.H"
#include"has_main.H"

namespace boda 
{

  struct lmdb_parse_datums_t : virtual public nesi, public has_main_t // NESI(help="parse caffe-style datums stored in an lmdb",
			       // bases=["has_main_t"], type_id="lmdb_parse_datums")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t db_fn; //NESI(default="%(datasets_dir)/imagenet_classification/ilsvrc12_train_lmdb",help="input lmdb dir filename")
    uint64_t num_read; //NESI(default=2,help="read this many records")

    lmdb_state_t lmdb;
    void main( nesi_init_arg_t * nia ) { 
      lmdb.env_open( db_fn.exp, MDB_RDONLY ); 
      MDB_val key, data;
      lmdb.txn_begin( MDB_RDONLY );
      lmdb.cursor_open();
      uint64_t num = 0;
      while( lmdb.cursor_next( &key, &data ) ) {
	//printf( "key.mv_size=%s data.mv_size=%s\n", str(key.mv_size).c_str(), str(data.mv_size).c_str() );
	p_datum_t datum = parse_datum( data.mv_data, data.mv_size );
	//printf( "datum->label=%s\n", str(datum->label).c_str() );
	++num; if( num == num_read ) { break; }
      }
      lmdb.cursor_close();
      lmdb.txn_abort();
    }
  };

#include"gen/lmdb_caffe_io.cc.nesi_gen.cc"

}
