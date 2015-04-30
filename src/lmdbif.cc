// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"lmdbif.H"

namespace boda 
{


  struct lmdb_bench_t : virtual public nesi, public has_main_t // NESI(help="utility to benchmark lmdb read access",
			// bases=["has_main_t"], type_id="lmdb_bench")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t db_fn_2; //NESI(default="./test_db",help="input lmdb dir filename")
    filename_t db_fn; //NESI(default="%(datasets_dir)/imagenet_classification/ilsvrc12_train_lmdb",help="input lmdb dir filename")
    uint32_t do_write_read_test; //NESI(default=0,help="if 1, run simple write/read test (from lmdb source sample-mdb.c).")
    uint32_t do_read_bw_test; //NESI(default=0,help="if 1, run sequential read bw test.")
    uint64_t meg_limit; //NESI(default=0,help="if non-zero, limit bw test to ~ this many meg (1024*1024) of data.")
    uint64_t skip_meg; //NESI(default=0,help="if non-zero, skip this many meg (1024*1024) before checksumming.")
    string skip_past_key; //NESI(default="",help="if non-zero, skip to entry with key > given val (using SET_RANGE).")

    uint64_t delay_after_keys; //NESI(default=0,help="if non-zero, add a delay (spin-loop) after reading N keys.")
    uint64_t delay_nsecs; //NESI(default=0,help="if needed/used delay time in ns.")

    lmdb_state_t lmdb;
    void main( nesi_init_arg_t * nia ) { 
      lmdb.env_open( db_fn.exp, (do_write_read_test ? 0 : MDB_RDONLY) ); //  | MDB_SEQUENTIAL ); // NOTE: needs lmdb sequential opt
      if( do_write_read_test ) { simple_write_read_test(); }
      if( do_read_bw_test ) { read_bw_test(); }
    }
    
    void read_bw_test( void ) {
      timer_t t("read_bw_test");
      MDB_val key, data;
      lmdb.txn_begin( MDB_RDONLY );
      lmdb.cursor_open();
      uint64_t tot_bytes = 0;
      uint64_t last_bytes = 0;
      uint64_t last_update = 0;
      uint64_t checksum = 0;
      uint64_t keys_read = 0;
      if( skip_past_key.size() ) { 
	key.mv_size = skip_past_key.size();
	key.mv_data = (void *)skip_past_key.c_str();
	lmdb.cursor_set_range( &key, &data ); 
      }
      bool done = 0;
      while( (!done) && lmdb.cursor_next( &key, &data ) ) {
	++keys_read;
	if( delay_after_keys && ( (keys_read % delay_after_keys) == 0 ) ) {
	  uint64_t const st = get_cur_time();
	  while( get_cur_time() < (st+delay_nsecs) ) { }
	}
	tot_bytes += data.mv_size;
	if( !( skip_meg && (tot_bytes < (skip_meg*1024*1024) ) ) ) {
	  for( uint32_t i = 0; i != data.mv_size; ++i ) { checksum += ((char *)data.mv_data)[i];}
	}
	if( meg_limit && (tot_bytes > (meg_limit*1024*1204)) ) { done = 1; }	
	if( done || ( t.cur() > (last_update + secs_to_nsecs(2) ) ) ) {
	  double const gbps = double(tot_bytes - last_bytes)/double(t.cur() - last_update) ;
	  printf( "tot_bytes=%s checksum=%s mbps=%s\n", str(tot_bytes).c_str(), str(checksum).c_str(), str(gbps*1000.0).c_str() );
	  last_update = t.cur();
	  last_bytes = tot_bytes;
	}
      }
      lmdb.cursor_close();
      lmdb.txn_abort();
    }
  
    void simple_write_read_test( void ) {
      MDB_val key, data;
      char sval[32];
      key.mv_size = sizeof(int);
      key.mv_data = sval;
      data.mv_size = sizeof(sval);
      data.mv_data = sval;
      sprintf(sval, "%03x %d foo bar", 32, 3141592);
      lmdb.txn_begin( 0 );
      lmdb_ret_check( mdb_put( lmdb.txn, lmdb.dbi, &key, &data, 0 ), "mdb_put" );
      lmdb.txn_commit();

      lmdb.txn_begin( MDB_RDONLY );
      lmdb.cursor_open();
      while( lmdb.cursor_next( &key, &data ) ) {
	printf("key: %p %.*s, data: %p %.*s\n",
	       key.mv_data,  (int) key.mv_size,  (char *) key.mv_data,
	       data.mv_data, (int) data.mv_size, (char *) data.mv_data);
      }
      lmdb.cursor_close();
      lmdb.txn_abort();
    }
  };

#include"gen/lmdbif.cc.nesi_gen.cc"
}
