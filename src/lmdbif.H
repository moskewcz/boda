// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _LMDBIF_H_
#define _LMDBIF_H_

#include"str_util.H"
#include"lmdb.h"

namespace boda 
{
  inline void lmdb_ret_check( int const & ret, char const * const func_name ) {
    if( ret ) { rt_err( strprintf( "error: %s: (%d) %s\n", func_name, ret, mdb_strerror( ret ) ) ); }
  }

  struct lmdb_state_t {
    MDB_env *env;
    MDB_dbi dbi;
    bool dbi_valid;
    MDB_txn *txn;
    MDB_cursor *cursor;

    lmdb_state_t( void ) {
      env = 0;
      dbi_valid = 0;
      txn = 0;
      cursor = 0;
    }

    void env_open( string const & fn, uint32_t const & flags ) {
      assert_st( !env ); lmdb_ret_check( mdb_env_create( &env ), "mdb_env_create" ); assert_st( env );
      lmdb_ret_check( mdb_env_open( env, fn.c_str(), flags, 0664 ), "mdb_env_open" );
    }

    void txn_begin( uint32_t const & flags ) { // note: opens db if not open
      assert_st( !txn ); lmdb_ret_check( mdb_txn_begin( env, NULL, flags, &txn ), "mdb_txn_begin" ); assert_st( txn ); 
      if( !dbi_valid ) { lmdb_ret_check( mdb_dbi_open( txn, NULL, 0, &dbi ), "mdb_dbi_open" ); dbi_valid = 1; }
    }
    void txn_abort( void ) { assert_st( txn ); mdb_txn_abort( txn ); txn = 0; }
    void txn_commit( void ) { assert_st(txn); lmdb_ret_check( mdb_txn_commit( txn ) , "mdb_txn_commit" ); txn = 0; }

    void cursor_open( void ) { 
      assert_st( !cursor ); lmdb_ret_check( mdb_cursor_open(txn, dbi, &cursor), "mdb_cursor_open" ); assert_st( cursor ); }
    void cursor_set_range( MDB_val * const key, MDB_val * const data ) {
      assert_st( cursor );
      lmdb_ret_check( mdb_cursor_get( cursor, key, data, MDB_SET_RANGE ), "mdb_cursor_get" );
    }
    bool cursor_next( MDB_val * const key, MDB_val * const data ) {
      assert_st( cursor );
      int rc = mdb_cursor_get( cursor, key, data, MDB_NEXT );
      if( !rc ) { return true; } // key/data read okay
      else if ( rc == MDB_NOTFOUND ) { return false; } // not really and error exactly, no more data
      else { lmdb_ret_check( rc, "mdb_cursor_get" ); } // 'real' error (will not return)
      assert(0); // silence compiler warning
    }
    void cursor_close( void ) { assert_st( cursor ); mdb_cursor_close( cursor ); cursor = 0; }

    void clear( void ) {
      if( cursor ) { cursor_close(); }
      if( txn ) { txn_abort(); }
      if( dbi_valid ) {	assert_st( env ); mdb_dbi_close( env, dbi ); dbi_valid = 0; }
      if( env ) { mdb_env_close( env ); env = 0; }
    }
    ~lmdb_state_t( void ) { clear(); }
  }; 
}
#endif /* _LMDBIF_H_ */
