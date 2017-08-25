// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"data-stream.H"
#include"data-stream-file.H"

// for test-gen stream only
#include"rand_util.H" 
#include<boost/functional/hash.hpp>

namespace boda 
{

  uint32_t const mxnet_brick_magic = 0xced7230a;
  uint32_t const mxnet_brick_max_rec_sz = 1<<29;
  uint32_t make_lrec( uint32_t const & cflag, uint32_t const & len ) {
    assert_st( cflag < ( 1 << 3 ) );
    assert_st( len < mxnet_brick_max_rec_sz );
    return len + ( cflag << 29 );
  }
  uint32_t lrec_get_cflag( uint32_t const & lrec ) { return lrec >> 29; }
  uint32_t lrec_get_len( uint32_t const & lrec ) { return (lrec << 3) >> 3; }
  
  struct data_stream_mxnet_brick_t : virtual public nesi, public data_stream_file_t // NESI(help="parse mxnet-brick-style-serialized data stream into data blocks",
                                     // bases=["data_stream_file_t"], type_id="mxnet-brick")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    vect_data_block_t parts;

    void consume_padding( uint32_t const & len ) {
      uint32_t const pad = u32_ceil_align( len, 4 ) - len;
      mfsr.consume_and_discard_bytes( pad );
    }
    
    virtual data_block_t proc_block( data_block_t const & db ) {
      assert_st( parts.empty() );
      data_block_t ret = db;
      if( mfsr.at_eof() ) { return ret; } // at end of stream
      while( 1 ) {
        uint32_t maybe_magic;
        uint32_t lrec;
        if( !mfsr.can_read( sizeof( maybe_magic ) + sizeof(lrec) ) ) {
          rt_err( strprintf( "data_stream_mxnet_brick_t: not at eof, but not enough bytes left in stream to read next record header: "
                             "bytes_left=%s", str( mfsr.bytes_left() ).c_str() ) );
          
        }
        mfsr.read_val( maybe_magic );
        if( maybe_magic != mxnet_brick_magic ) {
          rt_err( strprintf( "data_stream_mxnet_brick_t: expected magic uint32_t of %x, but got %x", mxnet_brick_magic, maybe_magic ) );
        }
        mfsr.read_val( lrec );
        uint32_t const cflag = lrec_get_cflag( lrec );
        uint32_t len = lrec_get_len( lrec );
        parts.push_back( mfsr.consume_borrowed_block( len ) );
        consume_padding( len );
        if( (cflag==0) || (cflag==1) ) {
          if( parts.size() != 1 ) {
            rt_err( strprintf( "error in mxnet brick stream, expected cflag == 2 or 3 in continutation of split record, but saw cflag=%s\n",
                               str(uint32_t(cflag)).c_str() ) );
          }
          if( cflag==0 ) { break; } // non-split record case
        }
        else if( (cflag==2) || (cflag==3) ) {
          if( parts.size() == 1 ) {
            rt_err( strprintf( "error in mxnet brick stream, expected cflag == 0 or 1 at rec start, saw cflag=%s\n",
                               str(uint32_t(cflag)).c_str() ) );
          }
          if( cflag==3 ) { break; } // end of split record case          
        }
      }
      assert_st( !parts.empty() );
      if( parts.size() == 1 ) { ret = parts[0]; parts.clear(); }
      else {
        //printf( "parts.size()=%s\n", str(parts.size()).c_str() );
        // stitch parts: 1) get size. 2) alloc. 3) copy. // FIXME: test!
        uint64_t tot_sz = 0;
        for( vect_data_block_t::const_iterator i = parts.begin(); i != parts.end(); ++i ) {
          if( i != parts.begin() ) { tot_sz += sizeof(mxnet_brick_magic); } // will join parts with magic val
          tot_sz += (*i).sz;
        }
        ret.sz = tot_sz;
        ret.d = ma_p_uint8_t( ret.sz, boda_default_align );
        uint64_t pos = 0;
        for( vect_data_block_t::const_iterator i = parts.begin(); i != parts.end(); ++i ) {
          if( i != parts.begin() ) { // join parts with magic val
            std::copy( (uint8_t const *)&mxnet_brick_magic, ((uint8_t const *)&mxnet_brick_magic)+sizeof(mxnet_brick_magic), ret.d.get()+pos ); 
            pos += sizeof(mxnet_brick_magic);
          }
          std::copy( (*i).d.get(), (*i).d.get()+(*i).sz, ret.d.get()+pos );      
          pos += (*i).sz;
        }
        assert_st( pos == tot_sz );
      }
      ret.timestamp_ns = tot_num_read;
      data_stream_file_block_done_hook( ret );
      parts.clear();
      return ret;
    }

    virtual void data_stream_init( nesi_init_arg_t * nia ) { data_stream_file_t::data_stream_init( nia );  }
  };


  struct data_sink_mxnet_brick_t : virtual public nesi, public data_stream_t // NESI(help="write sequence of blocks (i.e. records) into mxnet brick.",
                                                       // bases=["data_stream_t"],type_id="mxnet-brick")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    filename_t fn; //NESI(default="out.brick",help="output brick filename")
    p_ostream out;
    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      out = ofs_open( fn );
    }
    virtual string get_pos_info_str( void ) { return string("data_sink_mxnet_brick: wrote <NOT_IMPL> records in <NOT_IMPL> bytes"); }

    void write_chunk( uint32_t const & cflag, uint8_t const * const & start, uint64_t const & len ) {
      //printf( "write_chunk: cflag=%s len=%s\n", str(cflag).c_str(), str(len).c_str() );
      bwrite( *out, mxnet_brick_magic );
      uint32_t const lrec = make_lrec( cflag, len );
      bwrite( *out, lrec );
      bwrite_bytes( *out, (char * const)start, len );
    }
    
    virtual data_block_t proc_block( data_block_t const & db ) {
      if( !( db.sz < mxnet_brick_max_rec_sz ) ) { rt_err( strprintf( "mxnet_brick_max_rec_sz=%s but db.sz=%s",
                                                                     str(mxnet_brick_max_rec_sz).c_str(), str(db.sz).c_str() ) ); }
      // split payload at every occurance of magic number
      uint32_t const * const src_data = (uint32_t const *)db.d.get();
      uint32_t final_cflag = 0;
      uint32_t next_part_cflag = 1;
      uint64_t spos = 0;
      for( uint64_t i = 0; i < (db.sz>>2); ++i ) {
        if( src_data[i] == mxnet_brick_magic ) {
          uint64_t const ipos = i << 2;
          final_cflag = 3;
          write_chunk( next_part_cflag, db.d.get()+spos, ipos - spos );
          spos = ipos + 4;
          next_part_cflag = 2;
        }
      }
      write_chunk( final_cflag, db.d.get()+spos, db.sz - spos ); // last record (may be zero size)
      uint32_t const pad = u32_ceil_align( db.sz, 4 ) - db.sz; // padding
      uint32_t const zero = 0;
      bwrite_bytes( *out, (char * const)&zero, pad );
      return data_block_t();
    }
  };

  
  struct data_stream_test_gen_t : virtual public nesi, public data_stream_t // NESI(help="generate a data stream for testing",
                                     // bases=["data_stream_t"], type_id="test-gen")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint64_t rng_seed; //NESI(default="0",help="seed for rng")
    uint64_t blocks_to_gen; //NESI(default="1000",help="non-hash blocks to generate (actual stream will have 2X this many blocks)")
    uint64_t bsz_min; //NESI(default="0",help="min block size")
    uint64_t bsz_max; //NESI(default="32",help="max block size")

    string hex_magic_corruption; //NESI(default="deadbeef", help="hex string of specific binary data to maybe-include in gen'd blocks" )
    uint32_t magic_corruption_cnt; //NESI(default=0,help="how many times to put magic corruption in each gen'd block" )
    
    uint64_t tot_num_gen;
    boost::random::mt19937 gen;
    vect_uint8_t gen_data;
    size_t block_hash;
    string magic_corruption;
    
    // for now, alternate blocks are checksums of prior block
    virtual data_block_t proc_block( data_block_t const & db ) {
      data_block_t ret = db;
      if( !(tot_num_gen&1) ) {
        if( tot_num_gen == (2*blocks_to_gen) ) { return ret; }
        boost::random::uniform_int_distribution<uint64_t> bsz_dist( bsz_min, bsz_max );
        gen_data.resize( bsz_dist(gen) );
        rand_fill_int_vect( gen_data, uint8_t(0), uint8_t_const_max, gen );
        boost::random::uniform_int_distribution<uint64_t> pos_dist( 0, gen_data.size() );
        for( uint32_t i = 0; i != magic_corruption_cnt; ++i ) {
          uint64_t pos = pos_dist(gen);
          for( uint32_t j = 0; (j != magic_corruption.size()) && (pos < gen_data.size()); ++j, ++pos ) {
            gen_data[pos] = magic_corruption[j];
          }
        }
        block_hash = boost::hash_range( gen_data.begin(), gen_data.end() );
        ret.sz = gen_data.size();
        ret.d = ma_p_uint8_t( ret.sz, boda_default_align );
        ret.frame_ix = tot_num_gen;
        ret.timestamp_ns = tot_num_gen;
        std::copy( gen_data.begin(), gen_data.end(), ret.d.get() ); 
      } else {
        // last block was gen'd data. send it's hash.
        ret.sz = sizeof(size_t);
        ret.d = ma_p_uint8_t( ret.sz, boda_default_align );
        std::copy( (uint8_t const *)&block_hash, ((uint8_t const *)&block_hash)+sizeof(block_hash), ret.d.get() );
      }
      ret.frame_ix = tot_num_gen;
      ret.timestamp_ns = tot_num_gen;
      ++tot_num_gen;
      return ret;
    }

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      tot_num_gen = 0;
      gen.seed( rng_seed );
      magic_corruption = unhex( hex_magic_corruption );
    }
    virtual string get_pos_info_str( void ) { return strprintf( "tot_num_gen=%s", str(tot_num_gen).c_str() ); }
  };

  struct data_sink_hash_check_t : virtual public nesi, public data_stream_t // NESI(help="read sequence of block/prior-block-hash-block pairs.",
                                  // bases=["data_stream_t"],type_id="hash-check")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint64_t tot_num_read;
    size_t block_hash;
    virtual void data_stream_init( nesi_init_arg_t * const nia ) { tot_num_read = 0; }
    
    virtual string get_pos_info_str( void ) { return strprintf( "data_sink_hash_check: tot_num_read=%s\n", str(tot_num_read).c_str() ); }

    virtual data_block_t proc_block( data_block_t const & db ) {
      assert_st( db.valid() ); // don't pass invalid blocks?
      if( !(tot_num_read & 1) ) {
        block_hash = boost::hash_range( db.d.get(), db.d.get() + db.sz ); // cache hash
      } else {
        if( db.sz != sizeof(size_t) ) {
          rt_err( strprintf( "expected hash-only block at tot_num_read=%s, but db.sz=%s\n",
                             str(tot_num_read).c_str(), str(db.sz).c_str() ) );
        }
        size_t const fs_block_hash = *((size_t *)db.d.get());
        if( fs_block_hash != block_hash ) {
          rt_err( strprintf( "block hash compare failure: fs_block_hash=%s block_hash=%s\n",
                             str(fs_block_hash).c_str(), str(block_hash).c_str() ) );
        }
      }
      ++tot_num_read;
      return data_block_t();
    }
    // yeah, not the best way/place to check, but better than not checking? fails if we got an odd # of blocks (i.e. last block lost)
    // FIXME: maybe we need a stream-length check too? (to check for block level truncation?)
    ~data_sink_hash_check_t( void ) { assert_st( ! (tot_num_read & 1) ); } 
      
  };


  
#include"gen/data-stream-mxnet.cc.nesi_gen.cc"

}
