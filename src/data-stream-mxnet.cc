// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"data-stream.H"
#include"data-stream-file.H"

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
    
    virtual data_block_t read_next_block( void ) {
      assert_st( parts.empty() );
      data_block_t ret;
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
        ret.d = ma_p_uint8_t( tot_sz, boda_default_align );
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


  struct data_sink_mxnet_brick_t : virtual public nesi, public data_sink_t // NESI(help="write sequence of blocks (i.e. records) into mxnet brick.",
                                                       // bases=["data_sink_t"],type_id="mxnet-brick")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    filename_t fn; //NESI(default="vid.raw",help="input raw video filename")
    p_ostream out;
    virtual void data_sink_init( nesi_init_arg_t * const nia ) {
      out = ofs_open( fn );
    }
    void write_chunk( uint32_t const & cflag, uint8_t const * const & start, uint64_t const & len ) {
      bwrite( *out, mxnet_brick_magic );
      uint32_t const lrec = make_lrec( cflag, len );
      bwrite( *out, lrec );
      bwrite_bytes( *out, (char * const)start, len );
    }
    
    virtual void consume_block( data_block_t const & db ) {
      if( !( db.sz < mxnet_brick_max_rec_sz ) ) { rt_err( strprintf( "mxnet_brick_max_rec_sz=%s but db.sz=%s",
                                                                     str(db.sz).c_str(), str(mxnet_brick_max_rec_sz).c_str() ) ); }
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
    }
  };

  
#include"gen/data-stream-mxnet.cc.nesi_gen.cc"

}
