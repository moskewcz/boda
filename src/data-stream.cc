// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"img_io.H"
#include"has_main.H"
#include"str_util.H"
#include"data-stream.H"
#include"data-stream-file.H"

namespace boda 
{
  
  struct data_stream_start_stop_skip_t : virtual public nesi, public data_stream_t // NESI(help="wrap another data stream and optionally: skip initial blocks and/or skip blocks after each returned block and/or limit the number of blocks returned.",
                             // bases=["data_stream_t"], type_id="start-stop-skip")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    p_data_stream_t src; //NESI(req=1,help="wrapped data stream")

    // for debugging / skipping:
    uint64_t start_block; //NESI(default=0,help="start at this block")
    uint64_t skip_blocks; //NESI(default=0,help="drop/skip this many blocks after each returned block (default 0, no skipped/dropped blocks)")
    uint64_t num_to_read; //NESI(default=0,help="read this many records; zero for unlimited")

    // internal state:
    uint64_t tot_num_read; // num blocks read so far

    virtual string get_pos_info_str( void ) { return strprintf( "tot_num_read=%s; src info: %s", str(tot_num_read).c_str(), str(src->get_pos_info_str()).c_str() ); }
    // note: preserves frame_ix from nested src.
    virtual data_block_t read_next_block( void ) {
      if( num_to_read && (tot_num_read >= num_to_read) ) { return data_block_t(); }
      data_block_t ret = src->read_next_block();
      ++tot_num_read;
      for( uint32_t i = 0; i != skip_blocks; ++i ) { src->read_next_block(); } // skip blocks if requested
      return ret;
    }

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      src->data_stream_init( nia );
      printf( "data_stream_init(): mode=%s start_block=%s skip_blocks=%s num_to_read=%s\n",
              str(mode).c_str(), str(start_block).c_str(), str(skip_blocks).c_str(), str(num_to_read).c_str() );
      tot_num_read = 0;
      for( uint32_t i = 0; i != start_block; ++i ) { src->read_next_block(); } // skip to start block
    }    
  };

  // overlap timestamp of one stream onto a stream that is missing timestamps. checks that frame_ix's are equal across streams.
  struct data_stream_ts_merge_t : virtual public nesi, public data_stream_t // NESI(help="wrap one data and one timestamp stream and apply the timestamp stream timestamp to the data stream. will complain if data stream has a timestamp already of if frame_ix's don't agree across streams.",
                                  // bases=["data_stream_t"], type_id="ts-merge")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    p_data_stream_t data_src; //NESI(req=1,help="wrapped data stream")
    p_data_stream_t ts_src; //NESI(req=1,help="wrapped data stream")

    virtual string get_pos_info_str( void ) {
      return strprintf( " data_src info: %s -- ts_src info: %s",
                        str(data_src->get_pos_info_str()).c_str(),  str(ts_src->get_pos_info_str()).c_str() );
    }

    virtual data_block_t read_next_block( void ) {
      data_block_t ret = data_src->read_next_block();
      data_block_t ts_db = ts_src->read_next_block();
      if( (!ret.valid()) || (!ts_db.valid()) ) { return data_block_t(); } // if either stream is ended/invalid, silently give ... not ideal?
      if( ret.frame_ix != ts_db.frame_ix ) {
        rt_err( strprintf( "refusing to apply timestamp since stream frame_ix's don't match: data_src frame_ix=%s ts_src frame_ix=%s\n",
                           str(ret.frame_ix).c_str(), str(ts_db.frame_ix).c_str() ) );
      }
      if( ret.timestamp_ns != uint64_t_const_max ) {
        rt_err( strprintf( "refusing to apply timestamp since data stream already has timestamp: data_src timestamp_ns=%s\n",
                           str(ret.timestamp_ns).c_str() ) );
      }
      ret.timestamp_ns = ts_db.timestamp_ns;
      return ret;
    }

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      data_src->data_stream_init( nia );
      ts_src->data_stream_init( nia );
      printf( "data_stream_init(): mode=%s\n", str(mode).c_str() );
    }    
  };
  
  struct data_stream_qt_t : virtual public nesi, public data_stream_file_t // NESI(help="parse qt-style-serialized data stream into data blocks",
                            // bases=["data_stream_file_t"], type_id="qt")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t ts_jump_hack; //NESI(default="0",help="if non-zero, detect and try to fix large timestamp jumps. not a good idea.")

    uint64_t last_ts;
    uint64_t last_delta;
    uint64_t ts_jump_hack_off;
    virtual data_block_t read_next_block( void ) {
      data_block_t ret;
      if( mfsr.pos == timestamp_off ) {
#if 0
        printf( "at timestamp_off: pos=%s\n", str(mfsr.pos).c_str() );
        uint8_t ch;
        while( mfsr.can_read( sizeof(ch) ) ) {
          mfsr.read_val(ch);
          printf( "ch=%s ch=%s\n", str(uint32_t(ch)).c_str(), str(ch).c_str() );
        }
#endif     
        return ret;
      }
      if( !mfsr.can_read( sizeof( ret.timestamp_ns ) ) ) { return ret; } // not enough bytes left for another block
      mfsr.read_val( ret.timestamp_ns );
      if( !mfsr.can_read( sizeof( uint32_t ) ) ) { rt_err( "qt stream: read timestamp, but not enough data left to read payload size" ); }
      mfsr.read_val( ret );
      data_stream_file_block_done_hook( ret );
      if( ts_jump_hack ) {
        ret.timestamp_ns -= ts_jump_hack_off;
        if( last_ts != uint64_t_const_max ) {
          if( (ret.timestamp_ns - last_ts) > 1000000000 ) {
            ts_jump_hack_off += ret.timestamp_ns - last_ts - last_delta;
            printf( "WARNING: ts_jump_hack activated; ts_jump_hack_off=%s\n", str(ts_jump_hack_off).c_str() );
            ret.timestamp_ns = last_ts + last_delta;
          }
        }
        last_delta = ret.timestamp_ns - last_ts;
        last_ts = ret.timestamp_ns;
      }
      return ret;
    }

    // qt stream info/state
    uint64_t timestamp_off;
    uint64_t chunk_off;

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      if( ts_jump_hack ) {
        ts_jump_hack_off = 0;
        last_ts = uint64_t_const_max;
      }
      
      data_stream_file_t::data_stream_init( nia );
      mfsr.need_endian_reverse = 1; // assume stream is big endian, and native is little endian. could check this ...
      uint32_t ver;
      mfsr.read_val( ver );
      string tag;
      mfsr.read_val( tag );
      data_block_t header;
      mfsr.read_val( header );
      mfsr.read_val( timestamp_off );
      mfsr.read_val( chunk_off );
      uint64_t duration_ns;
      mfsr.read_val( duration_ns );
      printf( "  qt stream header: ver=%s tag=%s header.size()=%s timestamp_off=%s chunk_off=%s duration_ns=%s\n",
              str(ver).c_str(), str(tag).c_str(), str(header.sz).c_str(), str(timestamp_off).c_str(),
              str(chunk_off).c_str(), str(duration_ns).c_str() );
      if( mfsr.size() > timestamp_off ) {
        printf( "   !! warning: (size() - timestamp_off)=%s bytes at end of file will be ignored\n",
                str((mfsr.size() - timestamp_off)).c_str() );
      }
    }
  };

  struct data_stream_dumpvideo_t : virtual public nesi, public data_stream_file_t // NESI(help="parse dumpvideo data stream into data blocks",
                                   // bases=["data_stream_file_t"], type_id="dumpvideo")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    virtual data_block_t read_next_block( void ) {
      data_block_t ret;
      uint32_t block_sz;
      if( !mfsr.can_read( sizeof( block_sz ) ) ) { return ret; } // not enough bytes left for another block
      mfsr.read_val( block_sz );
      ret = mfsr.consume_borrowed_block( block_sz ); // note: timestamp not set here
      data_stream_file_block_done_hook( ret );
      return ret;
    }
  };

  // parse stream from text file, one block per line, with a one-line header (which is currently ignored)
  struct data_stream_text_t : virtual public nesi, public data_stream_file_t // NESI(help="parse data stream (dumpvideo/qt) into data blocks",
                             // bases=["data_stream_file_t"], type_id="text")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t timestamp_fix; //NESI(default=0,help="timestamp field-index: use the N'th field as a decimal timestamp in seconds (with fractional part).")
    uint32_t frame_ix_fix; //NESI(default=0,help="frame-ix field-index: use the N'th field as a integer frame index.")

    // set timestamp from field of text line stored in block
    void set_timestamp_from_text_line( data_block_t & v ) {
      string line( v.d.get(), v.d.get()+v.sz );
      vect_string parts = split( line, ' ' );
      if( !( timestamp_fix < parts.size() ) || !( frame_ix_fix < parts.size() ) ) {
        rt_err( strprintf( "can't parse timestamp and frame_ix from fields %s and %s; line had %s fields; full line=%s\n",
                           str(timestamp_fix).c_str(), str(frame_ix_fix).c_str(), str(parts.size()).c_str(), str(line).c_str() ) );
      }
      //if( verbose ) { printf( "parts[text_tsfix]=%s\n", str(parts[text_tsfix]).c_str() ); }
      double const ts_d_ns = lc_str_d( parts[timestamp_fix] ) * 1e9;
      v.timestamp_ns = lround(ts_d_ns);
      v.frame_ix = lc_str_u64(parts[frame_ix_fix]);
    }

    virtual data_block_t read_next_block( void ) {
      data_block_t ret;
      if( !mfsr.can_read( 1 ) ) { return ret; } // not enough bytes left for another block
      mfsr.read_line_as_block( ret );
      set_timestamp_from_text_line( ret );
      data_stream_file_block_done_hook( ret );
      return ret;
    }

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      data_stream_file_t::data_stream_init( nia );
      data_block_t header;
      mfsr.read_line_as_block( header );
      printf( "  text stream header.sz=%s\n", str(header.sz).c_str() );
    }
  };

  struct block_info_t {
    uint16_t block_id;
    uint16_t rot_pos;
  } __attribute__((packed));

  struct laser_info_t {
    uint16_t distance;
    uint8_t intensity;
  } __attribute__((packed));

  struct data_stream_velodyne_t : virtual public nesi, public data_stream_t // NESI(help="parse data stream (velodyne) into per-full-revolution data blocks by merging across packets",
                             // bases=["data_stream_t"], type_id="velodyne")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    p_data_stream_t vps; //NESI(req=1,help="underlying velodyne packet stream")

    uint32_t fbs_per_packet; //NESI(default="12",help="firing blocks per packet")
    uint32_t beams_per_fb; //NESI(default="32",help="beams per firing block")
    uint32_t status_bytes; //NESI(default="6",help="bytes of status at end of block")

    uint32_t fb_sz;
    uint32_t packet_sz;
    uint16_t last_rot;
    
    // internal state:
    uint64_t tot_num_read; // num blocks read so far

    virtual string get_pos_info_str( void ) { return strprintf( "tot_num_read=%s vps info:%s",
                                                                str(tot_num_read).c_str(), vps->get_pos_info_str().c_str() ); }

    virtual data_block_t read_next_block( void ) {
      bool packet_is_rot_start = 0;
      data_block_t db;
      while( !packet_is_rot_start ) {
        db = vps->read_next_block();
        if( !db.d.get() ) { return db; } // not enough data for another frame, give up
        if( db.sz != packet_sz ) { rt_err(
            strprintf( "lidar decode expected packet_sz=%s but got block with dv.sz=%s",
                       str(packet_sz).c_str(), str(db.sz).c_str() ) ); }
        if( verbose ) { printf( "data_to_img_null: db.sz=%s db.timestamp_ns=%s\n",
                                str(db.sz).c_str(), str(db.timestamp_ns).c_str() ); }
        for( uint32_t i = 0; i != fbs_per_packet; ++i ) {
          block_info_t bi = *(block_info_t *)(db.d.get()+fb_sz*i);
          if( verbose ) {
            printf( "bi.block_id=%hx bi.rot_pos=%hu\n", bi.block_id, bi.rot_pos );
          }
          if( bi.rot_pos < last_rot ) { packet_is_rot_start = 1; }
          last_rot = bi.rot_pos;
        }
      }
      // last packet was a frame start, so return it. FIXME: obv. this drops lots'o'data, but should emit one packet per
      // rotation, which is all we want for now.
      db.frame_ix = tot_num_read;
      ++tot_num_read;
      if( verbose ) { printf( "velodyne ret.sz=%s ret.timestamp_ns=%s\n", str(db.sz).c_str(), str(db.timestamp_ns).c_str() ); }
      return db;
    }

    // init/setup

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      printf( "data_stream_init(): mode=%s\n", str(mode).c_str() );
      tot_num_read = 0;
      // setup internal state
      fb_sz = sizeof( block_info_t ) + beams_per_fb * sizeof( laser_info_t );
      packet_sz = fbs_per_packet * fb_sz + status_bytes;
      last_rot = 0;        
      vps->data_stream_init( nia );
    }
    
    void main( nesi_init_arg_t * nia ) { 
      data_stream_init( nia );
      while( read_next_block().d.get() ) { }
    }
  };

  struct scan_data_stream_t : virtual public nesi, public has_main_t // NESI(
                              // help="testing mode to scan N data streams one-by-one, and print total number of blocks read for each.",
                              // bases=["has_main_t"], type_id="scan-data-stream")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    vect_p_data_stream_t stream; //NESI(help="data stream to read images from")
    uint32_t full_dump_ix; //NESI(default="-1",help="for this stream, dump all block sizes")
    p_data_sink_t sink; //NESI(help="optional; if specific, send all block to this data sink")

    void main( nesi_init_arg_t * nia ) {
      if( sink ) { sink->data_sink_init( nia ); }
      for( uint32_t i = 0; i != stream.size(); ++i ) {
        uint64_t last_ts = 0;
        stream[i]->data_stream_init( nia );
        while( 1 ) {
          data_block_t db = stream[i]->read_next_block();
          if( !db.valid() ) { break; }
          if( db.timestamp_ns <= last_ts ) {
            printf( "**ERROR: ts did not increase: stream[%s] db.timestamp_ns=%s last_ts=%s stream[i]->get_pos_info_str()=%s\n",
                    str(i).c_str(), str(db.timestamp_ns).c_str(), str(last_ts).c_str(), str(stream[i]->get_pos_info_str()).c_str() );
          }
          if( (i == full_dump_ix) || last_ts == 0 ) { // if on first block, dump out ts
            printf( "stream[%s] sz=%s first_ts=%s get_pos_info_str()=%s\n",
                    str(i).c_str(), str(db.sz).c_str(), str(db.timestamp_ns).c_str(), str(stream[i]->get_pos_info_str()).c_str() );
          }
          last_ts = db.timestamp_ns;
          if( sink ) { sink->consume_block( db ); }
        }
        printf( "stream[%s] last_ts=%s get_pos_info_str()=%s\n",
                str(i).c_str(), str(last_ts).c_str(), str(stream[i]->get_pos_info_str()).c_str() );
      }
    }

  };

  typedef vector< data_block_t > vect_data_block_t; 
  typedef vector< vect_data_block_t > vect_vect_data_block_t;


  uint64_t ts_delta( data_block_t const & a, data_block_t const & b ) {
    return ( a.timestamp_ns > b.timestamp_ns ) ? ( a.timestamp_ns - b.timestamp_ns ) : ( b.timestamp_ns - a.timestamp_ns );
  }
  
  struct multi_data_stream_sync_t : virtual public nesi, public multi_data_stream_t // NESI(
                                    // help="take N data streams, with one as primary, and output one block across all streams for each primary stream block, choosing the nearest-by-timestamp-to-the-primary-block-timestamp-block for each non-primary stream. ",
                                    // bases=["multi_data_stream_t"], type_id="sync")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    vect_p_data_stream_t stream; //NESI(help="input data streams")
    uint64_t max_delta_ns; //NESI(default="0",help="if non-zero, refuse to emit a primary block if, for any secondary stream, no block with a timestamp <= max_detla_ns from the primary block can be found (i.e. all secondary streams must have a 'current' block).")
    uint32_t psix; //NESI(default="0",help="primary stream index (0 based)")

    vect_vect_data_block_t cur_dbs;


    virtual uint32_t multi_data_stream_init( nesi_init_arg_t * const nia ) {
      if( !( psix < stream.size() ) ) { rt_err( strprintf( "psix=%s must be < stream.size()=%s\n",
                                                           str(psix).c_str(), str(stream.size()).c_str() ) ); }
      for( uint32_t i = 0; i != stream.size(); ++i ) {
        stream[i]->data_stream_init( nia );
      }
      cur_dbs.resize( stream.size() );
      for( uint32_t i = 0; i != stream.size(); ++i ) {
        if( i == psix ) { continue; }
        cur_dbs[i].push_back( stream[i]->read_next_block() );
        if( !cur_dbs[i][0].valid() ) { rt_err( strprintf( "no blocks at all in stream i=%s\n", str(i).c_str() ) ); }
        cur_dbs[i].push_back( stream[i]->read_next_block() );
      }
      return stream.size();
    }
    
    virtual string get_pos_info_str( void ) {
      string ret = "\n";
      for( uint32_t i = 0; i != stream.size(); ++i ) {
        ret += "  " + str(i) + ": " + stream[i]->get_pos_info_str() + "\n";
      }
      return ret;
    }

    virtual void multi_read_next_block( vect_data_block_t & dbs ) {
      while ( 1 ) {
        data_block_t pdb = stream[psix]->read_next_block();
        dbs.clear();
        dbs.resize( stream.size() );
        if( !pdb.valid() ) { return; } // done
        if( verbose ) { printf( "-- psix=%s pdb.timestamp=%s\n", str(psix).c_str(), str(pdb.timestamp_ns).c_str() ); }
        bool ret_valid = 1;
        for( uint32_t i = 0; i != stream.size(); ++i ) {
          if( i == psix ) { continue; }
          vect_data_block_t & i_dbs = cur_dbs[i];
          assert( i_dbs.size() == 2 ); // always 2 entries, but note that head may be invalid/end-of-stream
          while( i_dbs[1].valid() && ( i_dbs[1].timestamp_ns < pdb.timestamp_ns ) ) {
            i_dbs[0] = i_dbs[1];
            i_dbs[1] = stream[i]->read_next_block();
          }
          assert_st( i_dbs[0].valid() ); // tail should always be valid since we require all streams to be non-empty
          uint64_t const tail_delta = ts_delta( pdb, i_dbs[0] );
          bool const head_is_closer = i_dbs[1].valid() && ( ts_delta( pdb, i_dbs[1] ) < tail_delta );
          data_block_t sdb = i_dbs[head_is_closer];
          assert_st( sdb.valid() );
          if( verbose ) { printf( "i=%s sdb.timestamp=%s\n", str(i).c_str(), str(sdb.timestamp_ns).c_str() ); }
          if( max_delta_ns && (ts_delta( pdb, sdb ) > max_delta_ns) ) {
            if( verbose ) { printf( "*** no current-enough secondary block found. skipping primary block.\n" ); }
            ret_valid = 0;
          } else {
            dbs[i] = sdb;
          }
        }
        if( ret_valid ) {
          dbs[psix] = pdb;
          return;
        }
        // else continue
      }
    }
    
  };


  struct scan_multi_data_stream_t : virtual public nesi, public has_main_t // NESI(
                                    // help="scan multi data stream ",
                                    // bases=["has_main_t"], type_id="scan-data-stream-multi")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support    
    p_multi_data_stream_t multi_stream; //NESI(help="input data multi stream")

    void main( nesi_init_arg_t * nia ) {
      uint32_t num_srcs = multi_stream->multi_data_stream_init( nia );
      vect_data_block_t dbs;
      bool had_data = 1;
      while( had_data ) {
        multi_stream->multi_read_next_block( dbs );
        if( num_srcs ) { assert_st( dbs.size() == num_srcs ); } // num_srcs == 0 --> dynamic # of blocks
        had_data = 0;
        printf( "----\n" );
        for( uint32_t i = 0; i != dbs.size(); ++i ) {
          if( dbs[i].valid() ) {
            had_data = 1;
            printf( "  i=%s dbs[i].timestamp_sz=%s\n", str(i).c_str(), str(dbs[i].timestamp_ns).c_str() );
          }
        }
      }
    }
  };

#include"gen/data-stream.H.nesi_gen.cc"
#include"gen/data-stream.cc.nesi_gen.cc"

}
