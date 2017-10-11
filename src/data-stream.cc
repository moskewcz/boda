// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"img_io.H"
#include"has_main.H"
#include"str_util.H"
#include"data-stream.H"
#include"data-stream-file.H"
#include"ext/half.hpp" // for adj-angle over nda elems, which might be half

namespace boda 
{

  std::ostream & operator << ( std::ostream & out, data_block_t const & db ) { out << db.info_str(); return out; }
  string data_block_t::info_str( void ) const {
    string ret;
    if( nda.get() ) {
      ret += strprintf( "dims=%s frame_ix=%s timestamp_ns=%s meta=%s tag=%s",
                        str(nda->dims).c_str(), str(frame_ix).c_str(), str(timestamp_ns).c_str(), meta.c_str(), tag.c_str() );
    }
    if( subblocks ) {
      ret += strprintf( "subblocks->size()=%s [", str(subblocks->size()).c_str() );
      for( vect_data_block_t::const_iterator i = subblocks->begin(); i != subblocks->end(); ++i ) {
        if( i != subblocks->begin() ) { ret += " , "; }
        ret += (*i).info_str();
      }
    }
    return ret;
  }

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
    virtual void set_opt( data_stream_opt_t const & opt ) { src->set_opt( opt ); }

    virtual bool seek_to_block( uint64_t const & frame_ix ) {
      // FIXME: do something with tot_num_read here?
      return src->seek_to_block( frame_ix );
    }

    virtual string get_pos_info_str( void ) { return strprintf( "tot_num_read=%s; src info: %s", str(tot_num_read).c_str(), str(src->get_pos_info_str()).c_str() ); }
    // note: preserves frame_ix from nested src.
    virtual data_block_t proc_block( data_block_t const & db ) {
      if( num_to_read && (tot_num_read >= num_to_read) ) { return data_block_t(); }
      data_block_t ret = src->proc_block( db );
      ++tot_num_read;
      for( uint32_t i = 0; i != skip_blocks; ++i ) { src->proc_block(data_block_t()); } // skip blocks if requested
      return ret;
    }

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      src->data_stream_init( nia );
      printf( "data_stream_init(): mode=%s start_block=%s skip_blocks=%s num_to_read=%s\n",
              str(mode).c_str(), str(start_block).c_str(), str(skip_blocks).c_str(), str(num_to_read).c_str() );
      tot_num_read = 0;
      for( uint32_t i = 0; i != start_block; ++i ) { src->proc_block(data_block_t()); } // skip to start block // FIXME: use seek here? prob. not.
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

    virtual void set_opt( data_stream_opt_t const & opt ) { data_src->set_opt( opt ); ts_src->set_opt( opt ); }

    virtual string get_pos_info_str( void ) {
      return strprintf( " data_src info: %s -- ts_src info: %s",
                        str(data_src->get_pos_info_str()).c_str(),  str(ts_src->get_pos_info_str()).c_str() );
    }

    virtual bool seek_to_block( uint64_t const & frame_ix ) {
      if( !data_src->seek_to_block( frame_ix ) ) { return false; }
      if( !ts_src->seek_to_block( frame_ix ) ) {
        assert_st( 0 ); // FIXME: we need to roll back the seek that worked above here -- but with no 'tell' that's hard ..,
      }
      return true;
    }

    virtual data_block_t proc_block( data_block_t const & db ) {
      data_block_t ret = data_src->proc_block(db);
      data_block_t ts_db = ts_src->proc_block(data_block_t());
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
    virtual data_block_t proc_block( data_block_t const & db ) {
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
      printf( "  qt stream header: ver=%s tag=%s header.sz()=%s timestamp_off=%s chunk_off=%s duration_ns=%s\n",
              str(ver).c_str(), str(tag).c_str(), str(header.sz()).c_str(), str(timestamp_off).c_str(),
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
    virtual data_block_t proc_block( data_block_t const & db ) {
      data_block_t ret;
      uint32_t block_sz;
      if( !mfsr.can_read( sizeof( block_sz ) ) ) { return ret; } // not enough bytes left for another block. FIXME: should be an error?
      mfsr.read_val( block_sz );
      if( block_sz == uint32_t_const_max ) { return ret; } // end of dumpvideo stream marker
      ret.nda = mfsr.consume_borrowed_block( block_sz ); // note: timestamp not set here
      data_stream_file_block_done_hook( ret ); 
      ret.meta = "image";
      ret.tag = "camera-dumpvideo";
      return ret;
    }
  };

  // parse stream from text file, one block per line, with a one-line header (which is currently ignored)
  struct data_stream_text_t : virtual public nesi, public data_stream_file_t // NESI(help="parse text (line-oriented) stream into data blocks",
                             // bases=["data_stream_file_t"], type_id="text-src")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t timestamp_fix; //NESI(default=0,help="timestamp field-index: use the N'th field as a decimal timestamp in seconds (with fractional part).")
    uint32_t frame_ix_fix; //NESI(default=0,help="frame-ix field-index: use the N'th field as a integer frame index.")

    // set timestamp from field of text line stored in block
    void set_timestamp_from_text_line( data_block_t & v ) {
      assert( v.nda->dims.tn == "uint8_t" );
      string line( (uint8_t*)v.nda->rp_elems(), (uint8_t*)v.nda->rp_elems()+v.sz() );
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

    virtual data_block_t proc_block( data_block_t const & db ) {
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
      printf( "  text stream header.sz()=%s\n", str(header.sz()).c_str() );
    }
  };

  // parse stream from text file, one block per line, with a one-line header (which is currently ignored)
  struct data_stream_csv_t : virtual public nesi, public data_stream_file_t // NESI(help="parse csv stream into data blocks",
                             // bases=["data_stream_file_t"], type_id="csv-src")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t timestamp_fix; //NESI(default=0,help="timestamp field-index: use the N'th field as an integer timestamp in nanoseconds")
    uint32_t skip_header_lines; //NESI(default=1,help="skip this many initial lines (i.e. column or other headers).")
    
    uint64_t get_timestamp_from_parts( vect_string const & parts ) {
      if( !( timestamp_fix < parts.size() ) ) {
        rt_err( strprintf( "can't parse timestamp from field %s; line had %s fields; parts=%s\n",
                           str(timestamp_fix).c_str(), str(parts.size()).c_str(), str(parts).c_str() ) );
      }
      return lc_str_u64( parts[timestamp_fix] );
    }

    virtual data_block_t proc_block( data_block_t const & db ) {
      data_block_t ret = db;
      if( mfsr.at_eof() ) { return db; }
      vect_vect_string ts_lines_parts;
      uint32_t max_parts = 0;
      while( !mfsr.at_eof() ) { // gather block of same-ts lines
        string line = mfsr.read_line_as_string();
        strip_ws_inplace( line ); // mainly to remove trailing newline, which lc_str_d() doesn't like
        vect_string const parts = split( line, ',' );
        uint64_t const line_ts_ns = get_timestamp_from_parts( parts );
        if( ret.timestamp_ns == uint64_t_const_max ) { ret.timestamp_ns = line_ts_ns; } // first line sets block ts
        else if( ret.timestamp_ns != line_ts_ns ) { break; } // ready to emit block
        if( max_parts && (max_parts != parts.size() ) ) {
          // FIXME: print line # of error. also, could optionally allow ...
          rt_err( strprintf( "row with different # of fields: on first line there were %s fields, but on this line parts.size()=%s\n",
                             str(max_parts).c_str(), str(parts.size()).c_str() ) );
        }
        max_eq( max_parts, (uint32_t)parts.size() );
        ts_lines_parts.push_back( parts );
      }
      assert_st( !ts_lines_parts.empty() ); // since we would have returned early if we started at eof
      assert_st( max_parts ); // since we would have returned early if we started at eof
      if( max_parts < 2 ) { rt_err( "unhandled: csv file with no non-timestamp fields (i.e. 0/no data fields per timestamp)" ); }
      ret.nda = make_shared<nda_t>( dims_t{ vect_uint32_t{(uint32_t)ts_lines_parts.size(),max_parts - 1}, {"obj","attr"}, "float" } );
      float * data = nda_rp_elems<float>( ret.nda );
      for( uint32_t i = 0; i != ts_lines_parts.size(); ++i ) {
        vect_string const & parts = ts_lines_parts[i];
        assert_st( parts.size() <= max_parts );
        uint32_t fix = 0; // field ix
        for( uint32_t p = 0; p != parts.size(); ++p ) {
          if( p == timestamp_fix ) { continue; }
          data[ret.nda->dims.chk_ix2( i, fix )] = lc_str_d( parts[p] );
          ++fix;
        }        
        //printf( "parts=%s\n", str(parts).c_str() );
      }
      data_stream_file_block_done_hook( ret );
      return ret;
    }

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      data_stream_file_t::data_stream_init( nia );
      data_block_t header;
      for( uint32_t i = 0; i != skip_header_lines; ++i ) {
        string const line = mfsr.read_line_as_string();
        printf( "  csv stream header line:%s", line.c_str() );
      }
    }
  };

  

  typedef vector< data_block_t > vect_data_block_t; 
  typedef vector< vect_data_block_t > vect_vect_data_block_t;


  uint64_t ts_delta( data_block_t const & a, data_block_t const & b ) {
    return ( a.timestamp_ns > b.timestamp_ns ) ? ( a.timestamp_ns - b.timestamp_ns ) : ( b.timestamp_ns - a.timestamp_ns );
  }

  struct data_stream_merge_t : virtual public nesi, public data_stream_t // NESI(
                               // help="take N data streams and output one block across all streams for each stream-block read.",
                               // bases=["data_stream_t"], type_id="merge")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    vect_p_data_stream_t streams; //NESI(help="input data streams")

    virtual bool seek_to_block( uint64_t const & frame_ix ) {
      bool had_true = 0;
      for( uint32_t i = 0; i != streams.size(); ++i ) {
        bool const res = streams[i]->seek_to_block( frame_ix );
        if( !res ) {
          if( !had_true ) { return false; } // failed on first sub-stream (i.e. all prior failed), so okay to return failure
          rt_err( "merge stream seek failed on non-first substream index " + str(i) );
        }
        had_true |= res;
      }
      return true;
    }

    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      for( uint32_t i = 0; i != streams.size(); ++i ) {  streams[i]->data_stream_init( nia );  }
    }

    virtual void set_opt( data_stream_opt_t const & opt ) { for( uint32_t i = 0; i != streams.size(); ++i ) { streams[i]->set_opt( opt ); } }

    virtual string get_pos_info_str( void ) {
      string ret = "\n";
      for( uint32_t i = 0; i != streams.size(); ++i ) {
        ret += "  " + str(i) + ": " + streams[i]->get_pos_info_str() + "\n";
      }
      return ret;
    }

    // we keep producing blocks until *all* streams are invalid, then we ret an invalid block
    virtual data_block_t proc_block( data_block_t const & db ) {
      if( db.subblocks ) {
        if( db.num_subblocks() != streams.size() ) {
          rt_err( strprintf( "data_stream_merge: input data block must either have no subblocks (distribute-clone case), or have db.num_subblocks()=%s be equal to streams.size()=%s (which it is not).\n", str(db.num_subblocks()).c_str(), str(streams.size()).c_str() ) );
        }
      }
      data_block_t ret = db;
      ret.subblocks = make_shared<vect_data_block_t>(streams.size());
      bool has_valid_subblock = 0;
      for( uint32_t i = 0; i != streams.size(); ++i ) {
        ret.subblocks->at(i) = streams[i]->proc_block(db.subblocks ? db.subblocks->at(i) : db );
        if( ret.subblocks->at(i).valid() ) { has_valid_subblock = 1; }
      }
      if( !has_valid_subblock ) { ret.subblocks.reset(); }
      return ret;
    }
  };

  struct data_stream_fold_t : virtual public nesi, public data_stream_t // NESI(
                               // help="take a indexed subblock of a stream and move it to be a subsubblock of another indexed subblock.",
                               // bases=["data_stream_t"], type_id="fold")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t fold_targ; //NESI(default="0",help="fold target")
    uint32_t fold_src; //NESI(default="0",help="fold src")

    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      if( fold_targ == fold_src ) { rt_err( strprintf( "fold_targ=%s and fold_src=%s must be different\n", str(fold_targ).c_str(), str(fold_src).c_str() ) ); }
    }
    
    virtual string get_pos_info_str( void ) { return string( "fold: <no-state>" ); }

    void check_has_subblock( data_block_t const & db, uint32_t const & sbix, string const & tag ) {
      if( !( sbix < db.num_subblocks() ) ) {
        rt_err( strprintf( "data_stream_fold: can't extract subblock for '%s' with index %s from input data block with db.num_subblocks()=%s.\n",
                           tag.c_str(), str(sbix).c_str(), str(db.num_subblocks()).c_str() ) );
      }
    }

    virtual data_block_t proc_block( data_block_t const & db ) {
      if( !db.has_subblocks() ) { rt_err( "data_stream_fold: input data block has no subblocks" ); }
      check_has_subblock( db, fold_src, "fold_src" );
      check_has_subblock( db, fold_targ, "fold_targ" );
      assert_st( db.num_subblocks() > 1 );
      data_block_t ret = db.clone();
      data_block_t fold_src_db = ret.subblocks->at( fold_src );
      ret.subblocks->erase( ret.subblocks->begin() + fold_src );
      data_block_t & fold_targ_db = ret.subblocks->at( fold_targ );
      fold_targ_db.ensure_has_subblocks();
      fold_targ_db.subblocks->push_back( fold_src_db );
      return ret;
    }
  };

  
  struct data_stream_sync_t : virtual public nesi, public data_stream_t // NESI(
                              // help="take N data streams, with one as primary, and output one block across all streams for each primary stream block, choosing the nearest-by-timestamp-to-the-primary-block-timestamp-block for each non-primary stream. ",
                                    // bases=["data_stream_t"], type_id="sync")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    uint32_t sync_verbose; //NESI(default="0",help="sync-local verbosity level (max 99) -- if non-zero, locally overrides verbosity")
    vect_p_data_stream_t streams; //NESI(help="input data streams")
    uint64_t max_delta_ns; //NESI(default="0",help="if non-zero, refuse to emit a primary block if, for any secondary stream, no block with a timestamp <= max_detla_ns from the primary block can be found (i.e. all secondary streams must have a 'current' block).")
    uint32_t psix; //NESI(default="0",help="primary stream index (0 based)")

    uint64_t seek_buf_size; //NESI(default="4096",help="max depth of seek buffer (i.e. keep this many old frames for seeking)")
    uint64_t seek_buf_pos;
    deque_data_block_t seek_buf;

    uint64_t next_frame_ix;
    vect_vect_data_block_t cur_dbs;

    virtual void set_opt( data_stream_opt_t const & opt ) { for( uint32_t i = 0; i != streams.size(); ++i ) { streams[i]->set_opt( opt ); } }

    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      if( sync_verbose ) { verbose = sync_verbose; }
      next_frame_ix = 0;
      if( !( psix < streams.size() ) ) { rt_err( strprintf( "psix=%s must be < streams.size()=%s\n",
                                                           str(psix).c_str(), str(streams.size()).c_str() ) ); }
      seek_buf_pos = 0;
      for( uint32_t i = 0; i != streams.size(); ++i ) { streams[i]->data_stream_init( nia ); }
      cur_dbs.resize( streams.size() );
      for( uint32_t i = 0; i != streams.size(); ++i ) {
        if( i == psix ) { continue; }
        cur_dbs[i].push_back( streams[i]->proc_block(data_block_t()) );
        if( !cur_dbs[i][0].valid() ) { rt_err( strprintf( "no blocks at all in stream i=%s\n", str(i).c_str() ) ); }
        cur_dbs[i].push_back( streams[i]->proc_block(data_block_t()) );
      }
    }
    
    virtual string get_pos_info_str( void ) {
      string ret = "\n";
      for( uint32_t i = 0; i != streams.size(); ++i ) {
        ret += "  " + str(i) + ": " + streams[i]->get_pos_info_str() + "\n";
      }
      return ret;
    }

   virtual bool seek_to_block( uint64_t const & frame_ix ) {
     for( uint32_t i = 0; i != seek_buf.size() ; ++i ) {
       if( seek_buf[i].frame_ix == frame_ix ) { seek_buf_pos = i; return true; }
     }
     return false; // note: state/pos is unchanged on failure
   }

    
    virtual data_block_t proc_block( data_block_t const & db ) {
      assert_st( seek_buf_pos <= seek_buf.size() );
      if( seek_buf_pos < seek_buf.size() ) { // fill request from seek_buf if possible
        ++seek_buf_pos;
        return seek_buf[ seek_buf_pos - 1 ];
      }
      
      while ( 1 ) {
        data_block_t pdb = streams[psix]->proc_block(db);
        data_block_t ret;
        if( !pdb.valid() ) { return ret; } // done
        ret.subblocks = make_shared<vect_data_block_t>(streams.size());
        if( verbose ) { printf( "-- psix=%s pdb.timestamp=%s\n", str(psix).c_str(), str(pdb.timestamp_ns).c_str() ); }
        bool ret_valid = 1;
        for( uint32_t i = 0; i != streams.size(); ++i ) {
          if( i == psix ) { continue; }
          vect_data_block_t & i_dbs = cur_dbs[i];
          assert( i_dbs.size() == 2 ); // always 2 entries, but note that head may be invalid/end-of-stream
          while( i_dbs[1].valid() && ( i_dbs[1].timestamp_ns < pdb.timestamp_ns ) ) {
            i_dbs[0] = i_dbs[1];
            i_dbs[1] = streams[i]->proc_block(data_block_t());
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
            ret.subblocks->at(i) = sdb;
          }
        }
        if( ret_valid ) {
          ret.timestamp_ns = pdb.timestamp_ns;
          ret.frame_ix = next_frame_ix;
          ++next_frame_ix;
          ret.subblocks->at(psix) = pdb;
          assert_st( seek_buf.size() == seek_buf_pos ); // if we're reading a block, we should be at end of seek_buf
          if( seek_buf_size ) { // otherwise, we'd try to pop_front() on a empty buf to make room ...
            assert_st( seek_buf.size() <= seek_buf_size );
            if( seek_buf.size() == seek_buf_size ) { seek_buf.pop_front(); } // if buf was full, free a slot ...
            else { ++seek_buf_pos; } // ... else it will grow, so update pos.
            seek_buf.push_back( ret );
          }
          return ret;
        }
        // else continue
      }
    }
  };


  struct data_stream_pipe_t : virtual public nesi, public data_stream_t // NESI(
                               // help="take N data streams, connect them into a pipeline, and expose the result as a data stream.",
                               // bases=["data_stream_t"], type_id="pipe")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    vect_p_data_stream_t pipe; //NESI(help="streams to connect into single pipeline")
    uint32_t flush_mode; //NESI(default="0",help="if non-zero, for each input, run pipe until no stage has more out. only makes sense if last pipe stage is a sink. not pretty/nice!")

    vect_uint8_t have_more_out;
    
    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      for( uint32_t i = 0; i != pipe.size(); ++i ) {  pipe[i]->data_stream_init( nia );  }
      have_more_out.resize( pipe.size(), 0 );
    }
    virtual void set_opt( data_stream_opt_t const & opt ) { for( uint32_t i = 0; i != pipe.size(); ++i ) { pipe[i]->set_opt( opt ); } }

    virtual string get_pos_info_str( void ) {
      string ret = "\n";
      for( uint32_t i = 0; i != pipe.size(); ++i ) {
        ret += "  pipe stage " + str(i) + ": " + pipe[i]->get_pos_info_str() + "\n";
      }
      return ret;
    }

    virtual bool seek_to_block( uint64_t const & frame_ix ) {
      bool ret = pipe[0]->seek_to_block( frame_ix );
      if( ret ) {
        // if we seeked, abort if have_more_out is set. we could just fail in that case, but it seems likely to be
        // confusing. bear in mind that, currently, seeking is really a hack at best. at some point, we might need to
        // reconcile the desire for multi-rate with the desire for seeking, which seem to be pretty incompatible? maybe
        // we need more explicity multi-rate vs single-rate sections?
        for( uint32_t i = 0; i != pipe.size(); ++i ) {
          if( have_more_out[i] ) { rt_err( "unimplemented: can't seek on pipe when some stage has set have_more_info. could add flush functionality to fix ..." ); }
        }
      }
      return ret;
    }

    // return 0 is no stage has more out, other returns highest-stage-ix-with-more-out + 1 
    uint32_t get_pipe_start_stage( void ) {
      uint32_t pipe_start_stage = pipe.size();
      while( pipe_start_stage ) {
        if( have_more_out[pipe_start_stage - 1] ) { break; }
        --pipe_start_stage;
      }
      return pipe_start_stage;
    }
    
    virtual data_block_t proc_block( data_block_t const & db ) {
      // for now, the have_more_out handling assumes this is a default/template block, not real data, and will discard
      // it when there are stages in the pipeline where have_more_out was set. obviously this is a WIP/Hack. sigh, maybe
      // we need the whole graph-with-per-block-flow/firing ...
      data_block_t ret = db;

      while( 1 ) {
        bool fire_again = 0;
        uint32_t pipe_start_stage = get_pipe_start_stage();
        if( pipe_start_stage ) {
          // some stage has data, so start with it
          pipe_start_stage -= 1; // note: after this, starting at stage 0 due to have_more_out or because no stage set have_more_out can't be distinguished
          ret = data_block_t(); // FIXME(sorta): discard input block
        }
        // note: all stages after pipe_start_stage have have_more_out = 0 
      
        for( uint32_t i = pipe_start_stage ; i != pipe.size(); ++i ) {
          ret = pipe[i]->proc_block( ret );
          //printf( "i=%s ret.need_more_in=%s ret.have_more_out=%s\n", str(i).c_str(), str(ret.need_more_in).c_str(), str(ret.have_more_out).c_str() );
          if( ret.need_more_in ) {
            assert_st( !ret.have_more_out );
            assert_st( !have_more_out[i] ); // shouldn't have gotten here otherwise?
            if( i == 0 ) { rt_err( "unhandled: need_more_in set for first pipeline stage" ); }
            fire_again = 1;
            ret = data_block_t(); // discard need_more_in block, replace with default/null block
            break; // do another firing of the pipeline, without emiting anything ...
          }
          have_more_out[i] = ret.have_more_out;
          ret.have_more_out = 0; // clear out flag before we pass to next stage. also clears before return, which is FIXME-ish ...
          if( !ret.valid() ) {
            // terminate pipe on end-of-stream at any point after first stage, and return end-of-stream
            assert_st( !fire_again );
            break;
          } 
        }
        if( !fire_again ) {
          // real end-of-stream, or normal ran-all-stages
          if( flush_mode && get_pipe_start_stage() ) { continue; }
          break;
        } 
      }
      return ret;
    }
  };

  struct data_stream_crop_t : virtual public nesi, public data_stream_t // NESI(
                               // help="crop input nda in X/Y dims. (note: only Y dim supported currently)",
                               // bases=["data_stream_t"], type_id="crop")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support    
    u32_box_t crop; //NESI(default="0:0:0:0",help="crop pels from sides")
    virtual void data_stream_init( nesi_init_arg_t * const nia ) { }
    virtual string get_pos_info_str( void ) { return strprintf( "data_sink_crop: crop=%s <no-state>", str(crop).c_str() ); }
    virtual data_block_t proc_block( data_block_t const & db ) {
      data_block_t ret = db;
      dims_t const & in_dims = db.nda->dims;
      if( crop.p[0].d[0] || crop.p[1].d[0] ) { rt_err( "TODO: non-zero x-coord crop for to_nda (needs copy and/or padded/strided nda support)" ); }
      if( (!in_dims.valid()) || (in_dims.size() == 0) ) {
        rt_err( "can only crop data blocks with valid, non-scalar dims. dims were:" + str(in_dims) );
      }
      if( in_dims.names(0) != "y" ) {
        rt_err( "can only crop data blocks with first dim of 'y'; dims were:" + str(in_dims) );
      }
      // use aliasing constructor to get offset view of this data block
      uint8_t * const cropped_rp_data = (uint8_t *)db.nda->rp_elems() + crop.p[0].d[1]*in_dims.tsz()*in_dims.strides(0);
      p_uint8_t cropped_p_data( db.nda->get_internal_data(), cropped_rp_data );
      // make new dims by shrinking y size
      dims_t cropped_dims = in_dims.clone();
      cropped_dims.dims(0) -= crop.bnds_sum().d[1];
      cropped_dims.calc_strides(); // recalc since modified
      ret.nda = make_shared<nda_t>( cropped_dims, cropped_p_data );
      return ret;
    }
  };

  // normalize angles to the range [-180,180) (after adding an adjustment adj)
  struct nda_adj_and_normalize_angle_in_degrees_t {
    double adj;
    bool negate;
    nda_adj_and_normalize_angle_in_degrees_t( double const & adj_, bool const & negate_  ) :  adj(adj_), negate(negate_) {}
    template< typename T > void op( nda_t & nda ) const {
      ndat_info_t const & ni = get_ndat<T>();
      assert_st( ni.is_signed ); // not sensible otherwise?
      T * const elems = static_cast<T *>(nda.rp_elems());
      uint64_t const num_elems = nda.elems_sz();
      assert_st( elems );
      for( uint64_t i = 0; i != num_elems; ++i ) {
        double ret = fmod( ( negate ? (-double(elems[i])) : double(elems[i]) )+ adj + 180.0, 360.0 );
        ret += (ret<0) ? 180.0 : (-180.0);
        //if( (i % 1000) == 0 ) { printf( "PRE  elems[i]=%s    (adj=%s)\n", str(elems[i]).c_str(), str(adj).c_str() ); }
        if( ni.is_float ) { elems[i] = ret; }
        else { elems[i] = lround(ret); }
        //if( (i % 1000) == 0 ) { printf( "POST elems[i]=%s\n\n", str(elems[i]).c_str() ); }
      }
    }
  };

  
  struct data_stream_adj_angle_t : virtual public nesi, public data_stream_t // NESI(
                                   // help="normalize angle data in degrees to range [-180,180) (with optional added offset)",
                                   // bases=["data_stream_t"], type_id="adj-angle")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support    
    double adj; //NESI(default="0.0",help="adj amount (degress)")
    uint32_t negate; //NESI(default="0",help="if non-zero, negate input angle")
    virtual void data_stream_init( nesi_init_arg_t * const nia ) { }
    virtual string get_pos_info_str( void ) { return strprintf( "adj-angle: adj=%s <no-state>", str(adj).c_str() ); }
    virtual data_block_t proc_block( data_block_t const & db ) {
      data_block_t ret = db.clone();
      if( !ret.nda ) {
        rt_err( "can only adj-angle on data blocks with nda data, but nda was null in input db." );
      }
      ret.nda = ret.nda->clone(); // deep copy nda before we modify it
      nda_dispatch( *ret.nda, nda_adj_and_normalize_angle_in_degrees_t( adj, negate ) );      
      return ret;
    }
  };

  
  struct data_stream_pass_t : virtual public nesi, public data_stream_t // NESI(
                               // help="indentity (i.e. do-nothing) xform stream",
                               // bases=["data_stream_t"], type_id="pass")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support    
    virtual void data_stream_init( nesi_init_arg_t * const nia ) { }
    virtual string get_pos_info_str( void ) { return string( "pass: does nothing, <no-state>" ); }
    virtual data_block_t proc_block( data_block_t const & db ) { return db; }
  };

  
  struct scan_data_stream_t : virtual public nesi, public has_main_t // NESI(
                                    // help="scan data stream ",
                                    // bases=["has_main_t"], type_id="scan-data-stream")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support    
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    uint32_t timestamp_check; //NESI(default="1",help="if non-zero, check for increasing timestamp and print warnings if not seen.")
    uint64_t num_to_proc; //NESI(default=0,help="read/write this many records; zero for unlimited")
    p_data_stream_t stream; //NESI(req=1,help="input data stream")

    uint64_t tot_num_proc;
    
    void main( nesi_init_arg_t * nia ) {
      tot_num_proc = 0;
      stream->data_stream_init( nia );
      uint64_t last_ts = 0;
      while( 1 ) {
        if( num_to_proc && (tot_num_proc == num_to_proc) ) { break; } // proc'd req'd # of blocks --> done
        data_block_t db = stream->proc_block(data_block_t());
        if( !db.valid() ) { break; } // not more data --> done
        if( db.valid() && (verbose || (tot_num_proc == 0) ) ) {
          printf( "-- stream: db=%s @ %s\n", db.info_str().c_str(), stream->get_pos_info_str().c_str() );
        }
        // FIXME: make this recursive wrt subblocks or the like?
        if( timestamp_check && db.nda.get() ) { // if db has data, check timestamp (old non-multi-stream-scan functionality)
          if( db.timestamp_ns <= last_ts ) {
            printf( "**ERROR: ts did not increase: db.timestamp_ns=%s last_ts=%s stream->get_pos_info_str()=%s\n",
                    str(db.timestamp_ns).c_str(), str(last_ts).c_str(), str(stream->get_pos_info_str()).c_str() );
          }
          last_ts = db.timestamp_ns;
        }
        ++tot_num_proc;
      }
    }
  };

#include"gen/data-stream.H.nesi_gen.cc"
#include"gen/data-stream.cc.nesi_gen.cc"

}
