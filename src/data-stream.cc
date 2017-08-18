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

  std::ostream & operator << ( std::ostream & out, data_block_t const & db ) { out << db.info_str(); return out; }
  string data_block_t::info_str( void ) const {
    string ret;
    if( d.get() ) {
      ret += strprintf( "sz=%s frame_ix=%s timestamp_ns=%s",
                        str(sz).c_str(), str(frame_ix).c_str(), str(timestamp_ns).c_str() );
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
      if( !mfsr.can_read( sizeof( block_sz ) ) ) { return ret; } // not enough bytes left for another block. FIXME: should be an error?
      mfsr.read_val( block_sz );
      if( block_sz == uint32_t_const_max ) { return ret; } // end of dumpvideo stream marker
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


  struct laser_info_t {
    uint16_t distance;
    uint8_t intensity;
  } __attribute__((packed));

  struct block_info_t {
    uint16_t block_id;
    uint16_t rot_pos;
    laser_info_t lis[];
  } __attribute__((packed));
  struct status_info_t {
    uint32_t gps_timestamp_us;
    uint8_t status_type;
    uint8_t status_val;
  } __attribute__((packed));
  

  uint16_t const ang_max = 36000;
  uint16_t const half_ang_max = 18000;
  // for angles in the space [0,ang_max), the distance between two angles is always <= 180.0
  // here, we give a delta such that delta = a1 - a2  or a2 + delta = a1 , with delta in [-18000,18000)
  int16_t rel_angle_delta( uint16_t const a1, uint16_t const & a2 ) {
    assert_st( a1 < ang_max );
    assert_st( a2 < ang_max );
    int32_t delta = int32_t(a1) - int32_t(a2);
    if( delta >= half_ang_max ) { delta -= ang_max; }
    if( delta < -half_ang_max ) { delta += ang_max; }
    return delta;
  }

  // we can order two angles by saying (a1 < a2) if thier rel_angle_delta(a1,a2) < 0 
  bool rel_angle_lt( uint16_t const a1, uint16_t const & a2 ) { return rel_angle_delta(a1,a2) < 0; }
  
  struct data_stream_velodyne_t : virtual public nesi, public data_stream_t // NESI(help="parse data stream (velodyne) into per-full-revolution data blocks by merging across packets",
                             // bases=["data_stream_t"], type_id="velodyne")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    p_data_stream_t vps; //NESI(req=1,help="underlying velodyne packet stream")

    uint32_t fbs_per_packet; //NESI(default="12",help="firing blocks per packet")
    uint32_t beams_per_fb; //NESI(default="32",help="beams per firing block")
    uint32_t status_bytes; //NESI(default="6",help="bytes of status at end of block")

    double fov_center; //NESI(default=0.0,help="center of FoV to sample in degrees. frames will be split at (VAL + 180) degrees.")
    uint32_t fov_rot_samps; //NESI(default="384",help="number of samples-in-rotation to extract around fov_center")

    uint32_t tot_lasers; //NESI(default="64",help="total number of lasers. must be either 32 (one block) or 64 laser (two block) scanner.")
    uint32_t dual_return_and_use_only_first_return; //NESI(default="1",help="if 1, assume dual return mode, and use only first return.")
    string laser_to_row_ix_str; //NESI(req=1,help="':'-seperated list of 0-based dense-matrix-row values to which to map each laser id to. should have tot_lasers elements, and should be a permutation of [0,tot_lasers).") 

    vect_uint32_t laser_to_row_ix;
    uint32_t fb_sz;
    uint32_t packet_sz;
    uint16_t last_rot;
    uint16_t fov_center_rot;
    uint16_t split_rot;
    dims_t out_dims;
    // internal state:
    uint64_t tot_num_read; // num blocks read so far
    p_nda_uint16_t buf_nda;
    uint32_t buf_nda_rot;
    uint32_t rots_till_emit;
    
    virtual string get_pos_info_str( void ) { return strprintf( "tot_num_read=%s vps info:%s",
                                                                str(tot_num_read).c_str(), vps->get_pos_info_str().c_str() ); }

    virtual data_block_t read_next_block( void ) {
      p_nda_uint16_t out_nda = make_shared<nda_uint16_t>( out_dims );
      data_block_t ret_db;
      data_block_t db;
      uint16_t last_ub_rot = uint16_t_const_max;
      while( !ret_db.valid() ) {
        db = vps->read_next_block();
        if( !db.d.get() ) { return db; } // not enough data for another frame, give up
        if( db.sz != packet_sz ) { rt_err(
            strprintf( "lidar decode expected packet_sz=%s but got block with dv.sz=%s",
                       str(packet_sz).c_str(), str(db.sz).c_str() ) ); }
        if( verbose > 10 ) { printf( "data_stream_velodyne: %s\n", str(db).c_str() ); }
        status_info_t const * si = (status_info_t *)(db.d.get()+fb_sz*fbs_per_packet);
        if( verbose > 10 ) { printf( "  packet: si->gps_timestamp_us=%s si->status_type=%s si->status_val=%s\n",
                                     str(si->gps_timestamp_us).c_str(), str(si->status_type).c_str(), str(uint16_t(si->status_val)).c_str() ); }
        for( uint32_t fbix = 0; fbix != fbs_per_packet; ++fbix ) {
          block_info_t const * bi = (block_info_t *)(db.d.get()+fb_sz*fbix);
          uint32_t laser_id_base = 0;
          if( tot_lasers == 64 ) {
            if( bi->block_id != ( (fbix&1) ? 0xddff : 0xeeff ) ) {
              rt_err( strprintf( "(64 laser mode) saw unexpected bi->block_id=%s for firing block fbix=%s\n",
                                 str(bi->block_id).c_str(), str(fbix).c_str() ) );
            }
            if( fbix&1 ) { laser_id_base = 32; }
          } else if( tot_lasers == 32 ) {            
            assert_st( 0 ); // not implmented yet, should just check for 0xeeff?
          } else { assert_st( 0 ); }

          if( dual_return_and_use_only_first_return ) {
            if( fbix&2 ) { // skip second return blocks, but check that they are the same rot
              if( bi->rot_pos != last_rot ) {
                rt_err( strprintf( "error skipping second return block: expected bi->rot_pos=%s to equal processed block rot last_rot=%s."
                                   " refusing to proceed.",
                                   str(bi->rot_pos).c_str(), str(last_rot).c_str() ) );
              }
              continue; // if no error, we're good to skip.
            } 
          }
          
          if( verbose > 50 ) {
            printf( "fbix=%s laser_id_base=%s bi.block_id=%hx bi.rot_pos=%hu\n",
                    str(fbix).c_str(), str(laser_id_base).c_str(), bi->block_id, bi->rot_pos );
            for( uint32_t i = 0; i != beams_per_fb; ++i ) {
              printf( " %s", str(bi->lis[i].distance).c_str() );
            }
            printf("\n");
          }
          for( uint32_t i = 0; i != beams_per_fb; ++i ) {
            uint32_t const rix = laser_to_row_ix.at(laser_id_base+i);
            buf_nda->at2( rix, buf_nda_rot ) = bi->lis[i].distance;
          }
          if( (tot_lasers == 64) && (!(fbix&1)) ) { last_ub_rot = bi->rot_pos; continue; } // FIXME: handle upper/lower more cleanly
          else {
            if( bi->rot_pos != last_ub_rot ) {
              rt_err( strprintf( "error on second block for 64 laser sensor: expected bi->rot_pos=%s to equal last_ub_rot=%s."
                                 " refusing to proceed.",
                                 str(bi->rot_pos).c_str(), str(last_ub_rot).c_str() ) );
              
            }
          }
            
          if( rots_till_emit == uint32_t_const_max ) { // if not triggered yet
            if( (last_rot != uint16_t_const_max) &&
                rel_angle_lt(last_rot,fov_center_rot) && !rel_angle_lt(bi->rot_pos,fov_center_rot) ) { // trigger
              ret_db.timestamp_ns = db.timestamp_ns;
              if( verbose ) { printf( "-- TRIGGER -- bi->rot_pos=%s\n", str(bi->rot_pos).c_str() ); }
              if( verbose ) { printf( "  @TRIGGER: si->gps_timestamp_us=%s si->status_type=%s si->status_val=%s\n",
                                      str(si->gps_timestamp_us).c_str(), str(si->status_type).c_str(), str(uint16_t(si->status_val)).c_str() ); }
              rots_till_emit = fov_rot_samps >> 1; // have 1/2 of fov samps in buf, need other 1/2 now
            } 
          } else {
            assert_st( rots_till_emit );
            --rots_till_emit;
            if( !rots_till_emit ) {
              // done, copy rotated buf_nda into out_nda and emit buf as FoV
              for( uint32_t j = 0; j != tot_lasers; ++j ) {
                for( uint32_t i = 0; i != fov_rot_samps; ++i ) {
                  uint32_t const buf_rot = (i+buf_nda_rot+1)%fov_rot_samps;
                  out_nda->at2( j, i ) = buf_nda->at2( j, buf_rot );
                }
              }
              ret_db.nda_dims = out_dims;
              ret_db.d = out_nda->get_internal_data();
              ret_db.sz = out_dims.bytes_sz();
              rots_till_emit = uint32_t_const_max; // back to untriggered state
            } 
          }
          if( verbose > 50 ) printf( "last_rot=%s bi->rot_pos=%s\n", str(last_rot).c_str(), str(bi->rot_pos).c_str() );
          last_rot = bi->rot_pos;
          buf_nda_rot += 1; if( buf_nda_rot == fov_rot_samps ) { buf_nda_rot = 0; } // FIXME: don't we have inc_mod for this?
        }
      }
      ret_db.frame_ix = tot_num_read;
      ++tot_num_read;
      if( verbose ) { printf( "velodyne ret_db: %s\n", str(ret_db).c_str() ); }
      return ret_db;
    }

    // init/setup

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      if( !(tot_lasers == 64) ) { rt_err( "non-64 laser mode not implemented" ); }
      if( !dual_return_and_use_only_first_return ) { rt_err( "non-dual return mode not implemented" ); }
      if( !(fov_rot_samps >= 2) ) { rt_err( "fov_rot_samps must be >= 2" ); }
      printf( "data_stream_init(): mode=%s\n", str(mode).c_str() );
      tot_num_read = 0;
      // setup internal state
      fb_sz = sizeof( block_info_t ) + beams_per_fb * sizeof( laser_info_t );
      packet_sz = fbs_per_packet * fb_sz + status_bytes;
      last_rot = uint16_t_const_max;
      if( (fov_center < 0.0) || (fov_center >= 360.0) ) { rt_err( strprintf( "fov_center must be in [0.0,360.0) but was =%s",
                                                                             str(fov_center).c_str() ) ); }
      fov_center_rot = uint16_t( lround( fov_center * 100 ) );

      // FIXME: split_rot unused/unneeded?
      double split_rot_deg = fov_center + 180.0;
      while( split_rot_deg >= 360.0 ) { split_rot_deg -= 360.0; }
      assert_st( (split_rot_deg >= 0.0) && (split_rot_deg < 360.0) );
      split_rot = uint16_t( lround( split_rot_deg * 100 ) );

      out_dims = dims_t{ dims_t{ { tot_lasers, fov_rot_samps }, {"y","x"}, "uint16_t" }};
      buf_nda = make_shared<nda_uint16_t>( out_dims );
      buf_nda_rot = 0;
      rots_till_emit = uint32_t_const_max; // start in untriggered state
      
      vect_string laser_to_row_ix_str_parts = split(laser_to_row_ix_str,':');
      if( laser_to_row_ix_str_parts.size() != tot_lasers ) {
        rt_err( strprintf( "expected tot_lasers=%s ':' seperated indexes in laser_to_row_ix_str=%s, but got laser_to_row_ix_str_parts.size()=%s\n",
                           str(tot_lasers).c_str(), str(laser_to_row_ix_str).c_str(), str(laser_to_row_ix_str_parts.size()).c_str() ) );
      }
      for( uint32_t i = 0; i != tot_lasers; ++i ) {
        try {  laser_to_row_ix.push_back( lc_str_u32( laser_to_row_ix_str_parts[i] ) ); }
        catch( rt_exception & rte ) { rte.err_msg = "parsing element " + str(i) + " of laser_to_row_ix_str: " + rte.err_msg; throw; }
      }
      vect_uint32_t laser_to_row_ix_sorted = laser_to_row_ix;
      sort( laser_to_row_ix_sorted.begin(), laser_to_row_ix_sorted.end() );
      assert_st( laser_to_row_ix_sorted.size() == tot_lasers );
      for( uint32_t i = 0; i != tot_lasers; ++i ) {
        if( laser_to_row_ix_sorted[i] != i ) { rt_err( "the elements of laser_to_row_ix_sorted are not a permutation of [0,tot_lasers)" ); }
      }
      vps->data_stream_init( nia );
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
    vect_p_data_stream_t stream; //NESI(help="input data streams")

    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      for( uint32_t i = 0; i != stream.size(); ++i ) {  stream[i]->data_stream_init( nia );  }
    }    
    virtual string get_pos_info_str( void ) {
      string ret = "\n";
      for( uint32_t i = 0; i != stream.size(); ++i ) {
        ret += "  " + str(i) + ": " + stream[i]->get_pos_info_str() + "\n";
      }
      return ret;
    }

    // we keep producing blocks until *all* streams are invalid, then we ret an invalid block
    virtual data_block_t read_next_block( void ) {
      data_block_t ret;
      ret.subblocks = make_shared<vect_data_block_t>(stream.size());
      bool has_valid_subblock = 0;
      for( uint32_t i = 0; i != stream.size(); ++i ) {
        ret.subblocks->at(i) = stream[i]->read_next_block();
        if( ret.subblocks->at(i).valid() ) { has_valid_subblock = 1; }
      }
      if( !has_valid_subblock ) { ret.subblocks.reset(); }
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
    vect_p_data_stream_t stream; //NESI(help="input data streams")
    uint64_t max_delta_ns; //NESI(default="0",help="if non-zero, refuse to emit a primary block if, for any secondary stream, no block with a timestamp <= max_detla_ns from the primary block can be found (i.e. all secondary streams must have a 'current' block).")
    uint32_t psix; //NESI(default="0",help="primary stream index (0 based)")

    vect_vect_data_block_t cur_dbs;

    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      if( sync_verbose ) { verbose = sync_verbose; }
      if( !( psix < stream.size() ) ) { rt_err( strprintf( "psix=%s must be < stream.size()=%s\n",
                                                           str(psix).c_str(), str(stream.size()).c_str() ) ); }
      for( uint32_t i = 0; i != stream.size(); ++i ) { stream[i]->data_stream_init( nia ); }
      cur_dbs.resize( stream.size() );
      for( uint32_t i = 0; i != stream.size(); ++i ) {
        if( i == psix ) { continue; }
        cur_dbs[i].push_back( stream[i]->read_next_block() );
        if( !cur_dbs[i][0].valid() ) { rt_err( strprintf( "no blocks at all in stream i=%s\n", str(i).c_str() ) ); }
        cur_dbs[i].push_back( stream[i]->read_next_block() );
      }
    }
    
    virtual string get_pos_info_str( void ) {
      string ret = "\n";
      for( uint32_t i = 0; i != stream.size(); ++i ) {
        ret += "  " + str(i) + ": " + stream[i]->get_pos_info_str() + "\n";
      }
      return ret;
    }

    virtual data_block_t read_next_block( void ) {
      while ( 1 ) {
        data_block_t pdb = stream[psix]->read_next_block();
        data_block_t ret;
        if( !pdb.valid() ) { return ret; } // done
        ret.subblocks = make_shared<vect_data_block_t>(stream.size());
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
            ret.subblocks->at(i) = sdb;
          }
        }
        if( ret_valid ) {
          ret.subblocks->at(psix) = pdb;
          return ret;
        }
        // else continue
      }
    }
  };
  
  struct scan_data_stream_t : virtual public nesi, public has_main_t // NESI(
                                    // help="scan data stream ",
                                    // bases=["has_main_t"], type_id="scan-data-stream")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support    
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    uint64_t num_to_proc; //NESI(default=0,help="read/write this many records; zero for unlimited")
    uint32_t full_dump_ix; //NESI(default="-1",help="for this sub-stream (if it exists), dump all block sizes")
    p_data_stream_t src; //NESI(req=1,help="input data stream")
    p_data_sink_t sink; //NESI(help="output data sink")

    uint64_t tot_num_proc;
    
    void main( nesi_init_arg_t * nia ) {
      tot_num_proc = 0;
      src->data_stream_init( nia );
      if( sink ) { sink->data_sink_init( nia ); }
      uint64_t last_ts = 0;
      while( 1 ) {
        if( num_to_proc && (tot_num_proc == num_to_proc) ) { break; } // proc'd req'd # of blocks --> done
        data_block_t db = src->read_next_block();
        if( !db.valid() ) { break; } // not more data --> done
        if( verbose ) {
          printf( "-- src: frame_ix=%s ts=%s sz=%s @ %s\n",
                  str(db.frame_ix).c_str(), str(db.timestamp_ns).c_str(), str(db.sz).c_str(), src->get_pos_info_str().c_str() );
        }
        // FIXME: make this recursive wrt subblocks or the like?
        if( db.d.get() ) { // if db has data, check timestamp (old non-multi-stream-scan functionality)
          if( db.timestamp_ns <= last_ts ) {
            printf( "**ERROR: ts did not increase: db.timestamp_ns=%s last_ts=%s src->get_pos_info_str()=%s\n",
                    str(db.timestamp_ns).c_str(), str(last_ts).c_str(), str(src->get_pos_info_str()).c_str() );
          }
          last_ts = db.timestamp_ns;
        }
        if( db.subblocks.get() ) { // if db has sub-blocks, maybe dump some info on them (old multi-stream-scan functionality)
          if( verbose ) { printf( "----\n" ); }
          for( uint32_t i = 0; i != db.subblocks->size(); ++i ) {
            data_block_t const & sdb = db.subblocks->at(i);
            // if block valid, and: on first block, or if verbose, or if this is the full-dump-stream-ix substream
            if( sdb.valid() && (verbose || (i == full_dump_ix) || (tot_num_proc == 0) ) ) {
              printf( "substream[%s] frame_ix=%s ts=%s sz=%s\n",
                      str(i).c_str(), str(sdb.frame_ix).c_str(), str(sdb.timestamp_ns).c_str(), str(sdb.sz).c_str() );
            }            
          }
        }
        if( sink ) { sink->consume_block( db ); }
        ++tot_num_proc;
      }
    }
  };

#include"gen/data-stream.H.nesi_gen.cc"
#include"gen/data-stream.cc.nesi_gen.cc"

}
