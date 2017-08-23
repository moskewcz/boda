// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"nesi.H"
#include"data-stream.H"
#include<algorithm>

namespace boda 
{

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
  
  uint16_t const velo_crc_poly = 0x8005;  
  uint16_t velo_crc( uint8_t const * const & d, uint32_t const len ) {
    // note: below is (mostly) hard-coded for the above poly of 0x8005
    uint16_t ret = 0;
    for( uint32_t i = 0; i != len; ++i ) {
      ret ^= d[i] << 8;
      for( uint32_t b = 8; b > 0; --b ) {
        bool const hbs = ret & 0x8000;
        ret <<= 1;
        if( hbs ) { ret ^= velo_crc_poly; }
      }
    }
    return ret;
  }

  uint32_t const velo_packets_in_cycle = 16;
  uint32_t const velo_cycles_in_epoch = 260;
  string const velo_cycle_prefix_types = "HMSDNYGTV";

  struct test_velo_crc_t : virtual public nesi, public has_main_t // NESI(help="test velodyne crc16 function impl",
                     // bases=["has_main_t"], type_id="test-velo-crc")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t out_fn; //NESI(default="%(boda_output_dir)/test-velo-crc.txt",help="output expected/computed crc and note if they are ==.")
    string hex_input; //NESI(default="0607", help="hex string of binary data to compute velodync crc16 over" )
    string hex_crc; //NESI(default="9411", help="hex string of expect crc (must be 2 bytes)" )

    void main( nesi_init_arg_t * nia ) {
      p_ostream out = ofs_open( out_fn.exp );
      string const input = unhex( hex_input );
      string expected_crc_str = unhex( hex_crc );
      if( expected_crc_str.size() != 2 ) { rt_err( strprintf( "hex_crc=%s must unhex into exactly 2 bytes\n", str(hex_crc).c_str() ) ); }
      std::swap( expected_crc_str[0], expected_crc_str[1] ); // endian swap
      uint16_t const expected_crc = *(uint16_t *)&expected_crc_str[0];
      uint16_t const calc_crc = velo_crc( (uint8_t const *)&input[0], input.size() );
      (*out) << strprintf( "calc_crc=%hx expected_crc=%hx\n", calc_crc, expected_crc );
      (*out) << strprintf( ( calc_crc == expected_crc ) ? "OK\n" : "FAIL\n" );
    }    
  };
  
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

    double fov_center; //NESI(default=0.0,help="center of FoV to sample in degrees. frames will be split at (VAL + 180) degrees.")
    uint32_t fov_rot_samps; //NESI(default="384",help="number of samples-in-rotation to extract around fov_center")

    uint32_t tot_lasers; //NESI(default="64",help="total number of lasers. must be either 32 (one block) or 64 laser (two block) scanner.")
    uint32_t enable_proc_status; //NESI(default="0",help="if non-zero, process status bytes (only present for 64 laser scanner).")

    uint32_t dual_return_and_use_only_first_return; //NESI(default="1",help="if 1, assume dual return mode, and use only first return.")
    p_string laser_to_row_ix_str; //NESI(help="':'-seperated list of 0-based dense-matrix-row values to which to map each laser id to. should have tot_lasers elements, and should be a permutation of [0,tot_lasers).") 

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
        if( enable_proc_status ) { proc_status( *si ); }
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
            if( bi->block_id != 0xeeff ) {
              rt_err( strprintf( "(32 laser mode) saw unexpected bi->block_id=%s for firing block fbix=%s\n",
                                 str(bi->block_id).c_str(), str(fbix).c_str() ) );
            }
          } else { assert_st( 0 ); }

          if( (tot_lasers == 64) && dual_return_and_use_only_first_return ) {
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
          if( tot_lasers == 64 ) {
            if( !(fbix&1) ) { last_ub_rot = bi->rot_pos; continue; } // FIXME: handle upper/lower more cleanly
            else {
              if( bi->rot_pos != last_ub_rot ) {
                rt_err( strprintf( "error on second block for 64 laser sensor: expected bi->rot_pos=%s to equal last_ub_rot=%s."
                                   " refusing to proceed.",
                                   str(bi->rot_pos).c_str(), str(last_ub_rot).c_str() ) );
                
              }
            }
          } else if (tot_lasers == 32 ) {
            // every block should be a new rot. FIXME: check this?
          } else { assert_st(0); }
            
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

    // called on each packet. assumes packets are presented in stream order.
    uint32_t last_status_ts;
    // TODO: check timestamp sequence
    // TODO: extract config data
    uint32_t cycle_in_epoch;
    uint32_t packet_in_cycle;

    vect_uint16_t cycle_types;
    vect_uint16_t cycle_vals;
    
    void on_bad_status( string const & msg ) {
      // set state to confused/unsynced state
      last_status_ts = uint32_t_const_max;
      packet_in_cycle = uint32_t_const_max; 
      cycle_in_epoch = uint32_t_const_max; // confused/unsynced state
      cycle_types.clear();
      cycle_vals.clear();
      if( msg.empty() ) { return; } // initial/expected stream reset, no error
      printf( "%s\n", str(msg).c_str() );
    }
    
    void proc_status_epoch( void ) {

    }
    
    void proc_status_cycle( void ) {
      assert_st( cycle_types.size() == velo_packets_in_cycle ); // 16 (total packets/cycle) = 9 (# prefix statuses) + 7 (per-cycle stuff)
      assert_st( cycle_vals.size() == velo_packets_in_cycle );
      if( cycle_in_epoch == uint32_t_const_max ) { // if we're confused/unsynced, just look for 0xF7 as last status type
        if( cycle_types.back() == 0xF6u ) { cycle_in_epoch = 0; }
      }
      if( cycle_in_epoch == uint32_t_const_max ) { return; } // if (still) no cycle sync, give up for now
      // process cycle
      if( cycle_in_epoch == 0 ) {
        if( verbose ) { printf( "cycle_types=%s cycle_vals=%s\n", str(cycle_types).c_str(), str(cycle_vals).c_str() ); }
      }
      ++cycle_in_epoch;
      // if epoch done, do end-of-epoch processing (checksum, capture config)
      if( cycle_in_epoch == velo_cycles_in_epoch ) {
        proc_status_epoch();
        cycle_in_epoch = 0;
        // TODO: clear epoch-related stuff
      }
      
    }

    
    void proc_status( status_info_t const & si ) {
      //printf( "si.status_type=%s si.status_val=%s\n", str(si.status_type).c_str(), str(si.status_val).c_str() );
      if( last_status_ts != uint32_t_const_max ) { // if we had a prior timestamp
        if( si.gps_timestamp_us < last_status_ts ) {
          printf( "timestamp went backwards: last_status_ts=%s si.timestamp_ns=%s\n", str(last_status_ts).c_str(), str(si.gps_timestamp_us).c_str() );
        } else {
          uint32_t ts_delta = si.gps_timestamp_us - last_status_ts;
          uint32_t max_ts_delta = (tot_lasers == 32) ? 600 : 200;
          if( ts_delta > max_ts_delta ) {
            printf( "large (>max_ts_delta=%s) ts_delta=%s\n", str(max_ts_delta).c_str(), str(ts_delta).c_str() );
          }
        }
      }
      
      last_status_ts = si.gps_timestamp_us;
      
      if( tot_lasers != 64 ) { return; } // all the remaining processing is only for 64 laser scanners ...
      if( packet_in_cycle == uint32_t_const_max ) { // if we're confused/unsynced, just look for 'H'
        if( si.status_type == 'H' ) { packet_in_cycle = 0; }
      }
      if( packet_in_cycle == uint32_t_const_max ) { return; } // if (still) no packet sync, give up for now
      // check for expected value in start of cycle
      if( packet_in_cycle < velo_cycle_prefix_types.size() ) {
        if( si.status_type != velo_cycle_prefix_types[packet_in_cycle] ) {
          on_bad_status( strprintf( "velodyne stream corrupt; at packet_in_cycle=%s, saw status type byte si.status_type=%s but expected velo_cycle_prefix_types[packet_in_cycle]=%s",
                                    str(packet_in_cycle).c_str(), str(uint32_t(si.status_type)).c_str(), str(uint32_t(velo_cycle_prefix_types[packet_in_cycle])).c_str() ) );
          return;
          
        }
        // TODO: process prefix fields (as needed)
      }
      cycle_types.push_back( si.status_type ); cycle_vals.push_back( si.status_val ); // capture type/val
      // we need to know cycle_in_epoch to do more parsing/checking. for simplicity, we defer this until the cycle is complete
      ++packet_in_cycle;
      if( packet_in_cycle == velo_packets_in_cycle ) {
        proc_status_cycle();
        cycle_types.clear();
        cycle_vals.clear();
        packet_in_cycle = 0;
      }
    }
    
    // init/setup

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      on_bad_status("");
      
      if( ! ( (tot_lasers == 64) || (tot_lasers == 32) ) ) { rt_err( "non-32/64 laser mode not implemented" ); }
      if( tot_lasers == 64 ) {
        if( !dual_return_and_use_only_first_return ) { rt_err( "non-dual return mode not implemented for 64 laser sensor" ); }
      }
      if( tot_lasers == 32 ) {
        if( dual_return_and_use_only_first_return ) { rt_err( "dual return mode not implemented for 32 laser sensor (doesn't exist?)" ); }
      }
      
      if( !(fov_rot_samps >= 2) ) { rt_err( "fov_rot_samps must be >= 2" ); }
      printf( "data_stream_init(): mode=%s\n", str(mode).c_str() );
      tot_num_read = 0;
      // setup internal state
      fb_sz = sizeof( block_info_t ) + beams_per_fb * sizeof( laser_info_t );
      packet_sz = fbs_per_packet * fb_sz + sizeof( status_info_t );
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

      if( tot_lasers == 32 ) {
        if( laser_to_row_ix_str ) { rt_err( "you-can't/please-don't specify laser_to_row_ix_str (laser order) for 32 laser sensor" ); }
        // the velodyn hdl-32 uses a fixed firing order, with the downward-most laser first. then lasers from the top
        // and bottom blocks are interleaved, continuing from downward-most to upward-most. the nominal spacing of the
        // lasers is 4/3 (i.e. ~1.333) degrees. thus, the pattern is: -30.67, -9.33, -29.33, -8.00 ...

        // so, given this, we simply fill in laser_to_row_ix directly.
        laser_to_row_ix.resize(32);
        for( uint32_t blix = 0; blix != 16; ++blix ) { // most downward first
          for( uint32_t block = 0; block != 2; ++block ) { // lower, upper
            uint32_t const lix = blix*2 + block; // laser index in firing/packet order
            uint32_t const row = 31 - ( block*16 + blix ); // row in scanline order (note: 31 - (rest) flips y axis)
            laser_to_row_ix[lix] = row;
          }
        }
        
      } else {
        if( !laser_to_row_ix_str ) { rt_err( "you must specify laser_to_row_ix_str (laser order) for 64 laser sensor" ); }
        // for now, we take as input the laser order for the hdl64, as determined from the config file horiz angles. we
        // should instead read the config ... but all we need here for the vertical axis is the order of the lasers, not
        // the exact positions. now, the horizontal corrections we're ignoring are more of a problem ...
        vect_string laser_to_row_ix_str_parts = split(*laser_to_row_ix_str,':');
        if( laser_to_row_ix_str_parts.size() != tot_lasers ) {
          rt_err( strprintf( "expected tot_lasers=%s ':' seperated indexes in laser_to_row_ix_str=%s, but got laser_to_row_ix_str_parts.size()=%s\n",
                             str(tot_lasers).c_str(), str(laser_to_row_ix_str).c_str(), str(laser_to_row_ix_str_parts.size()).c_str() ) );
        }
        for( uint32_t i = 0; i != tot_lasers; ++i ) {
          try {  laser_to_row_ix.push_back( lc_str_u32( laser_to_row_ix_str_parts[i] ) ); }
          catch( rt_exception & rte ) { rte.err_msg = "parsing element " + str(i) + " of laser_to_row_ix_str: " + rte.err_msg; throw; }
        }
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
#include"gen/data-stream-velo.cc.nesi_gen.cc"
}
