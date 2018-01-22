// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"nesi.H"
#include"data-stream.H"
#include"data-stream-velo.H"
#include<algorithm>
#include<boost/circular_buffer.hpp>

#include"xml_util.H" // only for config parse function -- move to own header/file?

namespace boda 
{
  float radians( float const & a ) { return a * (M_PI / 180.0f); }

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

  typedef boost::circular_buffer<status_info_t> status_ring_t;
  
  std::ostream & operator <<(std::ostream & os, laser_corr_t const & v) {
    os << strprintf( "vert_corr=%s rot_corr=%s dist_corr=%s dist_corr_x=%s dist_corr_y=%s off_corr_vert=%s off_corr_horiz=%s focal_dist=%s focal_slope=%s", str(v.vert_corr).c_str(), str(v.rot_corr).c_str(), str(v.dist_corr).c_str(), str(v.dist_corr_x).c_str(), str(v.dist_corr_y).c_str(), str(v.off_corr_vert).c_str(), str(v.off_corr_horiz).c_str(), str(v.focal_dist).c_str(), str(v.focal_slope).c_str() );
    return os;
  }

  
  uint16_t const ang_max = 36000;
  uint16_t const half_ang_max = 18000;
  // for angles in the space [0,ang_max), the distance between two angles is always <= 180.0
  // here, we give a delta such that delta = a1 - a2  or a2 + delta = a1 , with delta in [-18000,18000)
  int16_t rel_angle_delta( uint16_t const a1, uint16_t const & a2 ) {
    assert_st( a1 < ang_max );
    assert_st( a2 < ang_max );
    int32_t delta = int32_t(a1) - int32_t(a2);
    if( delta >= half_ang_max ) { delta -= ang_max; }
    if( delta < -int32_t(half_ang_max) ) { delta += ang_max; }
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
  string const velo_cycle_types = velo_cycle_prefix_types + "1234567";

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

  // in velo stream format, lasers 0-31 used laser_block_ids[0], 32-63 use laser_block_ids[1]. note that the [1] value
  // is only present for 64 laser sensors -- 32 laser sensors use [0] as the id for every firing block.
  uint32_t laser_block_ids[2] = { 0xeeff, 0xddff }; 

  struct data_stream_velodyne_t : virtual public nesi, public data_stream_t // NESI(help="parse data stream (velodyne) into per-full-revolution data blocks by merging across packets",
                             // bases=["data_stream_t"], type_id="velodyne")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    // FIXME/NOTE: since this stream reads many packets from vps per-packet that it emits, we can't replace the usage of
    // vps here with a pipe. in general, this falls into the category of more general firing patterns and connections
    // that pipes support (which is one stick and 1-in-1-out per firing for all stages).
    p_data_stream_t vps; //NESI(req=1,help="underlying velodyne packet stream")

    uint32_t fbs_per_packet; //NESI(default="12",help="firing blocks per packet")
    uint32_t beams_per_fb; //NESI(default="32",help="beams per firing block")

    double fov_center; //NESI(default=0.0,help="center of FoV to sample in degrees. frames will be split at (VAL + 180) degrees.")
    uint32_t fov_rot_samps; //NESI(default="384",help="number of samples-in-rotation to extract around fov_center")

    uint32_t tot_lasers; //NESI(default="64",help="total number of lasers. must be either 32 (one block) or 64 laser (two block) scanner.")
    uint32_t enable_proc_status; //NESI(default="1",help="if non-zero, process status bytes (only present for 64 laser scanner).")
    uint32_t print_status_epoch; //NESI(default="0",help="if non-zero, print warning/status info each status epoch (every 4160 packets). only printed if status processing finds complete epoch based on detectioning known-position status type fields (otherwise can't get status errors will print).")

    uint32_t dual_return_and_use_only_first_return; //NESI(default="1",help="if 1, assume dual return mode, and use only first return.")
    p_string laser_to_row_ix_str; //NESI(help="':'-seperated list of 0-based dense-matrix-row values to which to map each laser id to. should have tot_lasers elements, and should be a permutation of [0,tot_lasers).") 

    vect_uint32_t laser_to_row_ix;
    uint32_t fb_sz;
    uint32_t packet_sz;
    uint16_t last_rot;
    uint16_t fov_center_rot;
    uint16_t split_rot;
    dims_t out_dims;
    dims_t out_dims_azi;
    // internal state:
    uint64_t tot_num_read; // num blocks read so far
    p_nda_uint16_t buf_nda;
    p_nda_uint16_t buf_nda_azi;
    uint32_t buf_nda_rot;
    uint32_t rots_till_emit;
    //float azi_step; // last azimith step in degrees
    
    virtual bool seek_to_block( uint64_t const & frame_ix ) {
      // FIXME: do something with internal state here?
      return vps->seek_to_block( frame_ix );
    }

    
    virtual string get_pos_info_str( void ) { return strprintf( "tot_num_read=%s vps info:%s",
                                                                str(tot_num_read).c_str(), vps->get_pos_info_str().c_str() ); }
    uint64_t packet_ix;
    
    virtual data_block_t proc_block( data_block_t const & db ) {
      p_nda_uint16_t out_nda = make_shared<nda_uint16_t>( out_dims );
      p_nda_uint16_t out_nda_azi = make_shared<nda_uint16_t>( out_dims_azi );
      data_block_t ret_db = db;
      data_block_t vps_db;
      uint16_t last_ub_rot = uint16_t_const_max;
      while( !ret_db.valid() ) {
        vps_db = vps->proc_block(data_block_t());
        ++packet_ix;
        if( !vps_db.valid() ) { return db; } // src at end-of-stream, return end-of-stream
        if( vps_db.sz() != packet_sz ) { rt_err(
            strprintf( "lidar decode expected packet_sz=%s but got block with dv.sz()=%s",
                       str(packet_sz).c_str(), str(vps_db.sz()).c_str() ) ); }
        if( verbose > 10 ) { printf( "data_stream_velodyne: %s\n", str(vps_db).c_str() ); }
        status_info_t const * si = (status_info_t *)( (uint8_t *)vps_db.d()+fb_sz*fbs_per_packet);
        if( enable_proc_status ) { proc_status( vps_db, *si ); }
        if( verbose > 10 ) { printf( "  packet: si->gps_timestamp_us=%s si->status_type=%s si->status_val=%s\n",
                                     str(si->gps_timestamp_us).c_str(), str(si->status_type).c_str(), str(uint16_t(si->status_val)).c_str() ); }
        for( uint32_t fbix = 0; fbix != fbs_per_packet; ++fbix ) {
          block_info_t const * bi = (block_info_t *)( (uint8_t *)vps_db.d()+fb_sz*fbix);
          //printf( "   fbix=%s bi->rot_pos=%s\n", str(fbix).c_str(), str(bi->rot_pos).c_str() );
          uint32_t laser_id_base = 0;
          if( tot_lasers == 64 ) {
            if( bi->block_id !=  laser_block_ids[fbix&1] ) {
              rt_err( strprintf( "(64 laser mode) saw unexpected bi->block_id=%s for firing block fbix=%s\n",
                                 str(bi->block_id).c_str(), str(fbix).c_str() ) );
            }
            if( fbix&1 ) { laser_id_base = 32; }
          } else if( tot_lasers == 32 ) {
            if( bi->block_id != laser_block_ids[0] ) {
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
          buf_nda_azi->at1( buf_nda_rot ) = bi->rot_pos;
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
              ret_db.timestamp_ns = vps_db.timestamp_ns;
              if( verbose > 4 ) { printf( "-- TRIGGER -- bi->rot_pos=%s\n", str(bi->rot_pos).c_str() ); }
              if( verbose > 4 ) { printf( "  @TRIGGER: si->gps_timestamp_us=%s si->status_type=%s si->status_val=%s\n",
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
                  if( !j ) { out_nda_azi->at1( i ) = buf_nda_azi->at1( buf_rot ); } // just do once per rot (i.e. sorta-should be outside j loop)
                }
              }
              ret_db.nda = out_nda; // note: setting this triggers loop exit after all FBs in this packet are done
              rots_till_emit = uint32_t_const_max; // back to untriggered state
            } 
          }
          if( verbose > 50 ) printf( "last_rot=%s bi->rot_pos=%s\n", str(last_rot).c_str(), str(bi->rot_pos).c_str() );
          if( last_rot != uint16_t_const_max ) {
            if( last_rot == bi->rot_pos ) {
              printf( "warning: rot unchanged: last_rot=%s bi->rot_pos=%s -- dual return data being proc'd as single return?\n",
                      str(last_rot).c_str(), str(bi->rot_pos).c_str() );
              // FIXME: suppress this warning after a while? use it to detect dual-return mode?
            } else {
              if( verbose > 50 ) {
                float azi_step = float( rel_angle_delta( bi->rot_pos, last_rot ) ) / 100.0; // FIXME: smooth?
                printf( "velo block: fbix=%s azi_step=%s packet_ix=%s\n", str(fbix).c_str(), str(azi_step).c_str(), str(packet_ix).c_str() );
              }
            }
          }
          last_rot = bi->rot_pos;
          buf_nda_rot += 1; if( buf_nda_rot == fov_rot_samps ) { buf_nda_rot = 0; } // FIXME: don't we have inc_mod for this?
        }
      }
      ret_db.meta = "lidar";
      ret_db.tag = "lidar-velodyne-" + str(tot_lasers);
      ret_db.frame_ix = tot_num_read;
      ++tot_num_read;
      if( verbose > 4 ) { printf( "velodyne ret_db: %s\n", str(ret_db).c_str() ); }
      ret_db.subblocks = make_shared<vect_data_block_t>();      
      // send azi-step 
      data_block_t azi_db;
      azi_db.meta = "azi";
      azi_db.nda = out_nda_azi;
      ret_db.subblocks->push_back( azi_db );
      if( laser_corrs_nda ) { // if we read corrections on this packet, wedge them into our output
        data_block_t laser_corrs_db;
        laser_corrs_db.meta = "lidar-corrections";
        laser_corrs_db.nda = laser_corrs_nda;
        // consume corrections so we only send them once per time we see them. FIXME: only send once total? validiate it's the same each time?
        laser_corrs_nda.reset();
        ret_db.subblocks->push_back( laser_corrs_db );
      }
      return ret_db;
    }

    // status-related state
    uint32_t last_status_gps_ts;
    uint64_t last_status_db_ts;
    string last_status_src_pos;
    // TODO: check timestamp sequence
    // TODO: extract config data
    uint32_t cycle_in_epoch;
    uint32_t packet_in_cycle;

    status_ring_t status_ring;
    
    // called on each packet. assumes packets are presented in stream order.
    void on_bad_status( string const & msg ) {
      // set state to confused/unsynced state
      last_status_gps_ts = uint32_t_const_max;
      last_status_db_ts = uint64_t_const_max;
      last_status_src_pos = string();
      packet_in_cycle = uint32_t_const_max; 
      cycle_in_epoch = uint32_t_const_max; // confused/unsynced state
      status_ring.clear();
      if( msg.empty() ) { return; } // initial/expected stream reset, no error
      printf( "%s\n", str(msg).c_str() );
    }

    status_info_t const & read_status_epoch( uint32_t const & laser, uint32_t const & offset ) {
      assert_st( tot_lasers == 64 ); // should only be called in this case
      assert_st( status_ring.full() );
      assert_st( laser < 65 );  // 64 real lasers + 1 'laser' worth of param/config data
      assert_st( offset < 28 ); // need to be able to read a byte at offset; per-laser data is of size 28
      // sigh on the indexing below. also, FIXME, since really this isn't quite right -- the lasers go from packet
      // [1,256], then then only the last three packets [257,259] are params. but last real laser (i.e 0-based laser 63)
      // has some info in the normally 'reserved' laser-info bytes.
      uint32_t const six = velo_packets_in_cycle * ( (4*laser) + (offset / 7) ) + (velo_cycle_prefix_types.size() + (offset % 7)); 
      assert_st( six < status_ring.size() ); // by constuction and earlier asserts, must hold even for any invalid stream.
      return status_ring[six];
    }

    uint8_t read_status_epoch_uint8_t( uint32_t const & laser, uint32_t const & offset ) { return read_status_epoch( laser, offset ).status_val; }
    uint16_t read_status_epoch_uint16_t( uint32_t const & laser, uint32_t const & offset ) {
      return uint16_t( read_status_epoch_uint8_t( laser, offset ) ) + ( uint16_t( read_status_epoch_uint8_t( laser, offset+1 ) ) << 8 );
    }
    float get_float_from_config_int16_t( vect_uint8_t const & d, uint32_t & p ) {
      int16_t ret = uint16_t( d.at( p ) ) + uint16_t( d.at( p + 1 ) << 8 ); // note: assign cases uint16_t to int16_t
      p += 2;
      return ret; // note: return casts int16_t to float
    }

  
    vect_laser_corr_t laser_corrs;
    p_nda_t laser_corrs_nda;

    string status_msg( string const & tag, status_info_t const & si ) {
      return strprintf( "%s gps_timestamp_us=%s status_type=%s status_val=%s\n", tag.c_str(),
                        str(si.gps_timestamp_us).c_str(), str(si.status_type).c_str(), str(uint16_t(si.status_val)).c_str() );
    }

    
    void proc_status_epoch( void ) {
      if( !status_ring.full() ) { on_bad_status( "velodyne stream corrupt; should be at end of epoch, but didn't see enough status data since last sync'd point." ); return; }
      vect_uint8_t config_data;
      for( uint32_t i = 0; i != (65*4) ; ++i ) {
        for( uint32_t j = 0; j != 7 ; ++j ) {
          config_data.push_back( status_ring.at(i*16 + velo_cycle_prefix_types.size() + j).status_val );
        }
      }
      if( print_status_epoch ) {
#if 0 // velo-64 S3
        status_info_t const & si = status_ring.at( 4*16 + velo_cycle_prefix_types.size() + 0 ); // cycle 5 (1-based), first non-repeated status byte
        if( si.status_type != 'W' ) {
          printstr( status_msg( "print_status_epoch: expected 'W' status at cycle 5, offset 9 (config byte slot '1'), but had: ", si ) );
        } else {
          printstr( status_msg( "warning status byte:", si ) );
        }
#else // S2 (or flashed-to-single-return S3?)
        printstr( status_msg( "cycle 260, offset 11 (config byte slot '3')", status_ring.at( 259*16 + velo_cycle_prefix_types.size() + 2 ) ) );
        printstr( status_msg( "cycle 260, offset 12 (config byte slot '4')", status_ring.at( 259*16 + velo_cycle_prefix_types.size() + 3 ) ) );
        printstr( status_msg( "cycle 260, offset 13 (config byte slot '5')", status_ring.at( 259*16 + velo_cycle_prefix_types.size() + 4 ) ) );
#endif
      }
      
      assert_st( config_data.size() == 1820 );

      vect_uint8_t real_config_data;
      //for( uint32_t j = 0; j != 7 ; ++j ) { real_config_data.push_back(status_ring.at( velo_cycle_prefix_types.size() + j ).status_val ); }
      for( uint32_t i = 0; i != 64 ; ++i ) {
        for( uint32_t j = 0; j != 21 ; ++j ) {
          real_config_data.push_back( status_ring.at((i*4 + 1 + (j/7))*16 + velo_cycle_prefix_types.size() + (j%7)).status_val );
        }
      }
      assert_st( real_config_data.size() == (21*64 + 7*0) );

      // FIXME: we have two-ish ways of reading the status data here: the 'compact' 1820 byte vector (no types), used
      // for checksum, and the 'full' reading by laser/offset (which can get the type). maybe we should only have one
      // somehow?
      bool has_checksum = read_status_epoch( 0, 6 ).status_type == 0xF6u; // not ideal, since could be corrupt ...
      uint16_t const len_or_cs = read_status_epoch_uint16_t(64,26);
      if( !has_checksum ) {
        if( len_or_cs != 1820 ) {
          on_bad_status( "velodyne stream corrupt or unsupported; assumed HDL64-S2 stream (with no checksum) had non-1820 config-data-length of " + str(len_or_cs) );
          return;
        }
      } else {
#if 0
        // FIXME: for now, we don't seem to know how to calculation the crc properly, since it never agrees, despite
        // various random attempts at guessing what data to include in the checksum ..
        uint16_t const calc_crc = velo_crc( &real_config_data[0], real_config_data.size() );
        if( calc_crc != len_or_cs ) {
          on_bad_status( strprintf( "velodyne stream corrupt; bad crc: calc_crc=%s len_or_cs=%s\n", str(calc_crc).c_str(), str(len_or_cs).c_str() ) );
          return;          
        }
#endif
      }
      laser_corrs.resize( 64 );
      for( uint32_t i = 0; i != laser_corrs.size(); ++i ) {
        uint32_t pos = 21*i;
        uint8_t const lid = real_config_data.at( pos ); ++pos;
        if( lid != i ) { on_bad_status( strprintf( "velodyne config corrupt: expected config for laser %s but saw lid=%s\n", str(i).c_str(), str(lid).c_str() ) ); return; }
        laser_corr_t & laser_corr = laser_corrs[i];
        laser_corr.vert_corr = get_float_from_config_int16_t( real_config_data, pos ) / 100.0f;
        laser_corr.rot_corr = get_float_from_config_int16_t( real_config_data, pos ) / 100.0f;
        laser_corr.dist_corr = get_float_from_config_int16_t( real_config_data, pos ) / 10.0f;
        laser_corr.dist_corr_x = get_float_from_config_int16_t( real_config_data, pos ) / 10.0f;
        laser_corr.dist_corr_y = get_float_from_config_int16_t( real_config_data, pos ) / 10.0f;
        laser_corr.off_corr_vert = get_float_from_config_int16_t( real_config_data, pos ) / 10.0f;
        laser_corr.off_corr_horiz = get_float_from_config_int16_t( real_config_data, pos ) / 10.0f;
        laser_corr.focal_dist = get_float_from_config_int16_t( real_config_data, pos ) / 10.0f;
        laser_corr.focal_slope = get_float_from_config_int16_t( real_config_data, pos ) / 10.0f;
      }
      if( verbose > 4 ) {
        printf( "epoch ok: len_or_cs=%s\n", str(len_or_cs).c_str() );
        for( uint32_t i = 0; i != laser_corrs.size(); ++i ) {
          printf( "laser_corrs[%2s] = %s\n", str(i).c_str(), str(laser_corrs[i]).c_str() );
        }
      }
      laser_corrs_nda = make_shared<nda_float_t>( dims_t{
          vect_uint32_t{uint32_t(laser_corrs.size()),sizeof(laser_corr_t)/sizeof(float)},
            vect_string{"l","v"}, "float" } );
      assert_st( laser_corrs_nda->dims.bytes_sz() == sizeof(laser_corr_t)*laser_corrs.size() );
      std::copy( (uint8_t *)&laser_corrs[0], (uint8_t *)&laser_corrs[0]+laser_corrs_nda->dims.bytes_sz(), (uint8_t *)laser_corrs_nda->rp_elems() );      
    }
    
    void proc_status_cycle( void ) {
      assert_st( status_ring.size() >= velo_packets_in_cycle ); // 16 (total packets/cycle) = 9 (# prefix statuses) + 7 (per-cycle stuff)
      if( cycle_in_epoch == uint32_t_const_max ) { // if we're confused/unsynced, just look for 0xFE as first non-prefix status type
        if( status_ring[status_ring.size()-7].status_type == 0xFEu ) { cycle_in_epoch = 257; } // at last-2 cycle of epoch
      }
      if( cycle_in_epoch == uint32_t_const_max ) { return; } // if (still) no cycle sync, give up for now
      // process cycle
      ++cycle_in_epoch;
      // if epoch done, do end-of-epoch processing (checksum, capture config)
      if( cycle_in_epoch == velo_cycles_in_epoch ) {
        proc_status_epoch();
        cycle_in_epoch = 0;
        // TODO: clear epoch-related stuff
      }
      
    }

    
    void proc_status( data_block_t const & db, status_info_t const & si ) {
      //printf( "si.status_type=%s si.status_val=%s\n", str(si.status_type).c_str(), str(si.status_val).c_str() );
      if( last_status_gps_ts != uint32_t_const_max ) { // if we had a prior timestamp
        bool had_err = 0;
        if( si.gps_timestamp_us < last_status_gps_ts ) {
          printf( "timestamp went backwards:\n" );
          had_err = 1;
        } else {
          uint32_t ts_delta = si.gps_timestamp_us - last_status_gps_ts;
          uint32_t max_ts_delta = (tot_lasers == 32) ? 600 : (dual_return_and_use_only_first_return ? 200 : 300 );
          if( ts_delta > max_ts_delta ) {
            printf( "large (>max_ts_delta=%s) ts_delta=%s:\n", str(max_ts_delta).c_str(), str(ts_delta).c_str() );
            had_err = 1;
          }
        }
        if( had_err ) {
          printf( "  @ si.gps_timestamp_us=%s db.timestamp_ns=%s src_pos=%s (prior packet: gps=%s db=%s src_pos=%s);\n",
                  str(si.gps_timestamp_us).c_str(), str(db.timestamp_ns).c_str(), vps->get_pos_info_str().c_str(),
                  str(last_status_gps_ts).c_str(), str(last_status_db_ts).c_str(), last_status_src_pos.c_str() );
        }
      }
      
      last_status_gps_ts = si.gps_timestamp_us;
      last_status_db_ts = db.timestamp_ns;
      last_status_src_pos = vps->get_pos_info_str();
      
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
      status_ring.push_back( si ); // capture type/val
      // we need to know cycle_in_epoch to do more parsing/checking. for simplicity, we defer this until the cycle is complete
      ++packet_in_cycle;
      if( packet_in_cycle == velo_packets_in_cycle ) {
        proc_status_cycle();
        packet_in_cycle = 0;
      }
    }
    
    // init/setup

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      packet_ix = 0;
      on_bad_status("");
      
      if( ! ( (tot_lasers == 64) || (tot_lasers == 32) ) ) { rt_err( "non-32/64 laser mode not implemented" ); }
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
      out_dims_azi = dims_t{ dims_t{ { fov_rot_samps }, {"x"}, "uint16_t" }};
      buf_nda = make_shared<nda_uint16_t>( out_dims );
      buf_nda_azi = make_shared<nda_uint16_t>( out_dims_azi );
      buf_nda_rot = 0;
      rots_till_emit = uint32_t_const_max; // start in untriggered state

      if( tot_lasers == 32 ) {
        if( !laser_to_row_ix_str ) {
          // if no mapping specified, put laser in packet/firing/raw order; we assume corrections will be done later
          for( uint32_t i = 0; i != tot_lasers; ++i ) { laser_to_row_ix.push_back(i); }
        } else {
          if( *laser_to_row_ix_str != "default-32" ) { rt_err( "currently, only the 'default-32' laserremapping order for 32 laser sensors is supported" ); }
          // the velodyn hdl-32 uses a fixed firing order, with the downward-most laser first. then lasers from the top
          // and bottom blocks are interleaved, continuing from downward-most to upward-most. the nominal spacing of the
          // lasers is 4/3 (i.e. ~1.333) degrees. thus, the pattern is: -30.67, -9.33, -29.33, -8.00 ...

          // so, given this, we simply fill in laser_to_row_ix directly.
          laser_to_row_ix.resize(tot_lasers);
          for( uint32_t blix = 0; blix != 16; ++blix ) { // most downward first
            for( uint32_t block = 0; block != 2; ++block ) { // lower, upper
              uint32_t const lix = blix*2 + block; // laser index in firing/packet order
              uint32_t const row = 31 - ( block*16 + blix ); // row in scanline order (note: 31 - (rest) flips y axis)
              laser_to_row_ix[lix] = row;
            }
          }
        }
      } else if( tot_lasers == 64 ) {
          status_ring.set_capacity( velo_cycles_in_epoch * velo_packets_in_cycle ); // i.e. 16*260 = 4160 packets
          
        if( !laser_to_row_ix_str ) {
          // if no mapping specified, put laser in packet/firing/raw order; we assume corrections will be done later
          for( uint32_t i = 0; i != tot_lasers; ++i ) { laser_to_row_ix.push_back(i); }
        } else {
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
      } else { assert_st(0); }
      vect_uint32_t laser_to_row_ix_sorted = laser_to_row_ix;
      sort( laser_to_row_ix_sorted.begin(), laser_to_row_ix_sorted.end() );
      assert_st( laser_to_row_ix_sorted.size() == tot_lasers );
      for( uint32_t i = 0; i != tot_lasers; ++i ) {
        if( laser_to_row_ix_sorted[i] != i ) { rt_err( "the elements of laser_to_row_ix_sorted are not a permutation of [0,tot_lasers)" ); }
      }
      
      vps->data_stream_init( nia );
        
    }   
  };

  using pugi::xml_document;
  using pugi::xml_node;
  
  // appends corrections to laser_corrs
  void read_velo_config( filename_t const & velo_config_xml_fn, vect_laser_corr_t & laser_corrs ) {
    assert_st( laser_corrs.empty() );
    char const * const fn = velo_config_xml_fn.exp.c_str();

    ensure_is_regular_file( fn );
    xml_document doc;
    xml_node cfg = xml_file_get_root( doc, velo_config_xml_fn.exp );

    xml_node points_ = xml_must_decend( fn, xml_must_decend( fn, cfg, "DB" ), "points_" );

    uint32_t lix = 0;
    for( xml_node item = points_.child("item"); item; item = item.next_sibling("item") ) {
      xml_node px = xml_must_decend( fn, item, "px" );
      // FIXME: does object_id matter?
      string const oid = xml_must_get_attr( fn, px, "object_id" ).value();
      ++lix;
      string const lix_as_str = "_"+str(lix);
      if( oid != lix_as_str ) { rt_err(strprintf( "velo config parse failed: expected object_id oid=%s to match lix_as_str=%s\n",
                                                  str(oid).c_str(), str(lix_as_str).c_str() ) ); } // FIXME: too strong?
      laser_corr_t laser_corr;
      laser_corr.vert_corr = lc_str_d( xml_must_decend( fn, px, "vertCorrection_" ).child_value() );
      laser_corr.rot_corr = lc_str_d( xml_must_decend( fn, px, "rotCorrection_" ).child_value() );
      laser_corr.dist_corr = lc_str_d( xml_must_decend( fn, px, "distCorrection_" ).child_value() );
      laser_corr.dist_corr_x = lc_str_d( xml_must_decend( fn, px, "distCorrectionX_" ).child_value() );
      laser_corr.dist_corr_y = lc_str_d( xml_must_decend( fn, px, "distCorrectionY_" ).child_value() );
      laser_corr.off_corr_vert = lc_str_d( xml_must_decend( fn, px, "vertOffsetCorrection_" ).child_value() );
      laser_corr.off_corr_horiz = lc_str_d( xml_must_decend( fn, px, "horizOffsetCorrection_" ).child_value() );
      laser_corr.focal_dist = lc_str_d( xml_must_decend( fn, px, "focalDistance_" ).child_value() );
      laser_corr.focal_slope = lc_str_d( xml_must_decend( fn, px, "focalSlope_" ).child_value() );

      laser_corrs.push_back( laser_corr );
    }
  }
  // for reference, here's what the first and second items in the corrections list look like
#if 0
  		<item class_id="8" tracking_level="0" version="1">
			<px class_id="9" tracking_level="1" version="1" object_id="_1">
				<id_>0</id_>
				<rotCorrection_>-4.6461787</rotCorrection_>
				<vertCorrection_>-7.2296872</vertCorrection_>
				<distCorrection_>118.20465</distCorrection_>
				<distCorrectionX_>123.05249</distCorrectionX_>
				<distCorrectionY_>121.988</distCorrectionY_>
				<vertOffsetCorrection_>21.569468</vertOffsetCorrection_>
				<horizOffsetCorrection_>2.5999999</horizOffsetCorrection_>
				<focalDistance_>2400</focalDistance_>
				<focalSlope_>1.15</focalSlope_>
			</px>
		</item>
		<item>
			<px class_id_reference="9" object_id="_2">
				<id_>1</id_>
				<rotCorrection_>-2.6037366</rotCorrection_>
				<vertCorrection_>-6.9851909</vertCorrection_>
				<distCorrection_>129.59227</distCorrection_>
				<distCorrectionX_>132.77873</distCorrectionX_>
				<distCorrectionY_>133.79263</distCorrectionY_>
				<vertOffsetCorrection_>21.538305</vertOffsetCorrection_>
				<horizOffsetCorrection_>-2.5999999</horizOffsetCorrection_>
				<focalDistance_>1000</focalDistance_>
				<focalSlope_>1.6</focalSlope_>
			</px>
		</item>
#endif

  string const corr_template = R"rstr(
		<item>
			<px class_id_reference="9" object_id="_%s">
				<id_>%s</id_>
				<rotCorrection_>%s</rotCorrection_>
				<vertCorrection_>%s</vertCorrection_>
				<distCorrection_>%s</distCorrection_>
				<distCorrectionX_>%s</distCorrectionX_>
				<distCorrectionY_>%s</distCorrectionY_>
				<vertOffsetCorrection_>%s</vertOffsetCorrection_>
				<horizOffsetCorrection_>%s</horizOffsetCorrection_>
				<focalDistance_>%s</focalDistance_>
				<focalSlope_>%s</focalSlope_>
			</px>
		</item>

  )rstr";

  // for now, this is a semi-manual procedure; this outputs only the inner corrections blocks, and must be spliced into a full config file. for now we don't attempt to reorder the other per-laser blocks (i.e. color, min_intensity) on the thoery they're not needed for our current applications.
  void write_velo_config( filename_t const & velo_config_xml_fn, vect_laser_corr_t const & laser_corrs ) {
    assert_st( laser_corrs.size() == 64 ); // yep, even for 32-laser case, this is the velodyne convention.
    p_ostream out = ofs_open( velo_config_xml_fn );

    for( uint32_t i = 0; i != laser_corrs.size(); ++i ) {
      laser_corr_t const & laser_corr = laser_corrs[i];
      (*out) << strprintf( corr_template.c_str(), str(i+1).c_str(), str(i).c_str(),
                           str(laser_corr.rot_corr).c_str(),
                           str(laser_corr.vert_corr).c_str(),
                           str(laser_corr.dist_corr).c_str(),
                           str(laser_corr.dist_corr).c_str(),
                           str(laser_corr.dist_corr_y).c_str(),
                           str(laser_corr.off_corr_vert).c_str(),
                           str(laser_corr.off_corr_horiz).c_str(),
                           str(laser_corr.focal_dist).c_str(),
                           str(laser_corr.focal_slope).c_str() );
      
    }
  }

  
  struct velo_std_block_info_t {
    uint16_t block_id;
    uint16_t rot_pos;
    laser_info_t lis[32];
  } __attribute__((packed));
  
  struct velo32_std_udp_payload_t {
    velo_std_block_info_t bis[12];
    status_info_t si;
  } __attribute__((packed));
  
  struct data_stream_velodyne_gen_t : virtual public nesi, public data_stream_t // NESI(help="convert dense point cloud data blocks into sequences of velodyne udp packet payloads",
                             // bases=["data_stream_t"], type_id="velodyne-gen")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")

    uint32_t fbs_per_packet; //NESI(default="12",help="firing blocks per packet")
    uint32_t beams_per_fb; //NESI(default="32",help="beams per firing block")

    double fov_center; //NESI(default=0.0,help="center of FoV to sample in degrees. frames will be split at (VAL + 180) degrees.")
    double azi_step; //NESI(default=0.165,help="per firing azimuith step in degrees.")
    uint32_t timestamp_step; //NESI(default="553",help="per-packet gps timestamp step in us.")
    uint32_t timestamp_start; //NESI(default="0",help="starting per-packet gps timestamp in us.")
    
    uint32_t tot_lasers; //NESI(default="32",help="total number of lasers. must be either 32 (one block) or 64 laser (two block) scanner.")

    p_string laser_to_row_ix_str; //NESI(help="':'-seperated list of 0-based dense-matrix-row values to which to map each laser id to. should have tot_lasers elements, and should be a permutation of [0,tot_lasers).") 
    vect_uint32_t laser_to_row_ix;
    
    // output/buffer state
    uint32_t cur_out_fb_ix;
    velo32_std_udp_payload_t cur_out;

    uint32_t cur_in_azi_ix;
    p_nda_t cur_in_nda;

    uint32_t cur_timestamp;
    uint32_t packet_in_cycle;

    
    virtual string get_pos_info_str( void ) { return strprintf( "velodyne-gen: info TODO." ); }

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      if( !( fbs_per_packet == 12 ) ) { rt_err( "only standard 12-firing-blocks-per-packet output is implemented" ); }
      if( !( beams_per_fb == 32 ) ) { rt_err( "only standard 32-beams-per-firing-block output is implemented" ); }

      cur_out_fb_ix = 0;
      packet_in_cycle = 0;
      cur_in_azi_ix = uint32_t_const_max;
      cur_timestamp = timestamp_start;
            
      printf( "data_stream_init(): mode=%s\n", str(mode).c_str() );
      // setup internal state
      if( (fov_center < 0.0) || (fov_center >= 360.0) ) { rt_err( strprintf( "fov_center must be in [0.0,360.0) but was =%s",
                                                                             str(fov_center).c_str() ) ); }
      // FIXME: dup'd with parser above, 64 case not handled ... factor out, and handle both.
      if( tot_lasers == 32 ) {
        if( !laser_to_row_ix_str ) {
          // if no mapping specified, put laser in packet/firing/raw order; we assume corrections will be done later
          for( uint32_t i = 0; i != tot_lasers; ++i ) { laser_to_row_ix.push_back(i); }
        } else {
          if( *laser_to_row_ix_str != "default-32" ) { rt_err( "currently, only the 'default-32' laserremapping order for 32 laser sensors is supported" ); }
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
        }
      } else if( tot_lasers == 64 ) {
        if( !laser_to_row_ix_str ) {
          // if no mapping specified, put laser in packet/firing/raw order; we assume corrections will be done later
          for( uint32_t i = 0; i != tot_lasers; ++i ) { laser_to_row_ix.push_back(i); }
        }
        if( laser_to_row_ix_str ) {
          rt_err( "currently, laser remapping is not support for the 64 laser case" ); 
        }
       
      } else { rt_err( "error: only 32 and 64 laser modes are implemented" ); }
        
      vect_uint32_t laser_to_row_ix_sorted = laser_to_row_ix;
      sort( laser_to_row_ix_sorted.begin(), laser_to_row_ix_sorted.end() );
      assert_st( laser_to_row_ix_sorted.size() == tot_lasers );
      for( uint32_t i = 0; i != tot_lasers; ++i ) {
        if( laser_to_row_ix_sorted[i] != i ) { rt_err( "the elements of laser_to_row_ix_sorted are not a permutation of [0,tot_lasers)" ); }
      }
    }   

    bool have_packet_ready( void ) const { return cur_out_fb_ix == fbs_per_packet; }
    
    virtual data_block_t proc_block( data_block_t const & db ) {
      if( have_packet_ready() ) { return emit_packet(); } // note: intput db discarded here; should get here exactly if
      assert_st( !cur_in_nda ); // no packet ready should imply no buffered input data remains (but, note there can be a packet ready with no data left).
      cur_in_nda = db.nda; cur_in_azi_ix = 0; 
      consume_some_input();
      // we assume we should always be able to produce at least one packet from each input data block, even if we had no buffered fbs      
      return emit_packet(); 
    }

    void consume_some_input( void ) { 
      assert_st( cur_in_nda );
      assert_st( cur_in_nda->dims.sz() == 2 );
      uint32_t const num_azis = cur_in_nda->dims.dims(1);
      uint32_t const cur_in_num_lasers = cur_in_nda->dims.dims(0);
      uint16_t const * const cur_in_rp_elems = nda_rp_elems<uint16_t>( cur_in_nda );
      if( cur_in_num_lasers != tot_lasers ) {
        rt_err( strprintf( "velodyne-gen: configured to output tot_lasers=%s but got block with cur_in_num_lasers=%s\n",
                           str(tot_lasers).c_str(), str(cur_in_num_lasers).c_str() ) );
      }
      assert_st( cur_out_fb_ix <= fbs_per_packet );
      while( cur_out_fb_ix < fbs_per_packet ) {
        // for each iter, fill in one firing block from one azi
        assert_st( cur_in_azi_ix < num_azis );
        double cur_azi_deg = fov_center + azi_step * ( double(cur_in_azi_ix) - double(num_azis)/2.0 );
        if( cur_azi_deg < 0.0 ) { cur_azi_deg += 360.0; }
        if( (cur_azi_deg < 0.0) || (cur_azi_deg >= 360.0) ) { rt_err( strprintf( "cur_azi_deg must be in [0.0,360.0) but was =%s -- fov_center bad? azi_step too high? too many azi samples in frame?", str(cur_azi_deg).c_str() ) ); }

        // fill in lasers (always 1 or 2 FBs per azi)
        for( uint32_t laser_id_base = 0; laser_id_base != tot_lasers; laser_id_base += beams_per_fb ) {
          velo_std_block_info_t & bi = cur_out.bis[cur_out_fb_ix];
          bi.rot_pos = uint16_t(cur_azi_deg*100); // FIXME: truncation okay/correct here?
          bi.block_id = laser_block_ids[laser_id_base/32]; // either 0 or 1
          // note: can't wrap packets across the (max 2) FBs per azi, since 2 divides 12 evenly.
          assert_st( cur_out_fb_ix < fbs_per_packet ); 
          for( uint32_t i = 0; i != beams_per_fb; ++i ) {
            uint32_t const rix = laser_to_row_ix.at(laser_id_base+i);
            bi.lis[i].distance = cur_in_rp_elems[ cur_in_nda->dims.chk_ix2( rix, cur_in_azi_ix ) ];
            bi.lis[i].intensity = 90;
          }
          ++cur_out_fb_ix;
        }

        ++cur_in_azi_ix;
        if( cur_in_azi_ix == num_azis ) { // if we exactly finished our current input block, clear it out, and we must return
          cur_in_azi_ix = uint32_t_const_max;
          cur_in_nda.reset();
          return;
        }
      }
    }
    data_block_t emit_packet( void ) {

      assert_st( have_packet_ready() );
      
      // finish packet and emit. first, status processing:
      assert_st( packet_in_cycle < velo_cycle_types.size() );
      if( tot_lasers == 64 ) {
        cur_out.si.status_type = velo_cycle_types[packet_in_cycle];
        cur_out.si.status_val = 0; // FIXME: no status values set yet ...
        //rt_err( "TODO: set status values for velo 64 ... maybe can mostly ignore? set at least 'V' (firmware ver) byte?" );

        // yeah ... sure, we'll set this to 0xa0, why not? that seems like a reasonable value for a velo 64, right? see
        // veloview src, vtkDataPacket.h ... however, for 64 data, it's probably not neccessary to set any status data,
        // and veloview seems to ignore it (or be okay ignoring it). maybe ideally we'd want to insert the calibration
        // data, but that seems hard and not-too-usefull.
        if( cur_out.si.status_type == 'V' ) { cur_out.si.status_val = 0xa0; } 
      } else if (tot_lasers == 32 ) {
        // values taken from random velo32 .pcap ... seem constant? there are no corrections for velo32, maybe no status stuff either?
        cur_out.si.status_type = 7;
        cur_out.si.status_val = 33; // HDL32E id-byte
      }
      cur_out.si.gps_timestamp_us = cur_timestamp;
      ++packet_in_cycle;
      if( packet_in_cycle == velo_cycle_types.size() ) { packet_in_cycle = 0; }
      cur_timestamp += timestamp_step;
      uint32_t const hour_in_us = 3600U*1000U*1000U;
      if( cur_timestamp >= hour_in_us  ) { cur_timestamp -= hour_in_us; } // wrap timestamp each hour; note: assumes step+hour_in_us fits in uint32_t
      
      data_block_t ret;
      ret.nda = make_shared<nda_t>( dims_t{ vect_uint32_t{uint32_t(sizeof(velo32_std_udp_payload_t))}, "uint8_t" } );
      //printf( "ret.nda->dims.bytes_sz()=%s\n", str(ret.nda->dims.bytes_sz()).c_str() );
      uint8_t const *udp_payload = (uint8_t const *)&cur_out;
      std::copy( udp_payload, udp_payload + ret.nda->dims.bytes_sz(), (uint8_t *)ret.d() );
      cur_out_fb_ix = 0; // mark packet data as ouput
      
      if( cur_in_nda ) { consume_some_input(); } // if we have any, consume some more input, which may create another entire packet and/or finish cur_in
      ret.have_more_out = have_packet_ready(); // exactly if we have another packet ready, we don't want a new db on the next call.
      return ret;
      
    }

  };

  // FIXME: split into base utility NESI class and derived stream-transformer class
  // FIXME: use split base as base for os-render (to eliminate duplicated code)

  // notes on velodyne coordinate conventions: the sensor returns results in the form of (range,intensity) pairs. each
  // laser has a fixed elevation angle, with 0 being level, and negative angles pointing down. the sensor spins
  // clockwise (viewed from above), with a zero azimuth angle being forward, and positive azimuth angles being more
  // clockwise. azimuth angles range from [0,360) degrees.

  // given these angle conventions, the standard conversion to XYZ (not including corrections/offsets) is as follows:

  // xyDistance=distance*cosVertAngle;
  // X = xyDistance*sinRotAngle;
  // Y = xyDistance*costRotAngle;
  // Z = distance*sinVertAngle;

  // so, this means that (color in veloview) the axis conventions are:
  // y forward (yellow)
  // x right (reg)
  // z up (green)

  // note on azimuth steps (azi_step):
  // for HDL32, normal (10HZ) should be 0.165 per sample
  // for HDL64, normal (10HZ) single-return should be 0.1729106628
  // for HDL64, normal (10HZ) dual-return should be ~0.21 (27 18 18 27 18 18 27 18 18 18 27 18 18 ...) 
  
  struct velo_pcdm_to_xyz_t : virtual public nesi, public data_stream_t // NESI(help="annotate data blocks (containing point cloud data) with image representations (in as_img field of data block). returns annotated data block.",
                           // bases=["data_stream_t"], type_id="velo-pcdm-to-xyz")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t use_live_laser_corrs; //NESI(default="1",help="if laser corrs present in stream, use them. note: only first set of corrs in stream will be used.")
    p_filename_t velo_cfg; //NESI(help="xml config filename (optional; will try to read from stream if not present. but note, do need config somehow, from stream or file!")
    p_filename_t write_velo_cfg; //NESI(help="if specified, dump initial velo corrections (after loading and maybe pcdm-ifing or other modifications) to this file")
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    float fov_center; //NESI(default="0.0",help="default center angle (only used when generating azimuths using azi_step)")
    float azi_step; //NESI(default=".165",help="default azimuth step (stream can override)")
    float dist_scale; //NESI(default=".002",help="to get meters, multiply raw distances by this value")
    float x_offset; //NESI(default="0.0",help="offset to add to X output values")
    float y_offset; //NESI(default="0.0",help="offset to add to Y output values")
    float z_offset; //NESI(default="0.0",help="offset to add to Z output values")

    virtual void data_stream_init( nesi_init_arg_t * const nia ) { } // init is mostly done per-frame (or at least deferred to first frame)

    vect_laser_corr_t laser_corrs;
    zi_bool got_live_laser_corrs;
    
    void setup_laser_corrs( string const & meta, data_block_t const & db ) {
      assert_st( db.nda.get() );
      uint32_t const y_sz = db.nda->dims.must_get_dim_by_name( "y" ).sz; // generally should be 32 or 64
      bool const is_pcdm = endswith( db.meta, "/PCDM" );
      bool init_no_corrs = laser_corrs.empty();
      // for all current cases, we want to load any corrs file if one is provided
      if( init_no_corrs ) { if( velo_cfg.get() ) { read_velo_config( *velo_cfg, laser_corrs ); } }
      // do per-sensor-type checking/setup
      if( 0 ) { }
      else if( is_pcdm || endswith( db.meta, "/VD_HDL64" ) ) {
        if( y_sz != 64 ) { rt_err( strprintf( "for PCDM or VD_HDL64 lidar data, there must be exactly 64 lines per frame, but saw %s lines.", str(y_sz).c_str() ) ); }
        azi_step = 0.1729106628; // FIXME: not right for dual-return velo-64 data ... hope we have azimuth data in that case? azi_step is mainly needed for PCDM case.
        if( laser_corrs.empty() ) {
          rt_err( "currently, for PCDM/VD_HDL64 lidar data, laser corrs must be set (by conf file) at init time, but they were not. please specify a velo-64 config file." );
        }
#if 0 // might want to (re-)allow this case later ...
        if( laser_corrs.empty() ) {
          printf( "WARNING: no laser corrections loaded for VD_HDL64 case; "
                  "display/output will be incorrect. This will persist until/unless in-stream calibration is found at some point.\n" );
        }
#endif
      }
      else if( endswith( db.meta, "/VD_HDL32" ) ) {
        if( y_sz != 32 ) { rt_err( strprintf( "for VD_HDL32 lidar data, there must be exactly 64 lines per frame, but saw %s lines.", str(y_sz).c_str() ) ); }
        azi_step = 0.165;
        if( laser_corrs.empty() ) {
          // if no config file, generate default laser_corrs that spread out beams. FIXME: not ideal and can be
          // confusing, but maybe better than doing nothing here? the actual values shoul be the correct ones for
          // already-reordered-velo32 case (using only the first 32 laser corrs).
          double const elev_start_degrees = 10.67;
          double const elev_per_row_degrees = 1.333;
          for( uint32_t i = 0; i != 64; ++i ) {
            float const row_elev = elev_start_degrees - elev_per_row_degrees*double(i);
            laser_corrs.push_back( laser_corr_t{ row_elev, 0, 0, 0, 0, 0, 0, 0, 0 } );
          }
        }
      }
      else { rt_err( "uknown lidar sensor type: " + meta ); }
      
      // check if we have live laser corrs in this block. if so, read them
      if( db.has_subblocks() && use_live_laser_corrs && (!got_live_laser_corrs.v) ) {
        for( uint32_t i = 0; i != db.subblocks->size(); ++i ) {
          data_block_t const & sdb = db.subblocks->at(i);
          if( sdb.meta == "lidar-corrections" ) {
            // could allow. pcdm ajust will occur below, since we set init_no_corrs = 1 here.
            if( is_pcdm ) { rt_err( "live corrections not expected/supported in pcdm mode." ); } 
            got_live_laser_corrs.v = 1;
            laser_corrs.clear();
            init_no_corrs = 1;
            p_nda_t const & laser_corrs_nda = sdb.nda;
            assert_st( laser_corrs_nda->dims.sz() == 2 );
            assert_st( laser_corrs_nda->dims.strides(0)*sizeof(float) == sizeof(laser_corr_t) );
            laser_corr_t const * laser_corr = (laser_corr_t const *)laser_corrs_nda->rp_elems();
            for( uint32_t i = 0; i != laser_corrs_nda->dims.dims(0); ++i ) {
              laser_corrs.push_back( *laser_corr );
              //printf( "i=%s (*laser_corr)=%s\n", str(i).c_str(), str((*laser_corr)).c_str() );
              ++laser_corr;
            }
          }
        }
      }      
      assert_st( laser_corrs.size() == 64 ); // for now, in all cases, we should have 64 lasers of corrections here (even in 32 laser case)
      // for pcdm mode, if these are new corrs (from file/stream), we need to disable horiz/rot corrections as well as reorder the lasers.
      if( init_no_corrs && is_pcdm ) {
        for( vect_laser_corr_t::iterator i = laser_corrs.begin(); i != laser_corrs.end(); ++i ) {  (*i).rot_corr = 0.0; }
        std::sort( laser_corrs.begin(), laser_corrs.end(), laser_corr_t_by_vert_corr() );
      }
      if( write_velo_cfg.get() ) { write_velo_config( *write_velo_cfg, laser_corrs ); }
    }

    p_nda_t azi_nda;
    void setup_azis( data_block_t const & db ) {
      assert_st( db.nda.get() );
      azi_nda.reset();
      if( db.has_subblocks() ) {
        for( uint32_t i = 0; i != db.subblocks->size(); ++i ) {
          data_block_t const & sdb = db.subblocks->at(i);
          if( sdb.meta == "azi" ) { azi_nda = sdb.nda; }
        }
      }
      if( !azi_nda ) {
        uint32_t const fov_rot_samps = db.nda->dims.dims(1);
        p_nda_uint16_t azi_nda_u16 = make_shared<nda_uint16_t>( dims_t{ dims_t{ { fov_rot_samps }, {"x"}, "uint16_t" }} );
        for( uint32_t i = 0; i != fov_rot_samps; ++i ) {
          double cur_azi_deg = fov_center + azi_step * ( double(i) - double(fov_rot_samps)/2.0 );
          if( cur_azi_deg < 0.0 ) { cur_azi_deg += 360.0; }
          azi_nda_u16->at1( i ) = uint16_t( cur_azi_deg * 100.0 );
        }
        azi_nda = azi_nda_u16;
      }

      // check azi data size
      assert_st( azi_nda );
      assert_st( azi_nda->dims.sz() == 1 );            
      assert_st( azi_nda->dims.tn == "uint16_t" );            
      assert_st( db.nda->dims.dims(1) == azi_nda->dims.dims(0) ); // i.e. size must be hbins
        
    }

    virtual data_block_t proc_block( data_block_t const & db ) {
      if( !db.nda.get() ) { rt_err( "velo-pcdm-to-xyz: expected nda data in block, but found none." ); }
      setup_laser_corrs( db.meta, db );
      setup_azis( db );

      u32_pt_t xy_sz = get_xy_dims_strict( db.nda->dims );

      nda_T<uint16_t> pcdm_nda( db.nda );
      p_nda_float_t xyz_nda = make_shared<nda_float_t>( dims_t{ dims_t{ { xy_sz.d[1], xy_sz.d[0], 3 }, {"y","x","xyz"}, "float" }} );

      nda_uint16_t azi_nda_u16( azi_nda );

      //uint16_t const * const azi_d = nda_rp_elems<uint16_t>( azi_nda );

      for( uint32_t y = 0; y != xy_sz.d[1] ; ++y ) {
        laser_corr_t const & lc = laser_corrs.at(y);

        float const elev_ang = radians( lc.vert_corr );

        float const dist_corr =  lc.dist_corr / 100.;
        float const off_corr_vert = lc.off_corr_vert / 100.;
        float const off_corr_horiz = lc.off_corr_horiz / 100.;
        
        for( uint32_t x = 0; x != xy_sz.d[0] ; ++x ) {
          float * const xyz = &xyz_nda->at2(y,x);
          float const dist = float(pcdm_nda.at2(y,x)) * dist_scale + dist_corr;
          float const azi_ang = radians(float(azi_nda_u16.at1(x))/100.0f - lc.rot_corr );
          
          float const sin_azi = sin(azi_ang);
          float const cos_azi = cos(azi_ang);
          float const sin_elev = sin(elev_ang);
          float const cos_elev = cos(elev_ang);
          
          float const dist_xy = dist * cos_elev - off_corr_vert * sin_elev; // elev 0 --> dist_xy = dist
          xyz[0] = dist_xy * sin_azi - off_corr_horiz * cos_azi + x_offset; // azi 0 --> x = 0; y = dist_xy
          xyz[1] = dist_xy * cos_azi + off_corr_horiz * sin_azi + y_offset;
          xyz[2] = dist * sin_elev + off_corr_vert + z_offset;
        }
      }
           
      data_block_t ret = db;
      ret.nda = xyz_nda;
      ret.meta = "pointcloud";
      return ret;
    }

    virtual string get_pos_info_str( void ) { return strprintf( "velo-pcdm-to-xyz: <no-state>" ); }
    
  };

  
#include"gen/data-stream-velo.cc.nesi_gen.cc"
}
