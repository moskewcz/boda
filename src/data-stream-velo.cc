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
#include"gen/data-stream-velo.cc.nesi_gen.cc"
}
