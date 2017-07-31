// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"img_io.H"
#include"str_util.H"
#include"data-to-img.H"
#include"data-stream.H"

namespace boda 
{
  
  struct data_to_img_raw_t : virtual public nesi, public data_to_img_t // NESI(help="convert data blocks (containing raw video frames) to images",
                           // bases=["data_to_img_t"], type_id="raw")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    u32_pt_t frame_sz; //NESI(req=1,help="X/Y frame size")
    u32_box_t crop; //NESI(default="0:0:0:0",help="crop pels from sides")
    string img_fmt; //NESI(req=1,help="image format; valid values '16u-grey', '32f-grey', '16u-RGGB' ")
    uint32_t level_adj; //NESI(default=1,help="if non-zero, adjust levels with filt_alpha LPF sliding window on per-frame min/max. otherwise (if 0), make some random assumptions about levels: 12-bit for 16u-grey/RGGB, direct-cast-to-uint8_t for 32f")
    uint32_t level_adj_log_for_float_data; //NESI(default=0,help="if non-zero, level_adj will use for log-scale normalization for float data values")
    float level_filt_alpha; //NESI(default=.9,help="LPF alpha constant for sliding window level adjustment")

    float rgb_levs_filt_min;
    float rgb_levs_filt_rng; 
    float rgb_levs_frame_min;
    float rgb_levs_frame_max;
    
    // internal state and buffers
    zi_uint64_t frame_sz_bytes;
    p_img_t frame_buf;
    i32_pt_t samp_pt;
    uint32_t get_bytes_per_pel( void ) {
      if( startswith(img_fmt, "32") ) { return 4; }
      if( startswith(img_fmt, "16") ) { return 2; }
      rt_err( "can't determine bytes-per-pel: unknown img_fmt: " + img_fmt );
    }
    
    virtual void set_samp_pt( i32_pt_t const & samp_pt_ ) { samp_pt = samp_pt_; }
    
    virtual void data_to_img_init( nesi_init_arg_t * const nia ) {
      frame_sz_bytes.v = frame_sz.dims_prod() * get_bytes_per_pel();
      frame_buf = make_shared< img_t >();
      assert_st( frame_sz.both_dims_gt( crop.bnds_sum() ) );
      frame_buf->set_sz_and_alloc_pels( frame_sz - crop.bnds_sum() );
      samp_pt = i32_pt_t(-1,-1); // invalid/sentinel value to suppress samp_pt prinouts

      rgb_levs_filt_min = float_const_min;
      rgb_levs_filt_rng = 0; 
    }
    virtual p_img_t data_block_to_img( data_block_t const & db ) {
      if( level_adj && (rgb_levs_filt_min == float_const_min) ) { // init level filt on first frame
        data_block_to_img_inner( db ); // note: writes (maybe-garbage) values to frame_buf which are unused, but sets frame levels
        rgb_levs_filt_min = rgb_levs_frame_min;
        float const rgb_levs_frame_rng = rgb_levs_frame_max - rgb_levs_frame_min;
        rgb_levs_filt_rng = rgb_levs_frame_rng;
      }
      return data_block_to_img_inner( db );
    }
    p_img_t data_block_to_img_inner( data_block_t const & db ) {
      if( db.sz != frame_sz_bytes.v ) {
        rt_err( strprintf( "error: can't convert data block to image, had db.sz=%s but frame_sz_bytes.v=%s\n", str(db.sz).c_str(), str(frame_sz_bytes.v).c_str() ) );
      }
      if( db.sz < frame_sz_bytes.v ) { return p_img_t(); } // not enough bytes left for another frame
      u32_pt_t const & img_sz = frame_buf->sz;
      rgb_levs_frame_min = float_const_max;
      rgb_levs_frame_max = float_const_min;
      // copy and convert frame data
      if( 0 ) {
      } else if( img_fmt == "16u-RGGB" ) {
        uint16_t const * const rp_frame = (uint16_t const *)(db.d.get());
        for( uint32_t d = 0; d != 2; ++d ) {
          assert_st( !(frame_sz.d[d]&1) );
          assert_st( !(crop.p[0].d[d]&1) );
          assert_st( !(crop.p[1].d[d]&1) );
        }
        for( uint32_t y = 0; y < img_sz.d[1]; y += 2 ) {
          uint32_t const src_y = crop.p[0].d[1] + y;
          uint16_t const * const src_data = rp_frame + (src_y)*frame_sz.d[0];
          uint16_t const * const src_data_yp1 = rp_frame + (src_y+1)*frame_sz.d[0];
          uint32_t * const dest_data = frame_buf->get_row_addr( y );
          uint32_t * const dest_data_yp1 = frame_buf->get_row_addr( y+1 );
          for( uint32_t x = 0; x < img_sz.d[0]; x += 2 ) {
            uint32_t const src_x = crop.p[0].d[0] + x;
            if( int32_t(x|1) == (samp_pt.d[0]|1) && int32_t(y|1) == (samp_pt.d[1]|1)  ) {
              printf( "\nx,y = %s,%s  --  %s %s\n                   %s %s\n",
                      str(uint32_t(x)).c_str(), str(uint32_t(y)).c_str(),
                      str(uint32_t(src_data[x])).c_str(), str(uint32_t(src_data[x+1])).c_str(),
                      str(uint32_t(src_data_yp1[x])).c_str(), str(uint32_t(src_data_yp1[x+1])).c_str() );
            }
            uint16_t rgb[3]; // as r,g,b
            // set raw values first
            rgb[0] = src_data[src_x+1];
            rgb[1] = (src_data[src_x] + src_data_yp1[src_x+1]) >> 1;
            rgb[2] = src_data_yp1[src_x];            
            if( level_adj ) {
              for( uint32_t d = 0; d != 3; ++d ) {
                min_eq( rgb_levs_frame_min, float(rgb[d]) );
                max_eq( rgb_levs_frame_max, float(rgb[d]) );
                rgb[d] = clamp( (float(rgb[d]) - rgb_levs_filt_min) * (float(uint8_t_const_max) + 1.0f) / rgb_levs_filt_rng, 0.0f, float(uint8_t_const_max) );
              }          
            } else { // hard-coded level adj
              rgb[0] >>= 4;
              rgb[1] >>= 4;
              rgb[2] >>= 4;
            }
            uint32_t const pel = rgba_to_pel(rgb[0],rgb[1],rgb[2]);
            dest_data[x] = pel;  dest_data[x+1] = pel;
            dest_data_yp1[x] = pel;  dest_data_yp1[x+1] = pel;
          }
        }
      } else if( img_fmt == "16u-grey" ) {
        for( uint32_t y = 0; y < img_sz.d[1]; ++y ) {
          uint32_t const src_y = crop.p[0].d[1] + y;
          uint16_t const * const src_data = (uint16_t const *)(db.d.get()) + src_y*frame_sz.d[0];
          uint32_t * const dest_data = frame_buf->get_row_addr( y );
          for( uint32_t x = 0; x < img_sz.d[0]; ++x ) {
            uint32_t const src_x = crop.p[0].d[0] + x;
            uint16_t gv = src_data[src_x];
            if( level_adj ) {
              min_eq( rgb_levs_frame_min, float(gv) );
              max_eq( rgb_levs_frame_max, float(gv) );
              gv = clamp( (float(gv) - rgb_levs_filt_min) * (float(uint8_t_const_max) + 1.0f) / rgb_levs_filt_rng, 0.0f, float(uint8_t_const_max) );
            } else { gv >>= 4; } // hard-coded assume 12-bits
            dest_data[x] = grey_to_pel( gv ); 
          }
        }
      } else if( img_fmt == "32f-grey" ) {
        for( uint32_t y = 0; y < img_sz.d[1]; ++y ) {
          uint32_t const src_y = crop.p[0].d[1] + y;
          float const * const src_data = ((float const *)db.d.get()) + (src_y*frame_sz.d[0]);
          uint32_t * const dest_data = frame_buf->get_row_addr( y );
          for( uint32_t x = 0; x < img_sz.d[0]; ++x ) {
            uint32_t const src_x = crop.p[0].d[0] + x;
            float gv = src_data[src_x];
            if( verbose ) { printf( " %s", str(gv).c_str() ); }
            if( level_adj ) {
              min_eq( rgb_levs_frame_min, gv );
              max_eq( rgb_levs_frame_max, gv );
              if( level_adj_log_for_float_data ) {
                gv = logf(gv - rgb_levs_filt_min + 1.0f) * (float(uint8_t_const_max) + 1.0f) / logf( rgb_levs_filt_rng + 1.0f );
              } else { gv = (gv - rgb_levs_filt_min) * (float(uint8_t_const_max) + 1.0f) / rgb_levs_filt_rng; }
              clamp_eq( gv, 0.0f, float(uint8_t_const_max) );
            } else { } // hard-coded assume already in [0,256)
            if( verbose ) { printf( "=%s", str(gv).c_str() ); }
            dest_data[x] = grey_to_pel( gv ); 
          }
          if( verbose ) { printf( "\n" ); }
        }
      } else { rt_err( "can't decode frame: unknown img_fmt: " + img_fmt ); }
      // update level filter
      if( level_adj ) {
        float const rgb_levs_frame_rng = rgb_levs_frame_max - rgb_levs_frame_min;
        if( verbose ) {
         printf( "rgb_levs_filt_rng=%s rgb_levs_filt_min=%s\n", str(rgb_levs_filt_rng).c_str(), str(rgb_levs_filt_min).c_str() );
         printf( "rgb_levs_frame_min=%s rgb_levs_frame_max=%s\n", str(rgb_levs_frame_min).c_str(), str(rgb_levs_frame_max).c_str() );
        }
        rgb_levs_filt_rng *= level_filt_alpha;
        rgb_levs_filt_rng += (1.0 - level_filt_alpha)*rgb_levs_frame_rng;
        rgb_levs_filt_min *= level_filt_alpha;
        rgb_levs_filt_min += (1.0 - level_filt_alpha)*rgb_levs_frame_min;
      }
      return frame_buf;
    }
  };

  struct data_to_img_null_t : virtual public nesi, public data_to_img_t // NESI(help="consume data blocks and return nothing (null images)",
                           // bases=["data_to_img_t"], type_id="null")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")

    virtual void set_samp_pt( i32_pt_t const & samp_pt_ ) { }    
    virtual void data_to_img_init( nesi_init_arg_t * const nia ) { }
    virtual p_img_t data_block_to_img( data_block_t const & db ) {
      if( verbose ) { printf( "data_to_img_null: db.sz=%s db.timestamp_ns=%s\n",
                              str(db.sz).c_str(), str(db.timestamp_ns).c_str() ); }
      return p_img_t();
    }
    
  };

  struct data_to_img_lidar_t : virtual public nesi, public data_to_img_t // NESI(help="consume velodyne lidar data blocks (packets) and return nothing (null images)",
                           // bases=["data_to_img_t"], type_id="lidar")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")

    virtual void set_samp_pt( i32_pt_t const & samp_pt_ ) { }    
    virtual void data_to_img_init( nesi_init_arg_t * const nia ) {
    }
    virtual p_img_t data_block_to_img( data_block_t const & db ) {
      if( verbose ) { printf( "data_to_img_lidar: db.sz=%s db.timestamp_ns=%s\n",
                              str(db.sz).c_str(), str(db.timestamp_ns).c_str() ); }
      return p_img_t();
    }
    
  };

#include"gen/data-to-img.H.nesi_gen.cc"
#include"gen/data-to-img.cc.nesi_gen.cc"

}
