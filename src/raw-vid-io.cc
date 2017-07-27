// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"img_io.H"
#include"has_main.H"
#include"str_util.H"
#include"data_stream.H"
#include<boost/iostreams/device/mapped_file.hpp>
#include<boost/endian/conversion.hpp>

#include<locale>
#include<codecvt>

template <typename T> std::string to_utf8(const std::basic_string<T, std::char_traits<T>, std::allocator<T>>& source) {
  std::string result;
  std::wstring_convert<std::codecvt_utf8_utf16<T>, T> convertor;
  result = convertor.to_bytes(source);
  return result;
}

namespace boda 
{
  
  struct data_stream_base_t : virtual public nesi, public data_stream_t // NESI(help="parse data stream (dumpvideo/qt) into data blocks",
                             // bases=["data_stream_t"], type_id="base")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    filename_t fn; //NESI(default="vid.raw",help="input raw video filename")
    string read_mode; //NESI(default="dumpvideo",help="file reading mode: dumpvideo=mplayer video dump format; qt=qt-style binary encapsulation")

    // for debugging / skipping:
    uint64_t start_block; //NESI(default=0,help="start at this block")
    uint64_t skip_blocks; //NESI(default=0,help="drop/skip this many blocks after each returned block (default 0, no skipped/dropped blocks)")
    uint64_t num_to_read; //NESI(default=0,help="read this many records; zero for unlimited")

    // internal state:
    bool need_endian_reverse;
    p_mapped_file_source fn_map;
    uint64_t fn_map_pos;
    uint64_t tot_num_read; // num blocks read so far

    // low-level stream reading
    bool can_read( uint64_t const & sz ) { return (sz+fn_map_pos) < fn_map->size(); }
    void check_can_read( uint64_t const & sz ) { if( !can_read(sz) ) { rt_err( "unexpected end of stream" ); } }
    
    template< typename T > void read_val( T & v ) {
      check_can_read( sizeof( T ) );
      v = *reinterpret_cast< T const * >((uint8_t const *)fn_map->data() + fn_map_pos);
      if( need_endian_reverse ) { boost::endian::endian_reverse_inplace(v); }
      fn_map_pos += sizeof(T);
    }

    void read_val( string & v ) {
      uint32_t str_len;
      read_val( str_len );
      if( str_len == uint32_t_const_max ) { rt_err( "unexpected null string" ); }
      //printf( "str_len=%s\n", str(str_len).c_str() );
      assert_st( !(str_len & 1) ); // should be utf-16 string, so must be even # of bytes      
      std::u16string u16s;
      for( uint32_t i = 0; i != str_len; i += 2 ) {
        uint16_t v;
        read_val( v );
        u16s.push_back( v );
      }
      v = to_utf8( u16s );
    }

    // note: does not read/fill-in timestamp_ns field of data_block_t, just size and data/pointer
    void read_val( data_block_t & v ) {
      uint32_t v_len;
      read_val( v_len );
      if( v_len == uint32_t_const_max ) { rt_err( "unexpected null byte array" ); }
      check_can_read( v_len );
      v.sz = v_len;
      v.d.reset( (uint8_t *)fn_map->data() + fn_map_pos, null_deleter<uint8_t>() ); // borrow pointer
      fn_map_pos += v_len;
    }

    // block-level reading
    
    virtual string get_pos_info_str( void ) { return strprintf( "fn_map_pos=%s tot_num_read=%s", str(fn_map_pos).c_str(), str(tot_num_read).c_str() ); }

    virtual data_block_t read_next_block( void ) {
      if( num_to_read && (tot_num_read == num_to_read) ) { return data_block_t(); }
      data_block_t ret = read_next_block_inner();
      for( uint32_t i = 0; i != skip_blocks; ++i ) { read_next_block_inner(); } // skip blocks if requested
      return ret;
    }
    
    data_block_t read_next_block_inner( void ) {
      data_block_t ret;
      if( 0 ) {}
      else if( read_mode == "dumpvideo" ) {
        uint32_t block_sz;
        if( !can_read( sizeof( block_sz ) ) ) { return ret; } // not enough bytes left for another block
        read_val( block_sz );
        check_can_read( block_sz );
        ret.sz = block_sz;
        //ret.timestamp_ns = ???; // FIXME: need nested stream
        ret.d.reset( (uint8_t *)fn_map->data() + fn_map_pos, null_deleter<uint8_t>() ); // borrow pointer
        fn_map_pos += block_sz;
      } else if( read_mode == "qt" ) {
        if( !can_read( sizeof( ret.timestamp_ns ) ) ) { return ret; } // not enough bytes left for another block
        read_val( ret.timestamp_ns );
        read_val( ret );
      } else { rt_err( "unknown read_mode: " + read_mode ); }
      ++tot_num_read;
      if( verbose ) { printf( "ret.sz=%s ret.timestamp_ns=%s\n", str(ret.sz).c_str(), str(ret.timestamp_ns).c_str() ); }
      return ret;
    }

    // init/setup
    
    void raw_vid_init_dumpvideo( void ) {
      need_endian_reverse = 0;
    }

    void raw_vid_init_qt( void ) {
      need_endian_reverse = 1; // assume stream is big endian, and native is little endian. could check this ...

      uint32_t ver;
      read_val( ver );
      string tag;
      read_val( tag );
      data_block_t header;
      read_val( header );
      uint64_t timestamp_off;
      read_val( timestamp_off );
      uint64_t chunk_off;
      read_val( chunk_off );
      uint64_t duration_ns;
      read_val( duration_ns );
      printf( "qt stream header: ver=%s tag=%s header.size()=%s timestamp_off=%s chunk_off=%s duration_ns=%s\n",
              str(ver).c_str(), str(tag).c_str(), str(header.sz).c_str(), str(timestamp_off).c_str(),
              str(chunk_off).c_str(), str(duration_ns).c_str() );
      
    }
    
    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      fn_map = map_file_ro( fn );
      fn_map_pos = 0;
      tot_num_read = 0;
      if( 0 ) {}
      else if( read_mode == "dumpvideo" ) { raw_vid_init_dumpvideo(); }
      else if( read_mode == "qt" ) { raw_vid_init_qt(); }
      else { rt_err( "unknown read_mode: " + read_mode ); }
      // skip to start block
      for( uint32_t i = 0; i != start_block; ++i ) { read_next_block_inner(); }
    }
    
    void main( nesi_init_arg_t * nia ) { 
      data_stream_init( nia );
      while( read_next_block().d.get() ) { }
    }

  };
  struct data_stream_base_t; typedef shared_ptr< data_stream_base_t > p_data_stream_base_t; 

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

    
  struct data_stream_base_t; typedef shared_ptr< data_stream_base_t > p_data_stream_base_t; 

#include"gen/raw-vid-io.cc.nesi_gen.cc"

}
