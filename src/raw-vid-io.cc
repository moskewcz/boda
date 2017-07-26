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
  
  struct raw_vid_io_t : virtual public nesi, public data_stream_t // NESI(help="parse raw video into frames",
                           // bases=["data_stream_t"], type_id="raw-vid-io")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    filename_t fn; //NESI(default="vid.raw",help="input raw video filename")
    u32_pt_t frame_sz; //NESI(default="10 5",help="X/Y frame size")
    string mosiac; //NESI(default="none",help="mosiac mode; valid values 'none' (gryyscale) or 'RGGB'")
    uint32_t padding; //NESI(default="0",help="frame padding")
    uint32_t bytes_per_pel; //NESI(default="2",help="bytes per pel")

    uint64_t start_frame; //NESI(default=0,help="start at this frame")
    uint64_t skip_frames; //NESI(default=0,help="drop/skip this many frames after each returned frame (default 0, no skipped/dropped frames)")

    uint64_t num_to_read; //NESI(default=0,help="read this many records; zero for unlimited")

    string read_mode; //NESI(default="raw",help="file reading mode: raw=raw frames; qt=qt-style binary encapsulation")
    
    uint64_t tot_num_read; // num frames read so far
    p_mapped_file_source fn_map;
    uint64_t fn_map_pos;
    uint32_t frame_stride;

    p_img_t frame_buf;
    
    bool need_endian_reverse;
    vect_uint8_t qt_frame_buf;

    i32_pt_t samp_pt;

    virtual void set_samp_pt( i32_pt_t const & samp_pt_ ) { samp_pt = samp_pt_; }
    
    virtual string get_pos_info_str( void ) { return strprintf( "fn_map_pos=%s tot_num_read=%s", str(fn_map_pos).c_str(), str(tot_num_read).c_str() ); }

    p_img_t read_next_frame_raw( void ) {
      uint64_t const frame_sz_bytes = frame_sz.dims_prod() * bytes_per_pel; // assumes 2 bytes per pel
      if( !( (frame_sz_bytes + fn_map_pos) < fn_map->size() ) ) { return p_img_t(); } // not enough bytes left for another frame
      uint16_t const * const rp_frame = (uint16_t const *)((uint8_t const *)fn_map->data() + fn_map_pos);
      // copy and convert frame data
      if( 0 ) {
      } else if( mosiac == "RGGB" ) {
        assert_st( !(frame_sz.d[1]&1) );
        for( uint32_t y = 0; y < frame_sz.d[1]; y += 2 ) {
          uint32_t * const dest_data = frame_buf->get_row_addr( y );
          uint32_t * const dest_data_yp1 = frame_buf->get_row_addr( y+1 );
          uint16_t const * const src_data = rp_frame + y*frame_sz.d[0];
          uint16_t const * const src_data_yp1 = rp_frame + (y+1)*frame_sz.d[0];
          for( uint32_t x = 0; x < frame_sz.d[0]; x += 2 ) {
            if( int32_t(x|1) == (samp_pt.d[0]|1) && int32_t(y|1) == (samp_pt.d[1]|1)  ) {
              printf( "\nx,y = %s,%s  --  %s %s\n                   %s %s\n",
                      str(uint32_t(x)).c_str(), str(uint32_t(y)).c_str(),
                      str(uint32_t(src_data[x])).c_str(), str(uint32_t(src_data[x+1])).c_str(),
                      str(uint32_t(src_data_yp1[x])).c_str(), str(uint32_t(src_data_yp1[x+1])).c_str() );
            }
            //uint8_t const r = src_data[x+1] >> 4;
            //uint8_t const g = (src_data[x] + src_data_yp1[x+1]) >> 5;
            //uint8_t const b = src_data_yp1[x] >> 4;
            uint16_t const lev = 2048;
            uint8_t const r = (std::max(src_data[x+1],lev)-lev) >> 3;
            uint8_t const g = ((std::max(src_data[x],lev)-lev) + (std::max(src_data_yp1[x+1],lev)-lev) ) >> 4;
            uint8_t const b = (std::max(src_data_yp1[x],lev)-lev) >> 3;

            uint32_t const pel = rgba_to_pel(r,g,b);
            dest_data[x] = pel;  dest_data[x+1] = pel;
            dest_data_yp1[x] = pel;  dest_data_yp1[x+1] = pel;
          }
        }
      } else if( mosiac == "none" ) {
        for( uint32_t y = 0; y < frame_sz.d[1]; ++y ) {
          uint32_t * const dest_data = frame_buf->get_row_addr( y );
          uint16_t const * const src_data = rp_frame + y*frame_sz.d[0];
          for( uint32_t x = 0; x < frame_sz.d[0]; ++x ) { dest_data[x] = grey_to_pel( src_data[x] >> 4 ); }
        }
      } else { rt_err( "unknown mosiac: " + mosiac ); }

      //frame_buf->fill_with_pel( grey_to_pel( (10 * tot_num_read) & 0xff ) );
      ++tot_num_read;
      fn_map_pos += frame_stride; // move to next raw frame
      for( uint32_t i = 0; i != skip_frames; ++i ) { fn_map_pos += frame_stride; } // skip frames if requested
      return frame_buf;
    }

    p_img_t read_next_frame_qt( void ) {
      uint64_t timestamp_ns;
      if( !can_read( sizeof( timestamp_ns ) ) ) { return p_img_t(); }
      read_val( timestamp_ns );
      read_val( qt_frame_buf );
      if( verbose ) { printf( "timestamp_ns=%s qt_frame_buf.size()=%s\n", str(timestamp_ns).c_str(), str(qt_frame_buf.size()).c_str() ); }
      if( qt_frame_buf.size() != frame_sz.dims_prod() * 4 ) {
        printf( "don't know how to convert qt_frame_buf.size()=%s to image with frame_sz.dims_prod()=%s (currenly only handling 4 bytes/pixel)\n", str(qt_frame_buf.size()).c_str(), str(frame_sz.dims_prod()).c_str() );
      }
      for( uint32_t y = 0; y < frame_sz.d[1]; ++y ) {
        uint32_t * const dest_data = frame_buf->get_row_addr( y );
        float const * const src_data = ((float const *)&qt_frame_buf[0]) + (y*frame_sz.d[0]);
        for( uint32_t x = 0; x < frame_sz.d[0]; ++x ) {
          dest_data[x] = grey_to_pel( src_data[x] );
          if( verbose ) { if( x == 128 && y == 32 ) { printf( "src_data[x]=%s\n", str(src_data[x]).c_str() ); } }
        }
      }
      
      return frame_buf;
    }

    virtual p_img_t read_next_frame( void ) {
      if( num_to_read && (tot_num_read == num_to_read) ) { return p_img_t(); }
      if( 0 ) {}
      else if( read_mode == "raw" ) { return read_next_frame_raw(); }
      else if( read_mode == "qt" ) { return read_next_frame_qt(); }
      else { rt_err( "unknown read_mode: " + read_mode ); }
    }

    void raw_vid_init_raw( void ) {
      frame_stride = frame_sz.dims_prod()*bytes_per_pel + padding;
      fn_map_pos = start_frame * frame_stride + padding;
    }

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

    void read_val( vect_uint8_t & v ) {
      uint32_t v_len;
      read_val( v_len );
      if( v_len == uint32_t_const_max ) { rt_err( "unexpected null byte array" ); }
      v.resize( v_len );
      check_can_read( v_len );
      uint8_t const * const src = (uint8_t const *)fn_map->data() + fn_map_pos;
      copy( src, src + v_len, v.begin() );
      fn_map_pos += v_len;
    }

    void raw_vid_init_qt( void ) {
      need_endian_reverse = 1; // asume stream is big endian, and native is little endian. could check this ...
      frame_stride = 0; // no fixed frame stride
      fn_map_pos = 0; // FIXME: skip to start frame

      uint32_t ver;
      read_val( ver );
      string tag;
      read_val( tag );
      vect_uint8_t header;
      read_val( header );
      uint64_t timestamp_off;
      read_val( timestamp_off );
      uint64_t chunk_off;
      read_val( chunk_off );
      uint64_t duration_ns;
      read_val( duration_ns );
      printf( "ver=%s tag=%s header.size()=%s timestamp_off=%s chunk_off=%s duration_ns=%s\n", str(ver).c_str(), str(tag).c_str(), str(header.size()).c_str(), str(timestamp_off).c_str(), str(chunk_off).c_str(), str(duration_ns).c_str() );
      
    }
    
    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      tot_num_read = 0;
      fn_map = map_file_ro( fn );
      frame_buf = make_shared< img_t >();
      frame_buf->set_sz_and_alloc_pels( frame_sz );
      samp_pt = i32_pt_t(-1,-1); // invalid/sentinel value to suppress samp_pt prinouts
      
      if( 0 ) {}
      else if( read_mode == "raw" ) { raw_vid_init_raw(); }
      else if( read_mode == "qt" ) { raw_vid_init_qt(); }
      else { rt_err( "unknown read_mode: " + read_mode ); }

    }
    
    void main( nesi_init_arg_t * nia ) { 
      data_stream_init( nia );
      while( read_next_frame() ) { }
    }

  };

  typedef shared_ptr< raw_vid_io_t > p_raw_vid_io_t; 

  
  uint64_t score_batch( p_nda_float_t const & out_batch, vect_uint32_t const & batch_labels_gt );

#include"gen/raw-vid-io.cc.nesi_gen.cc"

}
