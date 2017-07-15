// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _RAW_VID_IO_H_
#define _RAW_VID_IO_H_

#include"boda_base.H"
#include"geom_prim.H"
#include"img_io.H"
#include"has_main.H"
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

  struct raw_vid_io_t : virtual public nesi, public has_main_t // NESI(help="parse raw video into frames",
                           // bases=["has_main_t"], type_id="parse-raw-vid")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t fn; //NESI(default="vid.raw",help="input raw video filename")
    u32_pt_t frame_sz; //NESI(default="10 5",help="X/Y frame size")
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
    
    p_img_t read_next_frame_raw( void ) {
      uint64_t const frame_sz_bytes = frame_sz.dims_prod() * bytes_per_pel; // assumes 2 bytes per pel
      if( !( (frame_sz_bytes + fn_map_pos) < fn_map->size() ) ) { return p_img_t(); } // not enough bytes left for another frame
      uint16_t const * const rp_frame = (uint16_t const *)((uint8_t const *)fn_map->data() + fn_map_pos);
      // copy and convert frame data
      for( uint32_t y = 0; y < frame_sz.d[1]; ++y ) {
        uint32_t * const dest_data = frame_buf->get_row_addr( y );
        uint16_t const * const src_data = rp_frame + y*frame_sz.d[0];
        for( uint32_t x = 0; x < frame_sz.d[0]; ++x ) { dest_data[x] = grey_to_pel( src_data[x] >> 4 ); }
      }
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
      printf( "timestamp_ns=%s qt_frame_buf.size()=%s\n", str(timestamp_ns).c_str(), str(qt_frame_buf.size()).c_str() );
      if( qt_frame_buf.size() != frame_sz.dims_prod() * 4 ) {
        printf( "don't know how to convert qt_frame_buf.size()=%s to image with frame_sz.dims_prod()=%s (currenly only handling 4 bytes/pixel)\n", str(qt_frame_buf.size()).c_str(), str(frame_sz.dims_prod()).c_str() );
      }
      for( uint32_t y = 0; y < frame_sz.d[1]; ++y ) {
        uint32_t * const dest_data = frame_buf->get_row_addr( y );
        float const * const src_data = ((float const *)&qt_frame_buf[0]) + (y*frame_sz.d[0]);
        for( uint32_t x = 0; x < frame_sz.d[0]; ++x ) {
          dest_data[x] = grey_to_pel( src_data[x] );
          if( x == 128 && y == 32 ) {
            printf( "src_data[x]=%s\n", str(src_data[x]).c_str() );
          }
        }
      }
      
      return frame_buf;
    }

    p_img_t read_next_frame( void ) {
      if( num_to_read && (tot_num_read == num_to_read) ) { return p_img_t(); }
      if( 0 ) {}
      else if( read_mode == "raw" ) { return read_next_frame_raw(); }
      else if( read_mode == "qt" ) { return read_next_frame_qt(); }
      else { rt_err( "unknown read_mode: " + read_mode ); }
    }

    void raw_vid_init_raw( void ) {
      frame_stride = frame_sz.dims_prod()*bytes_per_pel + padding;
      fn_map_pos = start_frame * frame_stride;
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
    
    void raw_vid_init( void ) {
      tot_num_read = 0;
      fn_map = map_file_ro( fn );
      frame_buf = make_shared< img_t >();
      frame_buf->set_sz_and_alloc_pels( frame_sz );

      if( 0 ) {}
      else if( read_mode == "raw" ) { raw_vid_init_raw(); }
      else if( read_mode == "qt" ) { raw_vid_init_qt(); }
      else { rt_err( "unknown read_mode: " + read_mode ); }

    }
    
    void main( nesi_init_arg_t * nia ) { 
      raw_vid_init();
      while( read_next_frame() ) { }
    }

  };

  uint64_t score_batch( p_nda_float_t const & out_batch, vect_uint32_t const & batch_labels_gt );

}


#endif /* _RAW_VID_IO_H_ */
