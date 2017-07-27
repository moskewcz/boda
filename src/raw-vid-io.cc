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
    string img_fmt; //NESI(req=1,help="image format; valid values '16u-grey', '32f-grey', '16u-RGGB' ")

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
      frame_buf->set_sz_and_alloc_pels( frame_sz );
      samp_pt = i32_pt_t(-1,-1); // invalid/sentinel value to suppress samp_pt prinouts
    }

    virtual p_img_t data_block_to_img( data_block_t const & db ) {
      if( db.sz < frame_sz_bytes.v ) { return p_img_t(); } // not enough bytes left for another frame
      // copy and convert frame data
      if( 0 ) {
      } else if( img_fmt == "16u-RGGB" ) {
        uint16_t const * const rp_frame = (uint16_t const *)(db.d.get());
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
      } else if( img_fmt == "16u-grey" ) {
        uint16_t const * const rp_frame = (uint16_t const *)(db.d.get());
        for( uint32_t y = 0; y < frame_sz.d[1]; ++y ) {
          uint32_t * const dest_data = frame_buf->get_row_addr( y );
          uint16_t const * const src_data = rp_frame + y*frame_sz.d[0];
          for( uint32_t x = 0; x < frame_sz.d[0]; ++x ) { dest_data[x] = grey_to_pel( src_data[x] >> 4 ); }
        }
      } else if( img_fmt == "32f-grey" ) {
        for( uint32_t y = 0; y < frame_sz.d[1]; ++y ) {
          uint32_t * const dest_data = frame_buf->get_row_addr( y );
          float const * const src_data = ((float const *)db.d.get()) + (y*frame_sz.d[0]);
          for( uint32_t x = 0; x < frame_sz.d[0]; ++x ) {
            dest_data[x] = grey_to_pel( src_data[x] );
            if( verbose ) { if( x == 128 && y == 32 ) { printf( "src_data[x]=%s\n", str(src_data[x]).c_str() ); } }
          }
        }
      } else { rt_err( "can't decode frame: unknown img_fmt: " + img_fmt ); }
      return frame_buf;
    }
  };

    
  struct data_stream_base_t; typedef shared_ptr< data_stream_base_t > p_data_stream_base_t; 

#include"gen/raw-vid-io.cc.nesi_gen.cc"

}
