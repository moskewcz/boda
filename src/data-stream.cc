// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"img_io.H"
#include"has_main.H"
#include"str_util.H"
#include"data-stream.H"
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

    uint32_t text_tsfix; //NESI(default=0,help="text time-stamp-field-index; when read_mode==text, use the N'th field as a decimal timestamp in seconds (with fractional part).")

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
    void check_can_read( uint64_t const & sz ) {
      if( !can_read(sz) ) {
        rt_err( strprintf( "unexpected end of stream trying to read sz=%s bytes at fn_map_pos=%s\n",
                           str(sz).c_str(), str(fn_map_pos).c_str() ) );
      }
    }
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

    // read a newline-terminated text line as a block. notes: block includes newline if any found; will return rest of
    // file if no newline; doesn't set timestamp field of block.
    void read_line_as_block( data_block_t & v ) {
      v.d.reset( (uint8_t *)fn_map->data() + fn_map_pos, null_deleter<uint8_t>() ); // borrow pointer
      uint8_t *lb = (uint8_t *)fn_map->data() + fn_map_pos;
      uint8_t *le = lb;
      uint8_t *de = (uint8_t *)fn_map->data() + fn_map->size();
      if( !(le < de) ) { rt_err( "unexpected end of file when trying to read a line of text: started read at EOF." ); }
      while( le != de ) {
        if( *le == '\r' ) { // DOS "\r\n" newline
          ++le;
          if( le == de ) { rt_err( "unexpected end of file when trying to read a line of text: EOF after \\r." ); }
          if( *le != '\n' ) {
            rt_err( "error reading text string: expected \\n after \\r, but got char with decimal val = " +
                    str(uint32_t(*le)) ); }
          ++le;
          break;
        }
        if( *le == '\n' ) { ++le; break; } // regular unix "\n" newline
        ++le; // non-newline char, add to string
      }
      v.sz = le - lb;
      fn_map_pos += v.sz; // consume str
    }

    // set timestamp from field of text line stored in block
    void set_timestamp_from_text_line( data_block_t & v ) {
      string line( v.d.get(), v.d.get()+v.sz );
      vect_string parts = split( line, ' ' );
      if( !( text_tsfix < parts.size() ) ) {
        rt_err( strprintf( "can't parse timestamp from text_tsfix=%s; line had parts.size()=%s; full line=%s\n", str(text_tsfix).c_str(), str(parts.size()).c_str(), str(line).c_str() ) );
      }
      //if( verbose ) { printf( "parts[text_tsfix]=%s\n", str(parts[text_tsfix]).c_str() ); }
      double const ts_d_ns = lc_str_d( parts[text_tsfix] ) * 1e9;
      v.timestamp_ns = lround(ts_d_ns);
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
      if( num_to_read && (tot_num_read >= num_to_read) ) { return data_block_t(); }
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
        if( fn_map_pos == timestamp_off ) {
#if 0
          printf( "at timestamp_off: fn_map_pos=%s\n", str(fn_map_pos).c_str() );
          uint8_t ch;
          while( can_read( sizeof(ch) ) ) {
            read_val(ch);
            printf( "ch=%s ch=%s\n", str(uint32_t(ch)).c_str(), str(ch).c_str() );
          }
#endif     
          return ret;
        }
        if( !can_read( sizeof( ret.timestamp_ns ) ) ) { return ret; } // not enough bytes left for another block
        read_val( ret.timestamp_ns );
        if( !can_read( sizeof( uint32_t ) ) ) { rt_err( "qt stream: read timestamp, but not enough data left to read payload size" ); }
        read_val( ret );
      } else if( read_mode == "text" ) {
        if( !can_read( 1 ) ) { return ret; } // not enough bytes left for another block
        read_line_as_block( ret );
        set_timestamp_from_text_line( ret );
      } else { rt_err( "unknown read_mode: " + read_mode ); }
      
      ++tot_num_read;
      if( verbose ) { printf( "%s ret.sz=%s ret.timestamp_ns=%s\n", read_mode.c_str(), str(ret.sz).c_str(), str(ret.timestamp_ns).c_str() ); }
      return ret;
    }

    // init/setup
    
    void data_stream_init_dumpvideo( void ) {
      need_endian_reverse = 0;
    }

    // for qt stream
    uint64_t timestamp_off;
    uint64_t chunk_off;

    void data_stream_init_qt( void ) {
      need_endian_reverse = 1; // assume stream is big endian, and native is little endian. could check this ...

      uint32_t ver;
      read_val( ver );
      string tag;
      read_val( tag );
      data_block_t header;
      read_val( header );
      read_val( timestamp_off );
      read_val( chunk_off );
      uint64_t duration_ns;
      read_val( duration_ns );
      printf( "  qt stream header: ver=%s tag=%s header.size()=%s timestamp_off=%s chunk_off=%s duration_ns=%s\n",
              str(ver).c_str(), str(tag).c_str(), str(header.sz).c_str(), str(timestamp_off).c_str(),
              str(chunk_off).c_str(), str(duration_ns).c_str() );
      if( fn_map->size() > timestamp_off ) {
        printf( "   !! warning: (fn_map->size() - timestamp_off)=%s bytes at end of file will be ignored\n",
                str((fn_map->size() - timestamp_off)).c_str() );
      }
    }

    void data_stream_init_text( void ) {
      data_block_t header;
      read_line_as_block( header );
      printf( "  text stream header.sz=%s\n", str(header.sz).c_str() );
    }

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      printf( "data_stream_init(): fn.exp=%s read_mode=%s start_block=%s skip_blocks=%s\n", str(fn.exp).c_str(), str(read_mode).c_str(), str(start_block).c_str(), str(skip_blocks).c_str() );
      fn_map = map_file_ro( fn );
      fn_map_pos = 0;
      tot_num_read = 0;
      if( 0 ) {}
      else if( read_mode == "dumpvideo" ) { data_stream_init_dumpvideo(); }
      else if( read_mode == "qt" ) { data_stream_init_qt(); }
      else if( read_mode == "text" ) { data_stream_init_text(); }
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

  struct block_info_t {
    uint16_t block_id;
    uint16_t rot_pos;
  } __attribute__((packed));

  struct laser_info_t {
    uint16_t distance;
    uint8_t intensity;
  } __attribute__((packed));

  struct data_stream_velodyne_t : virtual public nesi, public data_stream_t // NESI(help="parse data stream (velodyne) into per-full-revolution data blocks by merging across packets",
                             // bases=["data_stream_t"], type_id="velodyne")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    p_data_stream_base_t vps; //NESI(req=1,help="underlying velodyne packet stream")

    // FIXME: somehow factor out shared start_block, skip_blocks, num_to_read handling
    // for debugging / skipping:
    uint64_t start_block; //NESI(default=0,help="start at this block")
    uint64_t skip_blocks; //NESI(default=0,help="drop/skip this many blocks after each returned block (default 0, no skipped/dropped blocks)")
    uint64_t num_to_read; //NESI(default=0,help="read this many records; zero for unlimited")

    uint32_t fbs_per_packet; //NESI(default="12",help="firing blocks per packet")
    uint32_t beams_per_fb; //NESI(default="32",help="beams per firing block")
    uint32_t status_bytes; //NESI(default="6",help="bytes of status at end of block")

    uint32_t fb_sz;
    uint32_t packet_sz;
    uint16_t last_rot;
    
    // internal state:
    uint64_t tot_num_read; // num blocks read so far

    virtual string get_pos_info_str( void ) { return strprintf( "tot_num_read=%s vps->tot_num_read=%s", str(tot_num_read).c_str(), str(vps->tot_num_read).c_str() ); }

    virtual data_block_t read_next_block( void ) {
      if( num_to_read && (tot_num_read >= num_to_read) ) { return data_block_t(); }
      data_block_t ret = read_next_block_inner();
      for( uint32_t i = 0; i != skip_blocks; ++i ) { read_next_block_inner(); } // skip blocks if requested
      return ret;
    }
    
    data_block_t read_next_block_inner( void ) {
      bool packet_is_rot_start = 0;
      data_block_t db;
      while( !packet_is_rot_start ) {
        db = vps->read_next_block();
        if( !db.d.get() ) { return db; } // not enough data for another frame, give up
        if( db.sz != packet_sz ) { rt_err(
            strprintf( "lidar decode expected packet_sz=%s but got block with dv.sz=%s",
                       str(packet_sz).c_str(), str(db.sz).c_str() ) ); }
        if( verbose ) { printf( "data_to_img_null: db.sz=%s db.timestamp_ns=%s\n",
                                str(db.sz).c_str(), str(db.timestamp_ns).c_str() ); }
        for( uint32_t i = 0; i != fbs_per_packet; ++i ) {
          block_info_t bi = *(block_info_t *)(db.d.get()+fb_sz*i);
          if( verbose ) {
            printf( "bi.block_id=%hx bi.rot_pos=%hu\n", bi.block_id, bi.rot_pos );
          }
          if( bi.rot_pos < last_rot ) { packet_is_rot_start = 1; }
          last_rot = bi.rot_pos;
        }
      }
      // last packet was a frame start, so return it. FIXME: obv. this drops lots'o'data, but should emit one packet per
      // rotation, which is all we want for now.
      ++tot_num_read;
      if( verbose ) { printf( "velodyne ret.sz=%s ret.timestamp_ns=%s\n", str(db.sz).c_str(), str(db.timestamp_ns).c_str() ); }
      return db;
    }

    // init/setup

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      printf( "data_stream_init(): mode=velodyne start_block=%s skip_blocks=%s\n",
              str(start_block).c_str(), str(skip_blocks).c_str() );
      tot_num_read = 0;

      // setup internal state
      fb_sz = sizeof( block_info_t ) + beams_per_fb * sizeof( laser_info_t );
      packet_sz = fbs_per_packet * fb_sz + status_bytes;
      last_rot = 0;
        
      // override/clear nested skip/etc params. FIXME: a bit ugly; use a wrapper to both factor out and fix this?
      vps->start_block = 0;
      vps->skip_blocks = 0;
      vps->num_to_read = 0;
      vps->data_stream_init( nia );
      // skip to start block
      for( uint32_t i = 0; i != start_block; ++i ) { read_next_block_inner(); }
    }
    
    void main( nesi_init_arg_t * nia ) { 
      data_stream_init( nia );
      while( read_next_block().d.get() ) { }
    }

  };

  
  struct scan_data_stream_t : virtual public nesi, public has_main_t // NESI(
                              // help="scan N data streams one-by-one, and print total number of blocks read for each.",
                              // bases=["has_main_t"], type_id="scan-data-stream")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    vect_p_data_stream_t stream; //NESI(help="data stream to read images from")

    void main( nesi_init_arg_t * nia ) {
      for( uint32_t i = 0; i != stream.size(); ++i ) {
        stream[i]->data_stream_init( nia );
        while( stream[i]->read_next_block().d.get() ) { }
        printf( "stream[%s]->get_pos_info_str()=%s\n", str(i).c_str(), str(stream[i]->get_pos_info_str()).c_str() );
      }
    }

  };

#include"gen/data-stream.H.nesi_gen.cc"
#include"gen/data-stream.cc.nesi_gen.cc"

}
