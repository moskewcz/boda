// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef MAPPED_FILE_UTIL_H
#define MAPPED_FILE_UTIL_H

#include"boda_base.H"
#include<boost/iostreams/device/mapped_file.hpp>
#include<boost/endian/conversion.hpp>

#include<locale>
#include<codecvt>


namespace boda
{

  template <typename T> std::string to_utf8(const std::basic_string<T, std::char_traits<T>, std::allocator<T>>& source) {
    std::string result;
    std::wstring_convert<std::codecvt_utf8_utf16<T>, T> convertor;
    result = convertor.to_bytes(source);
    return result;
  }

  struct mapped_file_stream_reader {
    p_mapped_file_source fn_map;
    uint64_t pos;
    bool need_endian_reverse;
    mapped_file_stream_reader( void ) : pos(0), need_endian_reverse(0) { }
    void init( filename_t const & fn ) {
      fn_map = map_file_ro( fn );
      pos = 0;
      need_endian_reverse = 0;
    }
    // low-level stream reading
    uint64_t size( void ) { return fn_map->size(); }
    bool can_read( uint64_t const & sz ) { return (sz+pos) < size(); }
    void check_can_read( uint64_t const & sz ) {
      if( !can_read(sz) ) {
        rt_err( strprintf( "unexpected end of stream trying to read sz=%s bytes at pos=%s\n",
                           str(sz).c_str(), str(pos).c_str() ) );
      }
    }

    // consume a block of bytes from the stream and return a borrowed pointer to it.
    p_uint8_t consume_borrowed_raw_block( uint64_t const block_sz ) {
      check_can_read( block_sz );
      p_uint8_t ret( (uint8_t *)fn_map->data() + pos, null_deleter<uint8_t>() ); // borrow pointer
      pos += block_sz;
      return ret;
    }
    // like above, but return result in a data_block_t with default/not-set timestamp
    data_block_t consume_borrowed_block( uint64_t const block_sz ) {
      data_block_t ret;
      ret.sz = block_sz;
      ret.d = consume_borrowed_raw_block( block_sz );
      return ret;
    }


    template< typename T > void read_val( T & v ) {
      check_can_read( sizeof( T ) );
      v = *reinterpret_cast< T const * >((uint8_t const *)fn_map->data() + pos);
      if( need_endian_reverse ) { boost::endian::endian_reverse_inplace(v); }
      pos += sizeof(T);
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
      v.d.reset( (uint8_t *)fn_map->data() + pos, null_deleter<uint8_t>() ); // borrow pointer
      uint8_t *lb = (uint8_t *)fn_map->data() + pos;
      uint8_t *le = lb;
      uint8_t *de = (uint8_t *)fn_map->data() + size();
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
      pos += v.sz; // consume str
    }

    // note: does not read/fill-in timestamp_ns field of data_block_t, just size and data/pointer
    void read_val( data_block_t & v ) {
      uint32_t v_len;
      read_val( v_len );
      if( v_len == uint32_t_const_max ) { rt_err( "unexpected null byte array" ); }
      check_can_read( v_len );
      v.sz = v_len;
      v.d.reset( (uint8_t *)fn_map->data() + pos, null_deleter<uint8_t>() ); // borrow pointer
      pos += v_len;
    }

  };
}

#endif /* MAPPED_FILE_UTIL_H */
