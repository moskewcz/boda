// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"img_io.H"
#include"has_main.H"
#include"str_util.H"
#include"data-stream.H"
#include"mapped-file-util.H"

namespace boda 
{

  struct data_stream_start_stop_skip_t : virtual public nesi, public data_stream_t // NESI(help="wrap another data stream and optionally: skip initial blocks and/or skip blocks after each returned block and/or limit the number of blocks returned.",
                             // bases=["data_stream_t"], type_id="start-stop-skip")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    p_data_stream_t src; //NESI(req=1,help="wrapped data stream")

    // for debugging / skipping:
    uint64_t start_block; //NESI(default=0,help="start at this block")
    uint64_t skip_blocks; //NESI(default=0,help="drop/skip this many blocks after each returned block (default 0, no skipped/dropped blocks)")
    uint64_t num_to_read; //NESI(default=0,help="read this many records; zero for unlimited")

    // internal state:
    uint64_t tot_num_read; // num blocks read so far

    virtual string get_pos_info_str( void ) { return strprintf( "tot_num_read=%s; src info: %s", str(tot_num_read).c_str(), str(src->get_pos_info_str()).c_str() ); }

    virtual data_block_t read_next_block( void ) {
      if( num_to_read && (tot_num_read >= num_to_read) ) { return data_block_t(); }
      data_block_t ret = src->read_next_block();
      ++tot_num_read;
      for( uint32_t i = 0; i != skip_blocks; ++i ) { src->read_next_block(); } // skip blocks if requested
      return ret;
    }

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      src->data_stream_init( nia );
      printf( "data_stream_init(): mode=%s start_block=%s skip_blocks=%s num_to_read=%s\n",
              str(mode).c_str(), str(start_block).c_str(), str(skip_blocks).c_str(), str(num_to_read).c_str() );
      tot_num_read = 0;
      for( uint32_t i = 0; i != start_block; ++i ) { src->read_next_block(); } // skip to start block
    }    
  };

  struct data_stream_file_t : virtual public nesi, public data_stream_t // NESI(help="parse serialized data stream from file into data blocks",
                              // bases=["data_stream_t"], type_id="file", is_abstract=1)
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    filename_t fn; //NESI(default="vid.raw",help="input raw video filename")
    uint64_t tot_num_read; // num blocks read so far
    mapped_file_stream_reader mfsr;
    
    virtual string get_pos_info_str( void ) { return strprintf( "pos=%s tot_num_read=%s", str(mfsr.pos).c_str(), str(tot_num_read).c_str() ); }
    
    virtual void data_stream_init( nesi_init_arg_t * nia ) { // note: to be called explicity by derived classes if they override
      printf( "data_stream_init(): mode=%s fn.exp=%s\n", str(mode).c_str(), str(fn.exp).c_str() );
      mfsr.init( fn );
      tot_num_read = 0;
    }
  };

  struct data_stream_qt_t : virtual public nesi, public data_stream_file_t // NESI(help="parse qt-style-serialized data stream into data blocks",
                            // bases=["data_stream_file_t"], type_id="qt")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    
    virtual data_block_t read_next_block( void ) {
      data_block_t ret;
      if( mfsr.pos == timestamp_off ) {
#if 0
        printf( "at timestamp_off: pos=%s\n", str(mfsr.pos).c_str() );
        uint8_t ch;
        while( mfsr.can_read( sizeof(ch) ) ) {
          mfsr.read_val(ch);
          printf( "ch=%s ch=%s\n", str(uint32_t(ch)).c_str(), str(ch).c_str() );
        }
#endif     
        return ret;
      }
      if( !mfsr.can_read( sizeof( ret.timestamp_ns ) ) ) { return ret; } // not enough bytes left for another block
      mfsr.read_val( ret.timestamp_ns );
      if( !mfsr.can_read( sizeof( uint32_t ) ) ) { rt_err( "qt stream: read timestamp, but not enough data left to read payload size" ); }
      mfsr.read_val( ret );

      ++tot_num_read;
      if( verbose ) { printf( "ret.sz=%s ret.timestamp_ns=%s\n", str(ret.sz).c_str(), str(ret.timestamp_ns).c_str() ); }
      return ret;
    }

    // qt stream info/state
    uint64_t timestamp_off;
    uint64_t chunk_off;

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      data_stream_file_t::data_stream_init( nia );
      mfsr.need_endian_reverse = 1; // assume stream is big endian, and native is little endian. could check this ...
      uint32_t ver;
      mfsr.read_val( ver );
      string tag;
      mfsr.read_val( tag );
      data_block_t header;
      mfsr.read_val( header );
      mfsr.read_val( timestamp_off );
      mfsr.read_val( chunk_off );
      uint64_t duration_ns;
      mfsr.read_val( duration_ns );
      printf( "  qt stream header: ver=%s tag=%s header.size()=%s timestamp_off=%s chunk_off=%s duration_ns=%s\n",
              str(ver).c_str(), str(tag).c_str(), str(header.sz).c_str(), str(timestamp_off).c_str(),
              str(chunk_off).c_str(), str(duration_ns).c_str() );
      if( mfsr.size() > timestamp_off ) {
        printf( "   !! warning: (size() - timestamp_off)=%s bytes at end of file will be ignored\n",
                str((mfsr.size() - timestamp_off)).c_str() );
      }
    }
  };

  struct data_stream_dumpvideo_t : virtual public nesi, public data_stream_file_t // NESI(help="parse dumpvideo data stream into data blocks",
                                   // bases=["data_stream_file_t"], type_id="dumpvideo")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    virtual data_block_t read_next_block( void ) {
      data_block_t ret;
      uint32_t block_sz;
      if( !mfsr.can_read( sizeof( block_sz ) ) ) { return ret; } // not enough bytes left for another block
      mfsr.read_val( block_sz );
      ret = mfsr.consume_borrowed_block( block_sz ); // note: timestamp not set here
      if( verbose ) { printf( "ret.sz=%s ret.timestamp_ns=%s\n", str(ret.sz).c_str(), str(ret.timestamp_ns).c_str() ); }
      return ret;
    }
  };

  struct data_stream_text_t : virtual public nesi, public data_stream_file_t // NESI(help="parse data stream (dumpvideo/qt) into data blocks",
                             // bases=["data_stream_file_t"], type_id="text")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t text_tsfix; //NESI(default=0,help="text time-stamp-field-index: use the N'th field as a decimal timestamp in seconds (with fractional part).")

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

    virtual data_block_t read_next_block( void ) {
      data_block_t ret;
      if( !mfsr.can_read( 1 ) ) { return ret; } // not enough bytes left for another block
      mfsr.read_line_as_block( ret );
      set_timestamp_from_text_line( ret );

      ++tot_num_read;
      if( verbose ) { printf( "ret.sz=%s ret.timestamp_ns=%s\n", str(ret.sz).c_str(), str(ret.timestamp_ns).c_str() ); }
      return ret;
    }

    virtual void data_stream_init( nesi_init_arg_t * nia ) {
      data_stream_file_t::data_stream_init( nia );
      data_block_t header;
      mfsr.read_line_as_block( header );
      printf( "  text stream header.sz=%s\n", str(header.sz).c_str() );
    }
  };

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
    p_data_stream_t vps; //NESI(req=1,help="underlying velodyne packet stream")

    uint32_t fbs_per_packet; //NESI(default="12",help="firing blocks per packet")
    uint32_t beams_per_fb; //NESI(default="32",help="beams per firing block")
    uint32_t status_bytes; //NESI(default="6",help="bytes of status at end of block")

    uint32_t fb_sz;
    uint32_t packet_sz;
    uint16_t last_rot;
    
    // internal state:
    uint64_t tot_num_read; // num blocks read so far

    virtual string get_pos_info_str( void ) { return strprintf( "tot_num_read=%s vps info:%s",
                                                                str(tot_num_read).c_str(), vps->get_pos_info_str().c_str() ); }

    virtual data_block_t read_next_block( void ) {
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
      printf( "data_stream_init(): mode=%s\n", str(mode).c_str() );
      tot_num_read = 0;
      // setup internal state
      fb_sz = sizeof( block_info_t ) + beams_per_fb * sizeof( laser_info_t );
      packet_sz = fbs_per_packet * fb_sz + status_bytes;
      last_rot = 0;        
      vps->data_stream_init( nia );
    }
    
    void main( nesi_init_arg_t * nia ) { 
      data_stream_init( nia );
      while( read_next_block().d.get() ) { }
    }
  };

  struct scan_data_stream_t : virtual public nesi, public has_main_t // NESI(
                              // help="testing mode to scan N data streams one-by-one, and print total number of blocks read for each.",
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
