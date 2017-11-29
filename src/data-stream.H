// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _DATA_STREAM_H_
#define _DATA_STREAM_H_

#include"boda_base.H"
#include"geom_prim.H"
#include<deque>

namespace boda
{
  struct lexp_name_val_map_t;
  typedef lexp_name_val_map_t nesi_init_arg_t;

  struct data_block_t;
  typedef vector< data_block_t > vect_data_block_t; 
  typedef std::deque< data_block_t > deque_data_block_t; 
  typedef shared_ptr< vect_data_block_t > p_vect_data_block_t; 
  struct data_block_t {
    p_nda_t nda; // pointer to data block, null if no data (usually a borrowed pointer from data_stream_t. TODO: add clone/copy iface?

    uint64_t sz( void ) const { return nda->dims.bytes_sz(); }
    void const * d( void ) const { return nda->rp_elems(); }

    // flow control data (experimental, used only by pipe?)
    bool need_more_in;
    bool have_more_out;

    // metadata (optional in general)
    // string tdn; // if we had an array of this block, what would the name of the dim that iterated of these be (i.e. tag/type name)
    string tag; // descritive tag of what this is: camera, camera-front, radar-intensity, ...
    string meta; // general class/type of data (i.e. how to set params for visualization, ...): image, lidar, radar, ... 

    uint64_t frame_ix; // what frame is this? zero based. const_max is invalid.
    uint64_t timestamp_ns; // data-stream-defined timestamp in nanoseconds. may not start at zero.
    p_img_t as_img; // if non-null, a image representation of this data (potentially smoothed/modified/etc)
    
    p_vect_data_block_t subblocks; // FIXME: make mapping from str->subblock? add meta data somewhere?
    data_block_t( void ) : need_more_in(0), have_more_out(0), frame_ix(uint64_t_const_max), timestamp_ns(uint64_t_const_max) { }
    bool has_subblocks( void ) const { return subblocks.get(); }
    void ensure_has_subblocks( void ) { if( !subblocks ) { subblocks = make_shared< vect_data_block_t >(); } }
    uint32_t num_subblocks( void ) const { return has_subblocks() ? subblocks->size() : uint32_t_const_max; }

    bool valid( void ) const { return nda.get() || subblocks.get(); } // if false, no fields are valid (in particular, sz/metadata is not valid)
    data_block_t clone( void ) const {
      data_block_t ret = *this;
      if( subblocks ) { // deep copy subblocks if any
        ret.subblocks = make_shared<vect_data_block_t>();
        for( vect_data_block_t::const_iterator i = subblocks->begin(); i != subblocks->end(); ++i ) {
          ret.subblocks->push_back( (*i).clone() );
        }
      }
      return ret;
    }
    string info_str( void ) const;
  protected:
    void info_str_int( string & prefix, string & out ) const;
  };
  std::ostream & operator << ( std::ostream & out, data_block_t const & db );


  // by convention:
  //  -- 'source' data_stream_t's are passed a default/empty data block (which they use generally use as a template). they generally do something like read a file as a side-effect.
  //  -- 'sink' data_stream_t's don't return anything of interest (i.e. they always echo the input block), but do something as a side-effect on each call: write a file, etc.
  //  -- 'xform' data_stream_t's transform/filter data blocks. they might keep some internal state as well.
  //
  // in general, seeking has stream-defined semantics, and if often not implemeneted (and/or not sensible). but for
  // 'source'-like streams, the idea is to allow going backwards in the stream, or perhaps to allow full
  // jumping. sometimes limited backwards seeking can be implemented with buffering. in other case, fast random seeking
  // might even be possible (i.e. with some existing index). but, often enabling seeking has a price in
  // time/memory/complexity ...

  struct data_stream_opt_t {  string name; p_nda_t val; };
  
  struct data_stream_t : virtual public nesi // NESI(help="modes that can be used for reading streams of timestamped data blocks"
                                              // ,is_abstract=1,tid_vn="mode")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string mode; //NESI(help="type of datastream",req=1)
    virtual void data_stream_init( nesi_init_arg_t * const nia ) = 0;
    virtual data_block_t proc_block( data_block_t const & db ) = 0;
    virtual bool seek_to_block( uint64_t const & frame_ix ) { return false; } // by default, not implemented / fails.

    // debugging iface(s) (may only have meaning for certain modes)
    virtual string get_pos_info_str( void ) = 0;

    // control/options: probably doesn't belong here, but unclear where we can put it?
    virtual void set_opt( data_stream_opt_t const & opt ) { } // by default, does nothing.

  };
  typedef vector< data_stream_t > vect_data_stream_t; 
  typedef shared_ptr< data_stream_t > p_data_stream_t; 
  typedef vector< p_data_stream_t > vect_p_data_stream_t;


  struct data_stream_tagged_frames_t : virtual public nesi, public data_stream_t // NESI(help="parse serialized data stream from file into data blocks",
                              // bases=["data_stream_t"], type_id="tagged-frames-src", is_abstract=1)
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    p_dims_t out_dims; // NESI(help="set dims of output to this. size must match data size. if not set, will typically emit 'bytes' (1D array of uint8_t, with one dim named 'v'), but default may depend on specific file reader.")
    p_string tag; // NESI(help="if set, set tag field in output data blocks to this. the tag field is a: descritive tag of what this is: camera, camera-front, radar-intensity, ...")
    p_string meta; // NESI(help="if set, set meta field in output data blocks to this. the meta field is a: general class/type of data (i.e. how to set params for visualization, ...) image, lidar, radar, ... ")

    uint64_t tot_num_read; // num blocks read so far

    data_stream_tagged_frames_t( void ) : tot_num_read(0) { }

    void data_stream_block_done_hook( data_block_t & ret ) {
      ret.frame_ix = tot_num_read;
      if( out_dims ) { assert_st( ret.nda ); ret.nda->reshape( *out_dims ); }
      if( tag.get() ) { ret.tag = *tag; }
      if( meta.get() ) { ret.meta = *meta; }
      ++tot_num_read;
      if( verbose ) { printf( "ret=%s\n", ret.info_str().c_str() ); }
    }
  };

  
}

#endif /* _DATA_STREAM_H_ */
