// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _IMG_IO_H_
#define _IMG_IO_H_

#include"boda_base.H"
#include"geom_prim.H"

namespace boda
{
  // image in RGBA format. 
  // TODO:
  // padding for vertical dim (i.e. entire blank rows?)
  struct img_t; 
  typedef shared_ptr< img_t > p_img_t; 
  typedef vector< p_img_t > vect_p_img_t; 
  typedef shared_ptr< vect_p_img_t > p_vect_p_img_t; 

  struct img_t
  {
    uint8_t depth;
    u32_pt_t sz;
    uint32_t row_align; // what power-of-2 alignement is ensured for all row starts
    uint32_t row_pitch; // bytes between rows
    uint32_t row_pitch_pels; // pixels between rows
    p_uint8_t pels; // raw pixel data
    vect_p_uint8_t yuv_pels; // pixel data in YUV 420 format. NOTE: only partially supported and used in selected areas.
    img_t( void ) : depth(4), row_align(0), row_pitch(0), row_pitch_pels(0) {}
    p_img_t clone( void ) const;
    // if desired, call to set requested row alignment *before* loading
    void set_row_align( uint32_t const row_align_ ) { assert(!row_align); row_align = row_align_; }
    void set_sz( u32_pt_t const & sz_ );
    void alloc_pels( void );
    void set_sz_and_alloc_pels( u32_pt_t const & sz_ );
    void set_sz_and_pels_from_yuv_420_planes( vect_p_nda_uint8_t const & yuv_pels_ );

    void fill_with_pel( uint32_t const & v );
    void load_fn( std::string const & fn );
    void load_fn_jpeg( std::string const & fn );
    p_uint8_with_sz_t to_jpeg( void );
    void save_fn_jpeg( std::string const & fn );
    void load_fn_png( std::string const & fn );
    void load_fn_stb_image_supported( std::string const & fn );
    
    // note: for the following from_* functions, the fn is for debugging prinouts only
    void from_jpeg( p_uint8_with_sz_t const & data, std::string const & fn );
    void from_stb_image_supported( p_uint8_with_sz_t const & data, std::string const & fn );

    void save_fn_png( std::string const & fn, bool const disable_compression = 0 );

    p_nda_uint8_t as_packed_RGBA_nda( void );

    // always equivalent to y*row_pitch_pels + x (i.e get_pel_ix(0,0) will always == 0)
    uint32_t get_pel_ix( u32_pt_t const & d ) const { 
      assert_st( d.both_dims_lt( sz ) );
      return d.d[1]*row_pitch_pels + d.d[0]; 
    }
    uint32_t get_pel( u32_pt_t const & d ) const { return ((uint32_t *)pels.get())[get_pel_ix(d)];}
    uint32_t * get_pel_addr( u32_pt_t const & d ) { return ((uint32_t *)pels.get()) + get_pel_ix(d);}
    uint32_t const * get_pel_addr( u32_pt_t const & d ) const { return ((uint32_t const *)pels.get()) + get_pel_ix(d);}
    void set_pel( u32_pt_t const & d, uint32_t const & v ) { *get_pel_addr(d) = v; }
    uint8_t get_pel_chan( u32_pt_t const & d, uint8_t const & c ) const { 
      return (get_pel(d) >> (c*8))&0xff;
    }
    uint32_t sz_raw_bytes( void ) const { return row_pitch*sz.d[1]; }
    uint32_t * get_row_addr( uint32_t const y ) { return get_pel_addr({0,y}); }
    void get_YUV_row_addr( uint32_t const y, uint8_t const * & yr, uint8_t const * & ur, uint8_t const * & vr );
    
    string WxH_str( void );
    void share_pels_from( p_img_t o );
  };

  inline p_img_t make_p_img_t( u32_pt_t const & sz ) { 
    p_img_t ret( new img_t ); ret->set_sz_and_alloc_pels( sz ); return ret; }

  inline uint32_t rgba_to_pel( uint8_t const r, uint8_t const g, uint8_t const b, int8_t const av = uint8_t_const_max ) { 
    return r + (g<<8) + (b<<16) + (av<<24); }
  inline uint32_t grey_to_pel( uint8_t const gv, uint8_t const av = uint8_t_const_max ) { return rgba_to_pel(gv,gv,gv,av); } 
  inline uint8_t get_chan( uint8_t const c, uint32_t const v ) { return v >> (c*8); }
  inline uint64_t get_chan_64( uint8_t const c, uint32_t const v ) { return get_chan( c, v ); }

  p_img_t resample_to_size( p_img_t img, u32_pt_t const & ds );
  p_img_t downsample_to_size( p_img_t img, u32_pt_t const & ds );
  p_img_t downsample_up_to_2x_to_size( p_img_t img, u32_pt_t const & ds ); // ds_w must be in [ceil(w/2),w]
  p_img_t downsample_2x_to_size( p_img_t img, u32_pt_t const & ds );
  p_img_t downsample_up_to_2x( p_img_t img, double const scale );
  p_img_t downsample_2x( p_img_t img ); // downsample by exactly 2X (special cased for speed, called by above when s=.5)
  p_img_t upsample_2x( p_img_t img ); // upsample by exactly 2X 
  void downsample_test( std::string const & fn );

  void img_copy_to_clip( img_t const * const src, img_t * const dest, u32_pt_t const & dest_pt = u32_pt_t{},
			 u32_pt_t const & src_pt = u32_pt_t{}, u32_pt_t const & max_copy_sz = u32_pt_t_const_max );
  void img_draw_pels( img_t * const dest, u32_pt_t const & d, uint32_t const & num, i32_pt_t const & stride,
		      uint32_t const & ic, uint32_t const & ec );

  // from fourcc.org, YUYV->RGB conversion formula. 
  inline uint8_t clip_to_u8( float const & v ) { if( v < 0 ) { return 0; } if( v > 255 ) { return 255; } return v; }
  inline uint32_t yuva2rgba( uint8_t const & y, uint8_t const & u, uint8_t const & v, uint8_t const a = uint8_t_const_max ) {
    float const yf = float(y)-16.0f;
    float const uf = float(u)-128.0f;
    float const vf = float(v)-128.0f;
    uint8_t const r = clip_to_u8( 1.164f*yf + 1.596f*vf );
    uint8_t const g = clip_to_u8( 1.164f*yf - 0.813f*vf - 0.391f*uf );
    uint8_t const b = clip_to_u8( 1.164f*yf + 2.018f*uf );
    return r + (g<<8) + (b<<16) + (a<<24);
  }
  inline void rgba2yuv( uint32_t const & p, uint8_t & y, uint8_t & u, uint8_t & v ) { 
    float const r = float(get_chan(0,p)); float const g = float(get_chan(1,p)); float const b = float(get_chan(2,p));
    y = clip_to_u8(  16.0f + 0.257f*r + 0.504f*g + 0.098f*b );
    u = clip_to_u8( 128.0f - 0.148f*r - 0.291f*g + 0.439f*b );
    v = clip_to_u8( 128.0f + 0.439f*r - 0.368f*g - 0.071f*b );
  }
  // versions of rgba2yuv for when we only want y or uv (for subsamp)
  inline void rgba2y( uint32_t const & p, uint8_t & y ) { 
    float const r = float(get_chan(0,p)); float const g = float(get_chan(1,p)); float const b = float(get_chan(2,p));
    y = clip_to_u8(  16.0f + 0.257f*r + 0.504f*g + 0.098f*b );
  }
  inline void rgba2uv( uint32_t const & p, uint8_t & u, uint8_t & v ) { 
    float const r = float(get_chan(0,p)); float const g = float(get_chan(1,p)); float const b = float(get_chan(2,p));
    u = clip_to_u8( 128.0f - 0.148f*r - 0.291f*g + 0.439f*b );
    v = clip_to_u8( 128.0f + 0.439f*r - 0.368f*g - 0.071f*b );
  }

}

#endif /* _IMG_IO_H_ */
