// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _IMG_IO_H_
#define _IMG_IO_H_

#include"boda_base.H"

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
    uint32_t w;
    uint32_t row_align; // what power-of-2 alignement is ensured for all row starts
    uint32_t row_pitch; // bytes between rows
    uint32_t row_pitch_pels; // pixels between rows
    uint32_t h;
    p_uint8_t pels; // raw pixel data    
    img_t( void ) : depth(4), w(0), row_align(0), row_pitch(0), row_pitch_pels(0), h(0) {}
    // if desired, call to set requested row alignment *before* loading
    void set_row_align( uint32_t const row_align_ ) { assert(!row_align); row_align = row_align_; }
    void set_sz_and_alloc_pels( uint32_t const w_, uint32_t const h_ );
    void fill_with_pel( uint32_t const & v );
    void load_fn( std::string const & fn );
    void load_fn_jpeg( std::string const & fn );
    void load_fn_png( std::string const & fn );
    void save_fn_png( std::string const & fn, bool const disable_compression = 0 );

    // always equivalent to y*row_pitch_pels + x (i.e get_pel_ix(0,0) will always == 0)
    uint32_t get_pel_ix( uint32_t const & x, uint32_t const & y ) const { 
      assert_st( x < w ); 
      assert_st( y < h );
      return y*row_pitch_pels + x; 
    }
    uint32_t get_pel( uint32_t const & x, uint32_t const & y ) const { return ((uint32_t *)pels.get())[get_pel_ix(x,y)];}
    void set_pel( uint32_t const & x, uint32_t const & y, uint32_t const & v ) { 
      ((uint32_t *)pels.get())[get_pel_ix(x,y)] = v; }
    uint8_t get_pel_chan( uint32_t const & x, uint32_t const & y, uint8_t const & c ) const { 
      return (get_pel( x, y ) >> (c*8))&0xff;
    }
    uint32_t min_dim( void ) { return std::min(w,h); }
    uint32_t sz_raw_bytes( void ) const { return row_pitch*h; }
    string WxH_str( void );
  };

  inline uint32_t grey_to_pel( uint8_t const gv, uint8_t const av = uint8_t_const_max ) {
    return gv + (gv<<8) + (gv<<16) + (av<<24);
  }
  inline uint8_t get_chan( uint8_t const c, uint32_t const v ) { return v >> (c*8); }
  inline uint64_t get_chan_64( uint8_t const c, uint32_t const v ) { return get_chan( c, v ); }


  p_img_t downsample_to_size( p_img_t img, uint32_t const ds_w, uint32_t const ds_h ); // ds_w must be in [ceil(w/2),w]
  p_img_t downsample( p_img_t img, double const scale );
  p_img_t downsample_2x( p_img_t img ); // downsample by exactly 2X (special cased for speed, called by above when s=.5)
  p_img_t upsample_2x( p_img_t img ); // upsample by exactly 2X 
  void downsample_test( std::string const & fn );

  void img_copy_to( img_t const * const src, img_t * const dest, uint32_t const & dx, uint32_t const & dy );
  void img_draw_pels( img_t * const dest, uint32_t const & dx, uint32_t const & dy, uint32_t const & num, 
		      int32_t const & stride_x, int32_t const & stride_y, 
		      uint32_t const & ic, uint32_t const & ec );

}

#endif /* _IMG_IO_H_ */
