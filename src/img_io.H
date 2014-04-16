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
    void load_fn( std::string const & fn );
    void load_fn_jpeg( std::string const & fn );
    void load_fn_png( std::string const & fn );
    void save_fn_png( std::string const & fn );

    uint32_t min_dim( void ) { return std::min(w,h); }
    uint32_t sz_raw_bytes( void ) const { return row_pitch*h; }
    string WxH_str( void );
  };

  p_img_t downsample_to_size( p_img_t img, uint32_t const ds_w, uint32_t const ds_h ); // ds_w must be in [ceil(w/2),w]
  p_img_t downsample( p_img_t img, double const scale );
  p_img_t downsample_2x( p_img_t img ); // downsample by exactly 2X (special cased for speed, called by above when s=.5)
  p_img_t upsample_2x( p_img_t img ); // upsample by exactly 2X 
  void downsample_test( std::string const & fn );

  void img_copy_to( img_t const * const src, img_t * const dest, uint32_t const & dx, uint32_t const & dy );

}

#endif /* _IMG_IO_H_ */
