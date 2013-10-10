#ifndef _IMG_IO_H_
#define _IMG_IO_H_

#include"boda_base.H"

namespace boda
{
  // image in RGBA format. 
  // TODO:
  // padding for vertical dim (i.e. entire blank rows?)
  struct img_t; typedef shared_ptr< img_t > p_img_t;
  struct img_t
  {
    uint8_t depth;
    uint32_t w;
    uint32_t row_align; // what power-of-2 alignement is ensured for all row starts
    uint32_t row_pitch; // bytes between rows
    uint32_t h;
    p_uint8_t pels; // raw pixel data    
    img_t( void ) : depth(4), w(0), row_align(0), row_pitch(0), h(0) {}
    // if desired, call to set requested row alignment *before* loading
    void set_row_align( uint32_t const row_align_ ) { assert(!row_align); row_align = row_align_; }
    void set_sz_and_alloc_pels( uint32_t const w_, uint32_t const h_ );
    void load_fn( std::string const & fn );
    void load_fn_jpeg( std::string const & fn );
    void load_fn_png( std::string const & fn );
    p_img_t downsample( uint16_t scale ); // scale is 0.16 fixed point, value must be in [.5,1]
  };

  void downsample_test( std::string const & fn );

}

#endif /* _IMG_IO_H_ */
