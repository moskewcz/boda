#ifndef _IMG_IO_H_
#define _IMG_IO_H_

#include"boda_base.H"

namespace boda
{
  // image in RGBA format. 
  // TODO:
  // padding for vertical dim (i.e. entire blank rows?)
  struct img_t
  {
    uint32_t w;
    uint32_t row_align; // what power-of-2 alignement is ensured for all row starts
    uint32_t row_pitch; // bytes between rows
    uint32_t h;
    p_uint8_t pels; // raw pixel data    
    img_t( void ) : w(0), row_align(0), row_pitch(0), h(0) {}
    // if desired, call to set requested row alignment *before* loading
    void set_row_align( uint32_t const row_align_ ) { assert(!row_align); row_align = row_align_; }
    void load_fn( std::string const & fn );
    void load_fn_jpeg( std::string const & fn );    
  };
  typedef shared_ptr< img_t > p_img_t;


}

#endif /* _IMG_IO_H_ */
