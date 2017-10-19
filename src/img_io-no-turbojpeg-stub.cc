// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"img_io.H"

namespace boda { 
  void img_t::from_jpeg( p_uint8_with_sz_t const & data, std::string const & fn ) { rt_err("boda not compiled with jpeg loading support."); }
  p_uint8_with_sz_t img_t::to_jpeg( void ) { rt_err("boda not compiled with jpeg creation support."); }
}
