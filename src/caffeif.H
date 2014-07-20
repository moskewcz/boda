// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _CAFFEIF_H_
#define _CAFFEIF_H_

#include"boda_base.H"
#include"caffe/caffe.hpp"

namespace boda 
{
  typedef caffe::Net< float > Net_float;
  typedef shared_ptr< Net_float > p_Net_float;
  p_Net_float init_caffe( string const & param_str, string const & trained_fn );
  void raw_do_forward( p_Net_float net_, vect_p_nda_float_t const & bottom );
  void copy_output_blob_data( p_Net_float net_, string const & out_layer_name, vect_p_nda_float_t & top );
}

#endif /* _CAFFEIF_H_ */
