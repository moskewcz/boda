// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _OCTIF_H_
#define _OCTIF_H_

namespace boda 
{
  void oct_init( void );
  void test_oct( void );

  struct vect_scored_det_t;
  typedef shared_ptr< vect_scored_det_t > p_vect_scored_det_t;

  void oct_dfc( std::ostream & out, std::string const & dpm_fast_cascade_dir, p_vect_scored_det_t scored_dets, 
		std::string const & image_fn, uint32_t const img_ix );
}

#endif // _OCTIF_H_
