// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _OCTIF_H_
#define _OCTIF_H_

namespace boda 
{
  void oct_init( void );
  void oct_exit( int const retval );
  void test_oct( void );

  struct per_class_scored_dets_t;
  typedef shared_ptr< per_class_scored_dets_t > p_per_class_scored_dets_t;

  void oct_dfc( std::ostream & out, std::string const & dpm_fast_cascade_dir, p_per_class_scored_dets_t scored_dets, 
		std::string const & image_fn, uint32_t const img_ix );
}

#endif // _OCTIF_H_
