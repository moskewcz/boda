#ifndef _OCTIF_H_
#define _OCTIF_H_

namespace boda 
{
  void oct_init( void );
  void oct_test( void );

  struct scored_det_t;
  typedef std::vector< scored_det_t > vect_scored_det_t;
  typedef shared_ptr< vect_scored_det_t > p_vect_scored_det_t;

  void oct_dfc( p_vect_scored_det_t scored_dets, 
		std::string const & class_name, std::string const & image_fn, uint32_t const img_ix );
}

#endif // _OCTIF_H_
