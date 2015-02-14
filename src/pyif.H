// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef PYIF_H
#define PYIF_H
#include"boda_base.H"

namespace boda 
{
  void py_init( char const * const prog_name );
  void py_finalize( void );
  struct prc_elem_t;
  typedef vector< prc_elem_t > vect_prc_elem_t;
  void prc_plot( std::string const & plot_fn, uint32_t const tot_num_class, vect_prc_elem_t const & prc_elems,
		 std::string const & plot_title );

  struct img_t; typedef shared_ptr< img_t > p_img_t;
  void py_img_show( p_img_t img, std::string const & save_as_filename );

  struct base_scored_det_t;
  typedef vector< base_scored_det_t > vect_base_scored_det_t;

  void show_dets( p_img_t img, vect_base_scored_det_t const & scored_dets );
  std::string const & py_boda_dir( void );
  std::string const & py_boda_test_dir( void );

}

#endif // PYIF_H
