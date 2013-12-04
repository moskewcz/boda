#ifndef PYIF_H
#define PYIF_H
#include"boda_base.H"

namespace boda 
{
  void py_init( char const * const prog_name );
  void py_finalize( void );
  struct prc_elem_t;
  typedef vector< prc_elem_t > vect_prc_elem_t;
  void prc_plot( std::string const & plot_fn, uint32_t const tot_num_class, vect_prc_elem_t const & prc_elems );

  struct img_t; typedef shared_ptr< img_t > p_img_t;
  void py_img_show( p_img_t img, std::string const & save_as_filename );

  struct scored_det_t;
  typedef std::vector< scored_det_t > vect_scored_det_t;
  typedef shared_ptr< vect_scored_det_t > p_vect_scored_det_t;

  void show_dets( p_img_t img, p_vect_scored_det_t scored_dets );
  std::string const & py_boda_dir( void );
  std::string const & py_boda_test_dir( void );

}

#endif // PYIF_H
