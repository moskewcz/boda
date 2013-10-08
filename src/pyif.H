#ifndef PYIF_H
#define PYIF_H
#include"boda_base.H"
#include"results_io.H"

namespace boda 
{
  void py_init( char const * const prog_name );
  void py_finalize( void );
  void prc_plot( std::string const & class_name, uint32_t const tot_num_class, vect_prc_elem_t const & prc_elems );

  struct img_t; typedef shared_ptr< img_t > p_img_t;
  void py_img_show( p_img_t img, std::string const & save_as_filename );

}

#endif // PYIF_H
