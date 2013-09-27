#ifndef PYIF_H
#define PYIF_H
#include"boda_base.H"
#include"results_io.H"

namespace boda 
{
  void py_init( char const * const prog_name );
  void py_finalize( void );
  void prc_plot( std::string const & class_name, uint32_t const tot_num_class, vect_prc_elem_t const & prc_elems );
}

#endif // PYIF_H
