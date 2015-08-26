// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"pyif.H"

namespace boda 
{
  void py_init( char const * const prog_name ) {}
  void py_finalize( void ) {}
  void prc_plot( std::string const & plot_fn, uint32_t const tot_num_class, vect_prc_elem_t const & prc_elems,
		 std::string const & plot_title )
  {
    printf("warning: python disabled, can't run prc_plot()\n");
  }
  void py_img_show( p_img_t img, std::string const & save_as_filename ) {
    printf("warning: python disabled, can't run py_img_show()\n");
  }
  void show_dets( p_img_t img, vect_base_scored_det_t const & scored_dets ) {
    printf("warning: python disabled, can't run show_dets()\n");
  }

}
