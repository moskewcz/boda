#ifndef RESULTS_IO_H
#define RESULTS_IO_H

#include"boda_base.H"

namespace boda
{
  // currently internal use only, but maybe can/should be exported
  //void read_results_file( p_img_db_t img_db, std::string const & fn, std::string const &class_name );
  //p_img_db_t read_image_list_file( std::string const & fn );
  struct prc_elem_t
  {
    uint32_t num_pos;
    uint32_t num_test;
    double score;
    double get_precision( void ) const { return double(num_pos)/num_test; }
    double get_recall( uint32_t const tot_num_class ) const { return double(num_pos)/tot_num_class; }
    prc_elem_t( uint32_t const num_pos_, uint32_t const num_test_, double const & score_ ) : 
      num_pos(num_pos_), num_test(num_test_), score(score_) { }
  };
  typedef std::vector< prc_elem_t > vect_prc_elem_t;

  void score_results_file( std::string const & il_fn, std::string const & res_fn, std::string const &class_name );

  class img_db_t;
  typedef shared_ptr< img_db_t > p_img_db_t;
  p_img_db_t read_image_list_file( std::string const & fn, bool load_imgs );

}
#endif // RESULTS_IO_H
