#ifndef RESULTS_IO_H
#define RESULTS_IO_H

#include"boda_base.H"
#include"geom_prim.H"
#include"has_main.H"
#include<map>

namespace boda
{

  struct scored_det_t : public u32_box_t
  {
    double score;
    uint32_t img_ix;
    scored_det_t( void ) : score(0), img_ix(0) { }
  };
  std::ostream & operator<<(std::ostream & os, const scored_det_t & v);
  typedef std::vector< scored_det_t > vect_scored_det_t;
  typedef shared_ptr< vect_scored_det_t > p_vect_scored_det_t;
  typedef std::map< std::string, p_vect_scored_det_t > name_vect_scored_det_map_t;

  // currently internal use only, but maybe can/should be exported
  //void read_results_file( p_img_db_t img_db, std::string const & fn, std::string const &class_name );

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

  void score_results_file( std::string const & pil_fn, std::string const & res_fn, std::string const &class_name );
  void run_dfc( std::string const & pil_fn, std::string const & res_fn, std::string const &class_name );

  class img_db_t;
  typedef shared_ptr< img_db_t > p_img_db_t;
  void read_pascal_image_list_file( p_img_db_t img_db, std::string const & pil_fn, bool const load_imgs, 
				    bool const check_ix_only );
  void img_db_set_results( p_img_db_t img_db, string const &class_name, p_vect_scored_det_t scored_dets );
  void write_results_file( p_img_db_t img_db, string const & fn, string const &class_name );
  void img_db_show_dets( p_img_db_t img_db, p_vect_scored_det_t scored_dets, uint32_t img_ix );

}
#endif // RESULTS_IO_H
