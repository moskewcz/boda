// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef RESULTS_IO_H
#define RESULTS_IO_H

#include"boda_base.H"
#include"geom_prim.H"
#include"has_main.H"
#include<map>

namespace boda
{
  // scored_dets to gt_dets mapping / evaluation data
  struct gt_match_t 
  {
    bool matched;
    double match_score;
    gt_match_t( void ) : matched(0), match_score(0) { }
  };
  typedef vector< gt_match_t > vect_gt_match_t;
  typedef map< string, vect_gt_match_t > name_vect_gt_match_map_t;
  typedef vector< name_vect_gt_match_map_t > vect_name_vect_gt_match_map_t;
  typedef vector< vect_gt_match_t > vect_vect_gt_match_t;

  struct scored_det_t : public u32_box_t
  {
    double score;
    uint32_t img_ix;
    scored_det_t( void ) : score(0), img_ix(0) { }
  };
  std::ostream & operator <<(std::ostream & os, const scored_det_t & v);
  struct vect_scored_det_t : public std::vector< scored_det_t >
  {
    vect_scored_det_t( string const & class_name_ ) : class_name( class_name_ ) { }
    string class_name;
    vect_vect_gt_match_t gt_matches;
    void clear_matches( void ) { gt_matches.clear(); }
    vect_gt_match_t & get_gtms( uint32_t const img_ix, uint32_t const img_num_gts ) {
      if( img_ix >= gt_matches.size() ) { gt_matches.resize( img_ix + 1 ); }
      vect_gt_match_t & ret = gt_matches[img_ix];
      if( ret.empty() ) { ret.resize( img_num_gts ); }
      assert_st( ret.size() == img_num_gts );
      return ret;
    }
  };
  typedef shared_ptr< vect_scored_det_t > p_vect_scored_det_t;
  typedef vector< p_vect_scored_det_t > vect_p_vect_scored_det_t;
  typedef shared_ptr< vect_p_vect_scored_det_t > p_vect_p_vect_scored_det_t; 
  
  // currently internal use only, but maybe can/should be exported
  //p_vect_scored_det_t read_results_file( p_img_db_t img_db, std::string const & fn );

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
  void read_pascal_image_list_file( p_img_db_t img_db, filename_t const & pil_fn, bool const load_imgs, 
				    bool const check_ix_only );
  void write_results_file( p_img_db_t img_db, string const & fn, p_vect_scored_det_t scored_dets );
  void img_db_show_dets( p_img_db_t img_db, p_vect_scored_det_t scored_dets, uint32_t img_ix );

  struct img_t; 
  typedef shared_ptr< img_t > p_img_t; 
  typedef vector< p_img_t > vect_p_img_t; 
  typedef shared_ptr< vect_p_img_t > p_vect_p_img_t; 

  void img_db_get_all_loaded_imgs( p_vect_p_img_t const & out, p_img_db_t img_db );


  struct load_imgs_from_pascal_classes_t : virtual public nesi // NESI(help="load PASCAL VOC images by classes" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t pascal_classes_fn; //NESI(default="%(boda_test_dir)/pascal/head_10/pascal_classes.txt",help="file with list of pascal classes to load")
    p_img_db_t img_db; //NESI(default="()", help="image database")
    filename_t pil_fn; //NESI(default="%(boda_test_dir)/pascal/head_10/%%s.txt",help="format for filenames of image list files. %%s will be replaced with the class name")
    p_vect_p_img_t all_imgs;

    void load_all_imgs( void );
  };
  typedef shared_ptr< load_imgs_from_pascal_classes_t > p_load_imgs_from_pascal_classes_t; 

}
#endif // RESULTS_IO_H
