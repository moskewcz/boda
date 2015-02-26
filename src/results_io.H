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

  struct base_scored_det_t : public u32_box_t
  {
    double score;
    base_scored_det_t( void ) : score(0) {}
    base_scored_det_t( u32_box_t const & b_, double const & score_ ) : u32_box_t(b_), score(score_) {}
  };
  typedef vector< base_scored_det_t > vect_base_scored_det_t;
  typedef vector< vect_base_scored_det_t > vect_vect_base_scored_det_t;
  typedef shared_ptr< vect_base_scored_det_t > p_vect_base_scored_det_t;
  typedef vector< p_vect_base_scored_det_t > vect_p_vect_base_scored_det_t;

  struct scored_det_t : public base_scored_det_t
  {
    uint32_t img_ix;
    scored_det_t( void ) : img_ix(0) {}
    scored_det_t( base_scored_det_t const & v_, uint32_t const & img_ix_ ) : base_scored_det_t(v_), img_ix(img_ix_) {}
  };
  typedef vector< scored_det_t > vect_scored_det_t;
  typedef shared_ptr< vect_scored_det_t > p_vect_scored_det_t;
  
  std::ostream & operator <<(std::ostream & os, const scored_det_t & v);
  struct per_class_scored_dets_t
  {
    per_class_scored_dets_t( string const & class_name_ ) : class_name( class_name_ ) { }
    string class_name;
    vect_p_vect_base_scored_det_t per_img_sds; // per_img_sds[ix] is all detections for image ix
    vect_vect_gt_match_t gt_matches; // gt_matches[ix] is matched bits for all class_name GT boxes for image ix
    vect_gt_match_t & get_gtms( uint32_t const img_ix, uint32_t const img_num_gts ) { 
      if( img_ix >= gt_matches.size() ) { gt_matches.resize( img_ix + 1 ); }
      vect_gt_match_t & ret = gt_matches[img_ix];
      if( ret.empty() ) { ret.resize( img_num_gts ); }
      assert_st( ret.size() == img_num_gts );
      return ret;
    }
    p_vect_base_scored_det_t & get_per_img_sds( uint32_t const & img_ix, bool const & alloc_if_null ) {
      if( img_ix >= per_img_sds.size() ) { per_img_sds.resize( img_ix + 1 ); }
      p_vect_base_scored_det_t & ret = per_img_sds[img_ix];
      if( alloc_if_null && !ret ) { ret.reset( new vect_base_scored_det_t ); }
      return ret;
    }
    void add_det( scored_det_t const & sd ) { get_per_img_sds(sd.img_ix,1)->push_back( sd ); }
    p_vect_scored_det_t get_merged_all_imgs_sds( void );
  };
  typedef shared_ptr< per_class_scored_dets_t > p_per_class_scored_dets_t;
  typedef vector< p_per_class_scored_dets_t > vect_p_per_class_scored_dets_t;
  typedef shared_ptr< vect_p_per_class_scored_dets_t > p_vect_p_per_class_scored_dets_t; 

  struct gt_det_t : public u32_box_t
  {
    uint32_t truncated;
    uint32_t difficult;    
    gt_det_t( void ) : truncated(0), difficult(0) { }
  };
  struct vect_gt_det_t : public vector< gt_det_t >
  {
    zi_uint32_t num_non_difficult;
  };
  typedef map< string, vect_gt_det_t > name_vect_gt_det_map_t;
  struct img_info_t
  {
    string id;
    string full_fn;
    uint32_t ix;
    u32_pt_t size;
    uint32_t depth;
    name_vect_gt_det_map_t gt_dets;

    p_img_t img;
    img_info_t( string const & id_ ) : id(id_), ix( uint32_t_const_max ), depth(0) { }
  };
  typedef shared_ptr< img_info_t > p_img_info_t;
  typedef map< string, p_img_info_t > id_to_img_info_map_t;
  typedef vector< p_img_info_t > vect_p_img_info_t;
  
  // currently internal use only, but maybe can/should be exported
  //p_per_class_scored_dets_t read_results_file( p_img_db_t img_db, std::string const & fn );

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
  void read_pascal_image_list_files( p_img_db_t img_db, filename_t const & pil_fn_template,
				     vect_string const & classes, bool const load_imgs );

  void write_results_file( p_img_db_t img_db, string const & fn, p_per_class_scored_dets_t scored_dets );
  void img_db_show_dets( p_img_db_t img_db, p_per_class_scored_dets_t scored_dets, uint32_t img_ix );

  struct img_t; 
  typedef shared_ptr< img_t > p_img_t; 
  typedef vector< p_img_t > vect_p_img_t; 
  typedef shared_ptr< vect_p_img_t > p_vect_p_img_t; 

  typedef map< string, zi_uint32_t > class_infos_t;

  struct match_res_t
  {
    bool is_pos;
    bool is_diff;
    match_res_t( bool const is_pos_, bool const is_diff_ ) : is_pos(is_pos_), is_diff(is_diff_) { }
  };

  void img_db_get_all_loaded_imgs( p_vect_p_img_t const & out, p_img_db_t img_db );

  struct img_db_t : virtual public nesi // NESI(help="store images and detections")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t pascal_ann_dir; // NESI(default="%(pascal_data_dir)/Annotations",help="pascal annotations directory")
    filename_t pascal_img_dir; // NESI(default="%(pascal_data_dir)/JPEGImages",help="pascal images directory")
    id_to_img_info_map_t id_to_img_info_map;
    vect_p_img_info_t img_infos;
    class_infos_t class_infos;

    // note: expects that only a single image list will be loaded
    // (with check_ix_only=0), and that any subsequent lists loaded
    // (with check_ix_only=1) will have the same images in the same
    // order. this strictness is primarily for error checking can be
    // relaxed as needed.
    void load_pascal_data_for_id( string const & img_id, bool load_img, uint32_t const in_file_ix, bool check_ix_only );
    string get_id_for_img_ix( uint32_t const img_ix );
    uint32_t get_ix_for_img_id( string const & img_id );
    match_res_t try_match( string const & class_name, p_per_class_scored_dets_t scored_dets, scored_det_t const & sd );
    void score_results_for_class( p_per_class_scored_dets_t name_scored_dets, 
				  string const & prc_txt_fn, string const & prc_png_fn );
    void score_results( p_vect_p_per_class_scored_dets_t name_scored_dets_map, 
			string const & prc_fn, string const & plot_base_fn, bool const pre_merge_post_clear );

  };
  typedef shared_ptr< img_db_t > p_img_db_t;


  struct load_pil_t : virtual public nesi, public has_main_t // NESI(help="load PASCAL VOC images by classes",
		      // bases=["has_main_t"], type_id="load_pil")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t pascal_classes_fn; //NESI(default="%(boda_test_dir)/pascal/head_10/pascal_classes.txt",help="file with list of pascal classes to load")
    p_img_db_t img_db; //NESI(default="()", help="image database")
    filename_t pil_fn; //NESI(default="%(boda_test_dir)/pascal/head_10/%%s.txt",help="format for filenames of image list files. %%s will be replaced with the class name")

    p_vect_string classes;
    void load_all_imgs( void );
    virtual void main( nesi_init_arg_t * nia ) { load_all_imgs(); }
  };
  typedef shared_ptr< load_pil_t > p_load_pil_t; 


}
#endif // RESULTS_IO_H
