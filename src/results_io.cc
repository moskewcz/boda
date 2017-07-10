// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"results_io.H"
#include"str_util.H"
#include"geom_prim.H"
#include<cassert>
#include<string>
#include<map>
#include<iostream>
#include<vector>
#include<boost/algorithm/string.hpp>
#include<boost/filesystem.hpp>
#include"xml_util.H"
#include"pyif.H"
#include"img_io.H"
#include"octif.H"
#include"timers.H"
#include"lexp.H"
#include"nesi.H"

namespace boda 
{

  using boost::algorithm::is_space;
  using boost::algorithm::token_compress_on;
  using boost::filesystem::path;
  using boost::filesystem::filesystem_error;
  using std::cout;
  using namespace pugi;

  typedef vector< string > vect_string;
  typedef map< string, uint32_t > str_uint32_t_map_t;

  void img_info_load_img( p_img_info_t img_info, string const & img_fn ) {
    img_info->full_fn = img_fn;
    img_info->img.reset( new img_t );
    img_info->img->load_fn( img_info->full_fn.c_str() );
  }

  u32_pt_t img_db_t::get_max_img_sz( void ) const {
    u32_pt_t ret;
    for( vect_p_img_info_t::const_iterator i = img_infos.begin(); i != img_infos.end(); ++i ) {
      if( (*i) && (*i)->img ) { ret.max_eq( (*i)->img->sz ); }
    }
    return ret;
  }

  string id_from_image_fn( string const & image )
  {
    size_t const last_dot = image.find_last_of('.');	
    string id = image.substr(0, last_dot);
    size_t const last_slash = id.find_last_of("/\\");
    if (last_slash != string::npos) {
      id = id.substr(last_slash + 1);
    }
    return id;
  }

  std::ostream & operator <<(std::ostream & os, const scored_det_t & v) {
    return os << ((u32_box_t &)v) << "@" << v.img_ix << "=" << v.score; }

  struct scored_det_t_comp_by_inv_score_t { 
    bool operator () ( scored_det_t const & a, scored_det_t const & b ) const { return a.score > b.score; } };
  

  std::ostream & operator <<(std::ostream & os, const gt_det_t & v) {
    return os << ((u32_box_t &)v) << ":T" << v.truncated << "D" << v.difficult; }

  p_vect_scored_det_t per_class_scored_dets_t::get_merged_all_imgs_sds( void ) {
    p_vect_scored_det_t ret( new vect_scored_det_t );
    for( uint32_t i = 0; i != per_img_sds.size(); ++i ) {
      p_vect_base_scored_det_t const & img_sds = per_img_sds[i];
      if( !img_sds ) { continue; }
      for( vect_base_scored_det_t::const_iterator j = img_sds->begin(); j != img_sds->end(); ++j ) {
	ret->push_back( scored_det_t{*j,i} );
      }
    }
    return ret;
  }

//#define PASCAL_LAX_PARSE

  void read_pascal_annotations_for_id( p_img_info_t img_info, path const & ann_dir, string const & id )
  {
    ensure_is_dir( ann_dir.string() );
    path const ann_path = ann_dir / (id + ".xml");
#ifdef PASCAL_LAX_PARSE // allow skipping non-existant annotations
    bool const is_reg_file = boost::filesystem::is_regular_file( ann_path ); 
    if( !is_reg_file ) { return; }
#endif
    ensure_is_regular_file( ann_path.string() );
    char const * const ann_fn = ann_path.c_str();
    xml_document doc;
    xml_node ann = xml_file_get_root( doc, ann_fn );
    // assert_st( ann.name() == string("annotation") ); // true, but we don't really care?
#if 1 // note: not currently needed
    xml_node ann_size = xml_must_decend( ann_fn, ann, "size" );
    img_info->size.d[0] = lc_str_u32( xml_must_decend( ann_fn, ann_size, "width" ).child_value() );
    img_info->size.d[1] = lc_str_u32( xml_must_decend( ann_fn, ann_size, "height" ).child_value() );
    img_info->depth = lc_str_u32( xml_must_decend( ann_fn, ann_size, "depth" ).child_value() );
#endif

    for( xml_node ann_obj = ann.child("object"); ann_obj; ann_obj = ann_obj.next_sibling("object") ) {
      gt_det_t gt_det;
#ifdef PASCAL_LAX_PARSE
      gt_det.truncated = 0;
      if( ann_obj.child("truncated") ) { lc_str_u32( xml_must_decend( ann_fn, ann_obj, "truncated" ).child_value() ); }
      gt_det.difficult = 0;
      if( ann_obj.child("difficult") ) { lc_str_u32( xml_must_decend( ann_fn, ann_obj, "difficult" ).child_value() ); }
#else
      gt_det.truncated = lc_str_u32( xml_must_decend( ann_fn, ann_obj, "truncated" ).child_value() );
      gt_det.difficult = lc_str_u32( xml_must_decend( ann_fn, ann_obj, "difficult" ).child_value() );
#endif

      xml_node ann_obj_bb = xml_must_decend( ann_fn, ann_obj, "bndbox" );
      gt_det.p[0].d[0] = lc_str_u32( xml_must_decend( ann_fn, ann_obj_bb, "xmin" ).child_value() );
      gt_det.p[0].d[1] = lc_str_u32( xml_must_decend( ann_fn, ann_obj_bb, "ymin" ).child_value() );
      gt_det.p[1].d[0] = lc_str_u32( xml_must_decend( ann_fn, ann_obj_bb, "xmax" ).child_value() );
      gt_det.p[1].d[1] = lc_str_u32( xml_must_decend( ann_fn, ann_obj_bb, "ymax" ).child_value() );
      gt_det.from_pascal_coord_adjust();
      assert_st( gt_det.is_strictly_normalized() );
      string ann_obj_name( xml_must_decend( ann_fn, ann_obj, "name" ).child_value() );
#ifdef PASCAL_LAX_PARSE
      boost::algorithm::trim( ann_obj_name ); // trim/remove whitespace/strip_ws 
#endif
      vect_gt_det_t & gt_dets = img_info->gt_dets[ann_obj_name];
      gt_dets.push_back(gt_det);
      if( !gt_det.difficult ) { ++gt_dets.num_non_difficult.v; } 
    }
  }



  string img_db_t::get_id_for_img_ix( uint32_t const img_ix ) {
    assert( img_ix < img_infos.size() );
    return img_infos[img_ix]->id;
  }
  uint32_t img_db_t::get_ix_for_img_id( string const & img_id ) {
    id_to_img_info_map_t::iterator img_info = id_to_img_info_map.find(img_id);
    if( img_info == id_to_img_info_map.end() ) { rt_err("tried to get ix for unloaded img_id '"+img_id+"'"); }
    return img_info->second->ix;
  }

  uint32_t lc_str_double_and_round_to_u32( string const & s ) {
    double const v = lc_str_d( s );
    double rv = round( v );
    // yolo seems to output negative/invalid coords sometimes. here, we clamp them to 1.
    // if( rv < 1.0 ) { rv = 1.0; } 
    assert_st( rv >= 1.0 );
    uint32_t ret = (uint32_t)rv;
    assert_st( double(ret) == rv ); // check that rv is representable as a uint32_t
    return ret;
  }
  
  p_per_class_scored_dets_t read_results_file( p_img_db_t img_db, string const & fn, string const & class_name )
  {
    timer_t t("read_results_file");
    p_ifstream in = ifs_open( fn );  
    string line;
    p_per_class_scored_dets_t scored_dets( new per_class_scored_dets_t( class_name ) );
    while( !ifs_getline( fn, in, line ) )
    {
      vect_string parts;
      split( parts, line, is_space(), token_compress_on );
      if( (parts.size() == 1) && parts[0].empty() ) { continue; } // skip ws-only lines
      assert( parts.size() == 6 );
      scored_det_t scored_det;
      string const img_id = parts[0];
      scored_det.img_ix = img_db->get_ix_for_img_id( img_id );
      scored_det.score = lc_str_d( parts[1] );
      scored_det.p[0].d[0] = lc_str_double_and_round_to_u32( parts[2] );
      scored_det.p[0].d[1] = lc_str_double_and_round_to_u32( parts[3] );
      scored_det.p[1].d[0] = lc_str_double_and_round_to_u32( parts[4] );
      scored_det.p[1].d[1] = lc_str_double_and_round_to_u32( parts[5] );
      scored_det.from_pascal_coord_adjust();
      assert_st( scored_det.is_strictly_normalized() );
      scored_dets->add_det( scored_det );
    }
    return scored_dets;
  }

  void write_results_file( p_img_db_t img_db, string const & fn, p_per_class_scored_dets_t scored_dets ) {
    p_ostream out = ofs_open( fn );  
    for( uint32_t img_ix = 0; img_ix != scored_dets->per_img_sds.size(); ++img_ix ) {
      string const img_id = img_db->get_id_for_img_ix(img_ix);
      p_vect_base_scored_det_t const & img_sds = scored_dets->per_img_sds[img_ix];
      if( !img_sds ) { continue; }
      for( vect_base_scored_det_t::const_iterator i = img_sds->begin(); i != img_sds->end(); ++i ) {
	(*out) << strprintf( "%s %s %s\n", str(img_id).c_str(), str(i->score).c_str(), i->pascal_str().c_str() );
      }
    }
    //out->close();// FIXME: can't do now that out is ostream, not ofstream, but should not be needed: dtor should do it.
  }

  void load_pil_t::load_pascal_data_for_id( string const & img_id, bool load_img, uint32_t const in_file_ix, bool check_ix_only )
  {
    p_img_info_t & img_info = img_db->id_to_img_info_map[img_id];
    if( check_ix_only ) {
      if( !img_info ) { rt_err( "expected image to already be loaded but it was not" ); }
      if( img_info->ix != in_file_ix ) { rt_err( strprintf( "already-loaded image had ix=%s, but expected %s",
							    str(img_info->ix).c_str(), str(in_file_ix).c_str() ) ); }
      return;
    }
    if( img_info ) { rt_err( "tried to load annotations multiple times for id '"+img_id+"'"); }
    img_info.reset( new img_info_t( img_id ) );
    read_pascal_annotations_for_id( img_info, pascal_ann_dir.exp, img_id ); 
    if( load_img ) { read_pascal_image_for_id( img_info ); }

    img_info->ix = img_db->img_infos.size();
    if( img_info->ix != in_file_ix ) { rt_err( strprintf( "newly-loaded image had ix=%s, but expected %s",
							  str(img_info->ix).c_str(), str(in_file_ix).c_str() ) ); }
    img_db->img_infos.push_back( img_info );
    for( name_vect_gt_det_map_t::const_iterator i = img_info->gt_dets.begin(); i != img_info->gt_dets.end(); ++i ) {
      img_db->class_infos[i->first].v += i->second.num_non_difficult.v;
    }
  }

  void load_pil_t::read_pascal_image_list_file( filename_t const & class_pil_fn, bool const load_imgs, bool const check_ix_only ) {
    timer_t t("read_pascal_image_list_file");
    p_ifstream in = ifs_open( class_pil_fn );  
    string line;
    uint32_t in_file_ix = 0; // almost (0 based) line number, but only for non-blank lines
    while( !ifs_getline( class_pil_fn.in, in, line ) )
    {
      vect_string parts;
      split( parts, line, is_space(), token_compress_on );
      if( (parts.size() == 1) && parts[0].empty() ) { continue; } // skip ws-only lines
      if( parts.size() != 2 ) { 
	rt_err( strprintf( "invalid line in image list file '%s': num of parts != 2 after space "
			   "splitting. line was:\n%s", class_pil_fn.in.c_str(), line.c_str() ) ); }
      string const id = parts[0];
      string const pn = parts[1];
      if( (pn != "1") && (pn != "-1") && (pn != "0") ) {
	rt_err( strprintf( "invalid type string in image list file '%s': saw '%s', expected '1', '-1', or '0'.",
			   class_pil_fn.in.c_str(), pn.c_str() ) );
      }
      load_pascal_data_for_id( id, false, in_file_ix, check_ix_only );
      ++in_file_ix;
    }
    if( load_imgs ) {
      vect_string load_errs( img_db->img_infos.size() );
#pragma omp parallel for
      for( uint32_t i = 0; i < img_db->img_infos.size(); ++i ) {
	try {
	  read_pascal_image_for_id( img_db->img_infos[i] ); 
	} catch( rt_exception const & rte ) {
	  load_errs[i] = rte.err_msg;
	}
      }
      for( vect_string::const_iterator i = load_errs.begin(); i != load_errs.end(); ++i ) {
	if( !i->empty() ) { rt_err( *i ); }
      }
    }
  }

  void load_pil_t::flickr_logos_load( void ) {
    p_vect_string fl_list_lines = readlines_fn( fl_list );
    set_string classes_set;
    classes.reset( new vect_string );
    for( vect_string::iterator i = fl_list_lines->begin(); i != fl_list_lines->end(); ++i ) {
      boost::algorithm::trim( *i ); // not ideal, but removes trailing newlines at least (and CRs if present)
      vect_string parts = split(*i,',');
      if( parts.size() != 2 ) { rt_err("failed to parse line in FlickrLogos-style image list. expected exactly 1 ',', had:" + *i ); }
      string const & cn = parts[0];
      string const & fn = parts[1];
      bool did_ins = classes_set.insert( cn ).second;
      if( did_ins ) { classes->push_back( cn ); }
      lexp_name_val_map_t fmt{ p_lexp_t() };
      fmt.insert_leaf( "cn", cn.c_str(), 0 ); 
      fmt.insert_leaf( "fn", fn.c_str(), 0 );
      string const img_fn = nesi_filename_t_expand( &fmt, fl_img.exp );
      string const bboxes_fn = nesi_filename_t_expand( &fmt, fl_bbox.exp );

      string const & img_id = img_fn; // for this mode, use fn as id
      p_img_info_t & img_info = img_db->id_to_img_info_map[img_id];

      if( img_info ) { rt_err( "FlickrLogos-style img_db load: tried to image multiple times: '"+img_id+"'"); }
      img_info.reset( new img_info_t( img_id ) );
      //read_pascal_annotations_for_id( img_info, pascal_ann_dir.exp, img_id ); 
      img_info->full_fn = img_fn;
      img_info->ix = img_db->img_infos.size();
      img_db->img_infos.push_back( img_info );

      // read gts / bboxes (if they exist)
      if( !boost::filesystem::is_regular_file( bboxes_fn ) ) { 
        if( cn == "no-logo" ) { continue; } // for no_logo, can skip loading gts / bboxes if no file
        printf( "for class cn=%s, missing bboxes file:\n", str(cn).c_str() );
      }
      p_vect_string bboxes_lines = readlines_fn( bboxes_fn );
      vect_gt_det_t & gt_dets = img_info->gt_dets[cn];
      for( vect_string::iterator i = bboxes_lines->begin(); i != bboxes_lines->end(); ++i ) {
        boost::algorithm::trim( *i ); // not ideal, but removes trailing newlines at least (and CRs if present)
        gt_det_t gt_det;
        gt_det.truncated = 0;
        gt_det.difficult = 0;
        vect_string box_parts;
        split( box_parts, *i, is_space(), token_compress_on );
        if( (box_parts.size() == 1) && parts[0].empty() ) { continue; } // skip ws-only lines
        assert( box_parts.size() == 4 );
        if( box_parts[0] == "x" ) { continue; } // skip header
        gt_det.p[0].read_from_line_parts( box_parts, 0 ); gt_det.p[0] -= u32_pt_t{1,1}; // 1-based, so adjust
        gt_det.p[1].read_from_line_parts( box_parts, 2 ); gt_det.p[1] += gt_det.p[0]; // read as size, so add nc to make pc	  
        assert_st( gt_det.is_strictly_normalized() );
        gt_dets.push_back(gt_det);
        if( !gt_det.difficult ) { ++gt_dets.num_non_difficult.v; } 
      }
      img_db->class_infos[cn].v += gt_dets.num_non_difficult.v;
    }
  }

  void load_pil_t::darknet_load( void ) {

    p_vect_string fl_list_lines = readlines_fn( darknet_imgs_fn );
    set_string classes_set;
    classes.reset( new vect_string );
    for( vect_string::iterator i = fl_list_lines->begin(); i != fl_list_lines->end(); ++i ) {
      string const & img_fn = (*i);
      string const & img_id = img_fn; // for this mode, use fn as id
      p_img_info_t & img_info = img_db->id_to_img_info_map[img_id];
      if( img_info ) { rt_err( "darknet img_db load: tried to image multiple times: '"+img_id+"'"); }
      img_info.reset( new img_info_t( img_id ) );
      img_info->full_fn = img_fn;
      img_info->ix = img_db->img_infos.size();
      img_db->img_infos.push_back( img_info );
      
    }
  }

  void load_pil_t::load_img_db( bool const load_imgs ) {
    if( load_mode == "pascal" ) { // pascal load; handle first and return, since it loads imgs itself in subfunction
      // note: this assumes (and checks) that all the per-class file lists
      // are identical. thus, it loads all images and annotations from the
      // first class's image list file, and then just verifies that the
      // other class image list files have the same set (or at least a
      // subset) of image ids of the first file.
      classes = readlines_fn( pascal_classes_fn );
      for( vect_string::const_iterator i = classes->begin(); i != classes->end(); ++i ) {
	bool const is_first_class = (i == classes->begin());
	read_pascal_image_list_file( filename_t_printf( pil_fn, (*i).c_str() ), load_imgs && is_first_class, !is_first_class );
      }
      return; // note: does not fall though to loading, since loading done above
    }

    // all other modes use below || iamge loading block after filling in img_infos
    if( 0 ) { }
    else if( load_mode == "flickr" ) { flickr_logos_load(); }
    else if( load_mode == "darknet" ) { darknet_load(); }
    else { rt_err( "unknown load_img_db mode: " + load_mode ); }

    if( load_imgs ) {
#pragma omp parallel for
      for( uint32_t i = 0; i < img_db->img_infos.size(); ++i ) { 
        p_img_info_t const & img_info = img_db->img_infos[i];
        img_info_load_img( img_info, img_info->full_fn );
      }
    }
    printf( "(*classes)=%s\n", str((*classes)).c_str() );
    
  }

  void load_pil_t::show_dets( p_per_class_scored_dets_t scored_dets, uint32_t img_ix ) {
    assert_st( img_ix < img_db->img_infos.size() );
    p_img_info_t img_info = img_db->img_infos[img_ix];
    if( !img_info->img ) { read_pascal_image_for_id( img_info ); }
    boda::show_dets( img_info->img, *scored_dets->get_per_img_sds( img_ix, 1 ) ); // FIXME: rename boda::show_dets?
  }

  void load_pil_t::read_pascal_image_for_id( p_img_info_t img_info ) { 
    img_info_load_img( img_info, filename_t_printf( pascal_img_fn, img_info->id.c_str() ).exp );
  }


  struct score_results_file_t : virtual public nesi, public load_pil_t // NESI(help="score a pascal-VOC-format results file",bases=["load_pil_t"], type_id="score")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t res_fn; //NESI(help="input: name of pascal-VOC format detection results file to read",req=1)
    filename_t prc_txt_fn; //NESI(default="%(boda_output_dir)/prc_",help="output: text prc curve base filename")
    filename_t prc_png_fn; //NESI(default="%(boda_output_dir)/mAP_",help="output: png prc curve base filename")
    virtual void main( nesi_init_arg_t * nia ) {
      load_img_db( 0 );
      assert_st( classes->size() == 1 ); // FIXME: only expects/handled single class
      p_per_class_scored_dets_t scored_dets = read_results_file( img_db, res_fn.exp, classes->front() );
      img_db->score_results_for_class( scored_dets, prc_txt_fn.exp, prc_png_fn.exp );
    }
  };

  struct score_results_files_t : virtual public nesi, public load_pil_t // NESI(help="score a set of pascal-VOC-format results files",bases=["load_pil_t"], type_id="score-files")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t res_fn; //NESI(default="%(bench_dir)/results/%%s_test.txt",help="format for filenames of pascal-VOC format DPM detection results files. %%s will be replaced with the class name")
    filename_t summary_fn; //NESI(default="%(boda_output_dir)/summary.txt",help="output: all-classes text summary filename")
    filename_t prc_txt_fn; //NESI(default="%(boda_output_dir)/prc_",help="output: text prc curve base filename")
    filename_t prc_png_fn; //NESI(default="%(boda_output_dir)/mAP_",help="output: png prc curve base filename")
    virtual void main( nesi_init_arg_t * nia ) {
      load_img_db( 0 );
      p_vect_p_per_class_scored_dets_t scored_dets( new vect_p_per_class_scored_dets_t );
      
      for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
	scored_dets->push_back( read_results_file( img_db, strprintf( res_fn.exp.c_str(), (*i).c_str() ), *i ) );
	printf( "(*i)=%s\n", str((*i)).c_str() );
      }
      img_db->score_results( scored_dets, prc_txt_fn.exp, prc_png_fn.exp, summary_fn.exp, 0 );
    }
  };

  // returns (matched, match_was_difficult). note that difficult is
  // only seems meaningfull when matched=true, although it is
  // sometimes valid when matched=false (i.e. if the match is false
  // due to being a duplicate, the difficult bit indicates if the
  // higher-scoring-matched gt detection was marked difficult).
  match_res_t img_db_t::try_match( string const & class_name, p_per_class_scored_dets_t scored_dets, scored_det_t const & sd )
  {
    assert_st( sd.img_ix < img_infos.size() );
    p_img_info_t img_info = img_infos[sd.img_ix];
    vect_gt_det_t & gt_dets = img_info->gt_dets[class_name]; // note: may be created here (i.e. may be empty)
    vect_gt_match_t & gtms = scored_dets->get_gtms( sd.img_ix, gt_dets.size() ); // note: may be created here

    uint64_t const sd_area = sd.get_area();
    u32_pt_t best_score(0,1); // as a fraction [0]/[1] --> num / dem
    gt_match_t * best = 0;
    uint32_t best_difficult = uint32_t_const_max; // cached difficult bit from the gt associated with best
    for( vect_gt_det_t::iterator i = gt_dets.begin(); i != gt_dets.end(); ++i )
    {
      uint64_t const gt_area = i->get_area();
      uint64_t const gt_overlap = i->get_overlap_with( sd );
      uint64_t const gt_sd_union = sd_area + gt_area - gt_overlap;
      // must overlap by 50% in terms of ( intersection area / union area ) (pascal VOC criterion) 
      bool has_min_overlap = ( gt_overlap * 2 ) >= gt_sd_union;
#if 0
      if( img_info->id == "xx001329" )
      {
	printf( "sd=%s gt=%s sd_area=%s gt_area=%s gt_overlap=%s gt_score=%s max_score=%s hmo=%s\n", str(sd).c_str(), str(*i).c_str(),str(sd_area).c_str(), str(gt_area).c_str(), str(gt_overlap).c_str(), str(double(gt_overlap)/gt_sd_union).c_str(), str(double(best_score.d[0])/best_score.d[1]).c_str(), str(has_min_overlap).c_str() );
      }
#endif
      if( !has_min_overlap ) { continue; }

      if( (uint64_t(gt_overlap)*best_score.d[1]) > (uint64_t(best_score.d[0])*gt_sd_union) ) {
	best_score.d[0] = gt_overlap; best_score.d[1] = gt_sd_union; best_difficult = i->difficult;
	best = &gtms.at(i-gt_dets.begin()); }
    }
    if( !best_score.d[0] ) { return match_res_t(0,0); }// no good overlap with any gt_det. note: difficult is unknown here
    if( best->matched ) { assert_st( best->match_score >= sd.score ); return match_res_t(0,best_difficult); }
    best->matched = 1; 
    best->match_score = sd.score;
    return match_res_t(1,best_difficult);
  }
  
  void print_prc_line( p_ostream const & out, prc_elem_t const & prc_elem, uint32_t const tot_num_class, double const & map )
  {
    (*out) << strprintf( "num_pos=%s num_test=%s score=%.6lf p=%s r=%s map=%s\n", 
			 str(prc_elem.num_pos).c_str(), str(prc_elem.num_test).c_str(), 
			 prc_elem.score,
			 str(prc_elem.get_precision()).c_str(), 
			 str(prc_elem.get_recall(tot_num_class)).c_str(), 
			 str(map).c_str() );
  }


  double img_db_t::score_results_for_class( p_per_class_scored_dets_t name_scored_dets,
                                            string const & prc_txt_fn, string const & prc_png_fn )
  {
    timer_t t("score_results_for_class");
    string const & class_name = name_scored_dets->class_name;
    //printf( "class_name=%s\n", str(class_name).c_str() );
    assert_st( !class_name.empty() );
    p_ostream prc_out = ofs_open( prc_txt_fn + class_name + ".txt" );
    p_vect_scored_det_t all_sds = name_scored_dets->get_merged_all_imgs_sds();
    sort( all_sds->begin(), all_sds->end(), scored_det_t_comp_by_inv_score_t() );
    uint32_t tot_num_class = class_infos[class_name].v;
    vect_prc_elem_t prc_elems;
    uint32_t num_pos = 0;
    uint32_t num_test = 0;
    double map = 0; // FIXME: misnamed, should be just 'ap'
    uint32_t print_skip = 1 + (tot_num_class / 20); // print about 20 steps in recall
    uint32_t next_print = 1;
    (*prc_out ) << strprintf( "---BEGIN--- class_name=%s tot_num_class=%s name_scored_dets->size()=%s\n", str(class_name).c_str(), str(tot_num_class).c_str(), str(all_sds->size()).c_str() );
    for( vect_scored_det_t::const_iterator i = all_sds->begin(); i != all_sds->end(); )
    {
      uint32_t const orig_num_pos = num_pos;
      double const cur_score = i->score;
      while( 1 ) // handle all dets with equal score as a unit
      {
	match_res_t const match_res = try_match( class_name, name_scored_dets, *i );
	if( match_res.is_pos && !match_res.is_diff ) { 
	  ++num_test; ++num_pos; // positive, recall increased (precision must == 1 or also be increased)
	} else if( !match_res.is_pos ) { ++num_test; } // negative, precision decreased
	++i; if( (i == all_sds->end()) || (cur_score != i->score) ) { break; }
      }
      if( orig_num_pos != num_pos ) // recall increased
      {
	prc_elem_t const prc_elem( num_pos, num_test, cur_score );
	map += prc_elem.get_precision() * (num_pos - orig_num_pos);
	if( num_pos >= next_print ) { 
	  next_print = num_pos + print_skip; 
	  print_prc_line( prc_out, prc_elem, tot_num_class, map / tot_num_class ); 
	}
	prc_elems.push_back( prc_elem );
      }
    }
    map /= tot_num_class;
    if( (next_print != (num_pos+print_skip)) && (!prc_elems.empty()) ) {
      print_prc_line( prc_out, prc_elems.back(), tot_num_class, map ); }
    (*prc_out ) << strprintf( "---END--- class_name=%s tot_num=%s num_pos=%s num_test=%s num_neg=%s final_map=%s\n", 
			      str(class_name).c_str(),
			      str(tot_num_class).c_str(), 
			      str(num_pos).c_str(), str(num_test).c_str(), 
			      str(num_test - num_pos).c_str(),
			      str(map).c_str() );
    if( !prc_png_fn.empty() ) {
      string const plt_fn = prc_png_fn+class_name+".png";
      prc_plot( plt_fn, tot_num_class, prc_elems, 
		strprintf( "class_name=%s map=%s\n", str(class_name).c_str(), str(map).c_str() ) );
    }
    return map;
  }

  void img_db_t::score_results( p_vect_p_per_class_scored_dets_t name_scored_dets_map,
				string const & prc_fn, string const & plot_base_fn,
                                string const & summary_fn,
				bool const pre_merge_post_clear ) {
    timer_t t("score_results");
    double mean_ap = 0.0;
    p_ostream summary_out;
    if( !summary_fn.empty() ) { summary_out = ofs_open( summary_fn ); }
    for( vect_p_per_class_scored_dets_t::iterator i = name_scored_dets_map->begin(); i != name_scored_dets_map->end(); ++i ) {
      double const class_ap = score_results_for_class( *i, prc_fn, plot_base_fn );
      if( summary_out ) { (*summary_out) << strprintf( "class_name=%s ap=%s\n", str((*i)->class_name).c_str(), str(class_ap).c_str() ); }
      mean_ap += class_ap;
      if( pre_merge_post_clear ) { (*i).reset(); }
    }
    mean_ap /= name_scored_dets_map->size();
    if( summary_out ) { (*summary_out) << strprintf( "all classes mean_ap=%s\n", str(mean_ap).c_str() ); }
  }

  struct is_comma { bool operator()( char const & c ) const { return c == ','; } };

  void read_hamming_csv_file( p_per_class_scored_dets_t scored_dets, string const & fn, uint32_t const img_ix )
  {
    p_ifstream in = ifs_open( fn );  
    string line;
    while( !ifs_getline( fn, in, line ) )
    {
      vect_string parts;
      split( parts, line, is_comma(), token_compress_on );
      if( (parts.size() == 1) && parts[0].empty() ) { continue; } // skip ws-only lines
      assert( parts.size() == 5 );
      scored_det_t scored_det;
      scored_det.img_ix = img_ix;
      scored_det.p[0].d[0] = uint32_t( lc_str_d( parts[0] ) );
      scored_det.p[0].d[1] = uint32_t( lc_str_d( parts[1] ) );
      scored_det.p[1].d[0] = uint32_t( lc_str_d( parts[2] ) );
      scored_det.p[1].d[1] = uint32_t( lc_str_d( parts[3] ) );
      scored_det.score = lc_str_d( parts[4] );
      scored_det.from_pascal_coord_adjust();
      
      assert_st( scored_det.is_strictly_normalized() );
      scored_dets->add_det( scored_det );
    }
  }


  struct hamming_analysis_t : virtual public nesi, public load_pil_t // NESI(help="hamming first-level cascade boxes analysis",bases=["load_pil_t"], type_id="ham_ana")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t ham_fn; //NESI(default="%(bench_dir)/hamming/voc-release5-hamming_top1000boxes/2007/%%s_boxes_test__hamming_imgNo%%s.csv",help="format for base filenames of hamming boxes. %%s, %%s will be replaced with the class name, img index")
    filename_t dpm_fn; //NESI(default="%(bench_dir)/hamming/pf_shog/%%s_test.txt",help="format for filenames of pascal-VOC format DPM detection results files. %%s will be replaced with the class name")
    filename_t prc_txt_fn; //NESI(default="%(boda_output_dir)/prc_",help="output: text prc curve base filename")
    filename_t prc_png_fn; //NESI(default="%(boda_output_dir)/mAP_",help="output: png prc curve base filename")

    filename_t score_diff_summary_fn; //NESI(default="%(boda_output_dir)/diff_summ.csv",help="output: text summary of differences between two scored sets.")
    
    virtual void main( nesi_init_arg_t * nia ) { 
      load_img_db( 0 );
      p_vect_p_per_class_scored_dets_t hamming_scored_dets( new vect_p_per_class_scored_dets_t );
      p_vect_p_per_class_scored_dets_t dpm_scored_dets( new vect_p_per_class_scored_dets_t );
      
      for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
	hamming_scored_dets->push_back( p_per_class_scored_dets_t( new per_class_scored_dets_t( *i ) ) );
	for (uint32_t ix = 0; ix < img_db->img_infos.size(); ++ix) {
	  read_hamming_csv_file( hamming_scored_dets->back(), 
				 strprintf( ham_fn.exp.c_str(), (*i).c_str(), str(ix+1).c_str() ), ix );
	}
	printf( "(*i)=%s (hamming)\n", str((*i)).c_str() );
	dpm_scored_dets->push_back( read_results_file( img_db, strprintf( dpm_fn.exp.c_str(), (*i).c_str() ), *i ) );
	printf( "(*i)=%s (DPM)\n", str((*i)).c_str() );
      }
      // FIXME: no summaries output here, since this code predated summary-writing code in score_results()
      img_db->score_results( hamming_scored_dets, prc_txt_fn.exp + "ham_", prc_png_fn.exp + "ham_", "", 0);
      img_db->score_results( dpm_scored_dets, prc_txt_fn.exp + "dpm_", prc_png_fn.exp + "dpm_", "", 0 );
#if 1      
      p_ostream summ_out = ofs_open( score_diff_summary_fn.exp );  
      (*summ_out) << strprintf( "class_name,num_tot,ham_only,dpm_only,num_ham,num_dpm,num_both,num_either,num_neither,\n");
      for( uint32_t cix = 0; cix != classes->size(); ++cix ) {
	string const & class_name = classes->at(cix);;
	p_per_class_scored_dets_t ham_sds = hamming_scored_dets->at(cix);
	p_per_class_scored_dets_t dpm_sds = dpm_scored_dets->at(cix);
	uint32_t ham_only = 0;
	uint32_t dpm_only = 0;
	uint32_t num_ham = 0;
	uint32_t num_dpm = 0;
	uint32_t num_both = 0;
	uint32_t num_neither = 0;
	uint32_t num_tot = 0;
	for (uint32_t ix = 0; ix < img_db->img_infos.size(); ++ix) {
	  p_img_info_t img_info = img_db->img_infos[ix];
	  vect_gt_det_t & gt_dets = img_info->gt_dets[class_name];
	  vect_gt_match_t & ham_gtms = ham_sds->get_gtms( ix, gt_dets.size() ); // must already exist, but not checked
	  vect_gt_match_t & dpm_gtms = dpm_sds->get_gtms( ix, gt_dets.size() ); // must already exist, but not checked
	  bool unmatched = 0;
	  for( uint32_t gtix = 0; gtix < gt_dets.size(); ++gtix ) {
	    ++num_tot;
	    if( ham_gtms.at(gtix).matched ) { ++num_ham; }
	    if( dpm_gtms.at(gtix).matched ) { ++num_dpm; }
	    if( ham_gtms.at(gtix).matched != dpm_gtms.at(gtix).matched ) {
	      if( ham_gtms.at(gtix).matched ) { ++ham_only; }
	      else { ++dpm_only; }
#if 0
	      printf( "unmatched: H=%s D=%s gtix=%s class_name=%s img_info->id=%s\n", 
		      str(ham_gtms.at(gtix).matched).c_str(),
		      str(dpm_gtms.at(gtix).matched).c_str(),
		      str(gtix).c_str(),
		      str(class_name).c_str(), str(img_info->id).c_str() );
#endif
	      unmatched = 1;
	    } else {
	      if( !ham_gtms.at(gtix).matched ) { ++num_neither; }
	      else { ++num_both; }
	    }
	  }
	  bool const show_mismatch_img = 0;
	  if( unmatched && show_mismatch_img ) { show_dets( ham_sds, ix ); }
	}
	assert_st( (num_ham+dpm_only) == (num_dpm+ham_only) );
	assert_st( (num_ham+dpm_only) == (num_dpm+ham_only) );
	uint32_t const num_either = num_ham+dpm_only;
	assert_st( num_neither + num_either == num_tot );
	// class num_tot ham_only dpm_only num_ham num_dpm num_both num_either num_neither
	(*summ_out) << strprintf( "%s,%s,%s,%s,%s,%s,%s,%s,%s,\n", str(class_name).c_str(), str(num_tot).c_str(), str(ham_only).c_str(), str(dpm_only).c_str(), str(num_ham).c_str(), str(num_dpm).c_str(), str(num_both).c_str(), str(num_either).c_str(), str(num_neither).c_str() );
      }
#endif
    }
  };

#include"gen/results_io.H.nesi_gen.cc"
#include"gen/results_io.cc.nesi_gen.cc"

}
