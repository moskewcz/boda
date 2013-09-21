#include"boda_tu_base.H"
#include"results_io.H"
#include"str_util.H"
#include"geom_prim.H"
#include<cassert>
#include<string>
#include<map>
#include<vector>
#include<boost/algorithm/string.hpp>
#include<boost/lexical_cast.hpp>
#include<boost/filesystem.hpp>
#include"pugixml.hpp"
#include"pyif.H"

namespace boda 
{
  using namespace std;
  using namespace boost;
  using filesystem::path;
  using filesystem::filesystem_error;
  using namespace pugi;

  double lc_str_d( char const * const s )
  { 
    try { return lexical_cast< double >( s ); }
    catch( bad_lexical_cast & e ) { rt_err( strprintf("can't convert '%s' to double.", s ) ); }
  }
  uint32_t lc_str_u32( char const * const s )
  { 
    try { return lexical_cast< uint32_t >( s ); }
    catch( bad_lexical_cast & e ) { rt_err( strprintf("can't convert '%s' to uint32_t.", s ) ); }
  }
  double lc_str_d( string const & s ) { return lc_str_d( s.c_str() ); } 
  uint32_t lc_str_u32( string const & s ) { return lc_str_u32( s.c_str() ); } 

  typedef vector< string > vect_string;
  typedef map< string, uint32_t > str_uint32_t_map_t;
  string id_from_image_fn( string const & image )
  {
    const size_t lastDot = image.find_last_of('.');	
    string id = image.substr(0, lastDot);
    const size_t lastSlash = id.find_last_of("/\\");
    if (lastSlash != string::npos)
      id = id.substr(lastSlash + 1);
    return id;
  }

  // clears line and reads one line from in. returns true if at EOF. 
  // note: calls rt_err() if a complete line cannot be read.
  bool ifs_getline( std::string const &fn, p_ifstream in, string & line )
  {
    line.clear();
    // the file should initially be good (including if we just
    // opened it).  note the eof is not set until trying to read
    // past the end. after each line is read, we check for eof, and
    // if we're not at eof, we check that the stream is still good
    // for more reading.
    assert_st( in->good() ); 
    getline(*in, line);
    if( in->eof() ) { 
      if( !line.empty() ) { rt_err( "reading "+fn+": incomplete (no newline) line at EOF:'" + line + "'" ); } 
      return 1;
    }
    else {
      if( !in->good() ) { rt_err( "reading "+fn+ " unknown failure" ); }
      return 0;
    }
  }

  struct scored_det_t : public u32_box_t
  {
    double score;
    uint32_t img_ix;
    scored_det_t( void ) : score(0), img_ix(0) { }
  };
  std::ostream & operator<<(std::ostream & os, const scored_det_t & v) {
    return os << ((u32_box_t &)v) << "@" << v.img_ix << "=" << v.score; }
  typedef vector< scored_det_t > vect_scored_det_t;
  typedef map< string, vect_scored_det_t > name_vect_scored_det_map_t;
  typedef shared_ptr< vect_scored_det_t > p_vect_scored_det_t;
  struct scored_det_t_comp_by_inv_score_t { 
    bool operator () ( scored_det_t const & a, scored_det_t const & b ) const { return a.score > b.score; } };
  
  struct gt_det_t : public u32_box_t
  {
    uint32_t truncated;
    uint32_t difficult;    
    // temporary data using during evaluation
    bool matched;
    double match_score;
    gt_det_t( void ) : truncated(0), difficult(0), matched(0), match_score(0) { }
  };
  std::ostream & operator<<(std::ostream & os, const gt_det_t & v) {
    return os << ((u32_box_t &)v) << ":T" << v.truncated << "D" << v.difficult; }
  struct vect_gt_det_t : public vector< gt_det_t >
  {
    zi_uint32_t num_non_difficult;
  };
  typedef map< string, vect_gt_det_t > name_vect_gt_det_map_t;

  struct img_info_t
  {
    string id;
    uint32_t ix;
    u32_pt_t size;
    uint32_t depth;
    name_vect_gt_det_map_t gt_dets;
    img_info_t( string const & id_ ) : id(id_), ix( uint32_t_const_max ), depth(0) { }
  };
  typedef shared_ptr< img_info_t > p_img_info_t;
  typedef map< string, p_img_info_t > id_to_img_info_map_t;
  typedef vector< p_img_info_t > vect_p_img_info_t;

  xml_node xml_must_decend( char const * const fn, xml_node const & node, char const * const child_name )
  {
    xml_node ret = node.child(child_name);
    if( !ret ) { 
      rt_err( strprintf( "error: parsing xml file: '%s': expected to find child named '%s' from %s",
			 fn, child_name, (node==node.root()) ? "document root" : 
			 ("node with name '" + string(node.name()) + "'").c_str() ) ); }
    return ret;
  }
  
  p_img_info_t read_pascal_annotations_for_id( path const & pascal_base_path, string const & id )
  {
    path const ann_dir = pascal_base_path / "Annotations";
    ensure_is_dir( ann_dir );
    path const ann_path = ann_dir / (id + ".xml");
    ensure_is_regular_file( ann_path );
    char const * const ann_fn = ann_path.c_str();
    xml_document doc;
    xml_parse_result result = doc.load_file( ann_fn );
    if( !result ) { 
      rt_err( strprintf( "loading xml file '%s' failed: %s", ann_fn, result.description() ) );
    }
    p_img_info_t img_info( new img_info_t( id ) );
    
    xml_node ann = xml_must_decend( ann_fn, doc, "annotation" );
    xml_node ann_size = xml_must_decend( ann_fn, ann, "size" );
    img_info->size.d[0] = lc_str_u32( xml_must_decend( ann_fn, ann_size, "width" ).child_value() );
    img_info->size.d[1] = lc_str_u32( xml_must_decend( ann_fn, ann_size, "height" ).child_value() );
    img_info->depth = lc_str_u32( xml_must_decend( ann_fn, ann_size, "depth" ).child_value() );

    for( xml_node ann_obj = ann.child("object"); ann_obj; ann_obj = ann_obj.next_sibling("object") ) {
      gt_det_t gt_det;
      gt_det.truncated = lc_str_u32( xml_must_decend( ann_fn, ann_obj, "truncated" ).child_value() );
      gt_det.difficult = lc_str_u32( xml_must_decend( ann_fn, ann_obj, "difficult" ).child_value() );
      xml_node ann_obj_bb = xml_must_decend( ann_fn, ann_obj, "bndbox" );
      gt_det.p[0].d[0] = lc_str_u32( xml_must_decend( ann_fn, ann_obj_bb, "xmin" ).child_value() );
      gt_det.p[0].d[1] = lc_str_u32( xml_must_decend( ann_fn, ann_obj_bb, "ymin" ).child_value() );
      gt_det.p[1].d[0] = lc_str_u32( xml_must_decend( ann_fn, ann_obj_bb, "xmax" ).child_value() );
      gt_det.p[1].d[1] = lc_str_u32( xml_must_decend( ann_fn, ann_obj_bb, "ymax" ).child_value() );
      gt_det.from_pascal_coord_adjust();
      assert_st( gt_det.is_strictly_normalized() );
      string const ann_obj_name( xml_must_decend( ann_fn, ann_obj, "name" ).child_value() );
      vect_gt_det_t & gt_dets = img_info->gt_dets[ann_obj_name];
      gt_dets.push_back(gt_det);
      if( !gt_det.difficult ) { ++gt_dets.num_non_difficult.v; } 
    }
    return img_info;
  }

  typedef map< string, zi_uint32_t > class_infos_t;

  struct match_res_t
  {
    bool is_pos;
    bool is_diff;
    match_res_t( bool const is_pos_, bool const is_diff_ ) : is_pos(is_pos_), is_diff(is_diff_) { }
  };

  struct img_db_t 
  {
    path pascal_base_path;
    id_to_img_info_map_t id_to_img_info_map;
    vect_p_img_info_t img_infos;
    name_vect_scored_det_map_t scored_dets;
    class_infos_t class_infos;

    img_db_t( void ) : pascal_base_path( "/home/moskewcz/bench/VOCdevkit/VOC2007" ) { }
    void load_ann_for_id( string const & img_id )
    {
      p_img_info_t & img_info = id_to_img_info_map[img_id];
      if( img_info ) { rt_err( "tried to load annotations multiple times for id '"+img_id+"'"); }
      img_info = read_pascal_annotations_for_id( pascal_base_path, img_id ); 
      img_info->ix = img_infos.size();
      img_infos.push_back( img_info );
      for( name_vect_gt_det_map_t::const_iterator i = img_info->gt_dets.begin(); i != img_info->gt_dets.end(); ++i )
      {
	class_infos[i->first].v += i->second.num_non_difficult.v;
      }
    }
    uint32_t get_ix_for_img_id( string const & img_id )
    {
      id_to_img_info_map_t::iterator img_info = id_to_img_info_map.find(img_id);
      if( img_info == id_to_img_info_map.end() ) { rt_err("tried to get ix for unloaded img_id '"+img_id+"'"); }
      return img_info->second->ix;
    }
    void add_scored_det( string const & class_name, scored_det_t const & scored_det )
    {
      scored_dets[class_name].push_back( scored_det );
    }
    match_res_t try_match( string const & class_name, scored_det_t const & sd );
    void score_results_for_class( string const & class_name, vect_scored_det_t & name_scored_dets );
    void score_results( void );
  };
  typedef shared_ptr< img_db_t > p_img_db_t;

  void read_results_file( p_img_db_t img_db, string const & fn, string const &class_name )
  {
    p_ifstream in = ifs_open( fn );  
    string line;
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
      scored_det.p[0].d[0] = lc_str_u32( parts[2] );
      scored_det.p[0].d[1] = lc_str_u32( parts[3] );
      scored_det.p[1].d[0] = lc_str_u32( parts[4] );
      scored_det.p[1].d[1] = lc_str_u32( parts[5] );
      scored_det.from_pascal_coord_adjust();
      assert_st( scored_det.is_strictly_normalized() );
      img_db->add_scored_det( class_name, scored_det );
    }
    img_db->score_results();
  }

  p_img_db_t read_image_list_file( string const & fn )
  {
    p_img_db_t img_db( new img_db_t );
    p_ifstream in = ifs_open( fn );  
    string line;
    while( !ifs_getline( fn, in, line ) )
    {
      vect_string parts;
      split( parts, line, is_space(), token_compress_on );
      if( (parts.size() == 1) && parts[0].empty() ) { continue; } // skip ws-only lines
      if( parts.size() != 2 ) { 
	rt_err( strprintf( "invalid line in image list file '%s': num of parts != 2 after space "
			   "splitting. line was:\n%s", fn.c_str(), line.c_str() ) ); }
      string const id = parts[0];
      string const pn = parts[1];
      if( (pn != "1") && (pn != "-1") && (pn != "0") ) {
	rt_err( strprintf( "invalid type string in image list file '%s': saw '%s', expected '1', '-1', or '0'.",
			   fn.c_str(), pn.c_str() ) );
      }
      img_db->load_ann_for_id( id );
    }
    return img_db;
  }

  void score_results_file( string const & il_fn, string const & res_fn, string const &class_name )
  {
    p_img_db_t img_db = read_image_list_file( il_fn );
    read_results_file( img_db, res_fn, class_name );
  }

  // returns (matched, match_was_difficult). note that difficult is
  // only seems meaningfull when matched=true, although it is
  // sometimes valid when matched=false (i.e. if the match is false
  // due to being a duplicate, the difficult bit indicates if the
  // higher-scoring-matched gt detection was marked difficult).
  match_res_t img_db_t::try_match( string const & class_name, scored_det_t const & sd )
  {
    assert_st( sd.img_ix < img_infos.size() );
    p_img_info_t img_info = img_infos[sd.img_ix];
    vect_gt_det_t & gt_dets = img_info->gt_dets[class_name]; // note: may be created here (i.e. may be empty)
    uint64_t const sd_area = sd.get_area();
    u32_pt_t best_score(0,1); // as a fraction [0]/[1] --> num / dem
    vect_gt_det_t::iterator best = gt_dets.end();
    for( vect_gt_det_t::iterator i = gt_dets.begin(); i != gt_dets.end(); ++i )
    {
      uint64_t const gt_area = i->get_area();
      uint64_t const gt_overlap = i->get_overlap_with( sd );
      uint64_t const gt_sd_union = sd_area + gt_area - gt_overlap;
      // must overlap by 50% in terms of ( intersection area / union area ) (pascal VOC criterion) 
      bool has_min_overlap = ( gt_overlap * 2 ) >= gt_sd_union;
      if( img_info->id == "xx001329" )
      {
	printf( "sd=%s gt=%s sd_area=%s gt_area=%s gt_overlap=%s gt_score=%s max_score=%s hmo=%s\n", str(sd).c_str(), str(*i).c_str(),str(sd_area).c_str(), str(gt_area).c_str(), str(gt_overlap).c_str(), str(double(gt_overlap)/gt_sd_union).c_str(), str(double(best_score.d[0])/best_score.d[1]).c_str(), str(has_min_overlap).c_str() );
      }
      if( !has_min_overlap ) { continue; }

      if( (uint64_t(gt_overlap)*best_score.d[1]) > (uint64_t(best_score.d[0])*gt_sd_union) ) {
	best_score.d[0] = gt_overlap; best_score.d[1] = gt_sd_union; best = i; }
    }
    if( !best_score.d[0] ) { return match_res_t(0,0); }// no good overlap with any gt_det. note: difficult is unknown here
    if( best->matched ) { assert_st( best->match_score >= sd.score ); return match_res_t(0,best->difficult); }
    best->matched = 1; 
    best->match_score = sd.score;
    return match_res_t(1,best->difficult);
  }
  
  void print_prc_line( prc_elem_t const & prc_elem, uint32_t const tot_num_class, double const & map )
  {
    printf( "num_pos=%s num_test=%s score=%.6lf p=%s r=%s map=%s\n", 
	    str(prc_elem.num_pos).c_str(), str(prc_elem.num_test).c_str(), 
	    prc_elem.score,
	    str(prc_elem.get_precision()).c_str(), 
	    str(prc_elem.get_recall(tot_num_class)).c_str(), 
	    str(map).c_str() );
  }


  void img_db_t::score_results_for_class( string const & class_name, vect_scored_det_t & name_scored_dets )
  {
    sort( name_scored_dets.begin(), name_scored_dets.end(), scored_det_t_comp_by_inv_score_t() );
    uint32_t tot_num_class = class_infos[class_name].v;
    vect_prc_elem_t prc_elems;
    uint32_t num_pos = 0;
    uint32_t num_test = 0;
    double map = 0;
    uint32_t print_skip = 1 + (tot_num_class / 20); // print about 20 steps in recall
    uint32_t next_print = 1;
    for( vect_scored_det_t::const_iterator i = name_scored_dets.begin(); i != name_scored_dets.end(); )
    {
      uint32_t const orig_num_pos = num_pos;
      double const cur_score = i->score;
      while( 1 ) // handle all dets with equal score as a unit
      {
	match_res_t const match_res = try_match( class_name, *i );
	if( match_res.is_pos && !match_res.is_diff ) { 
	  ++num_test; ++num_pos; // positive, recall increased (precision must == 1 or also be increased)
	} else if( !match_res.is_pos ) { ++num_test; } // negative, precision decreased
	++i; if( (i == name_scored_dets.end()) || (cur_score != i->score) ) { break; }
      }
      if( orig_num_pos != num_pos ) // recall increased
      {
	prc_elem_t const prc_elem( num_pos, num_test, cur_score );
	map += prc_elem.get_precision() * (num_pos - orig_num_pos);
	if( num_pos >= next_print ) { 
	  next_print = num_pos + print_skip; 
	  print_prc_line( prc_elem, tot_num_class, map / tot_num_class ); 
	}
	prc_elems.push_back( prc_elem );
      }
    }
    map /= tot_num_class;
    if( next_print != (num_pos+print_skip) ) { print_prc_line( prc_elems.back(), tot_num_class, map ); }
    printf( "--- tot_num=%s num_pos=%s num_test=%s num_neg=%s final_map=%s\n", 
	    str(tot_num_class).c_str(), 
	    str(num_pos).c_str(), str(num_test).c_str(), 
	    str(num_test - num_pos).c_str(),
	    str(map).c_str() );
    prc_plot( class_name, tot_num_class, prc_elems );

  }

  void img_db_t::score_results( void )
  {
    for( name_vect_scored_det_map_t::iterator i = scored_dets.begin(); i != scored_dets.end(); ++i )
    {
      score_results_for_class( i->first, i->second );
    }
  }


}
