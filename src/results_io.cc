#include"boda_tu_base.H"
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
  };
  typedef vector< scored_det_t > vect_scored_det_t;
  typedef map< string, vect_scored_det_t > name_vect_scored_det_map_t;
  typedef shared_ptr< vect_scored_det_t > p_vect_scored_det_t;

  struct gt_det_t : public u32_box_t
  {
    uint32_t truncated;
    uint32_t difficult;    
  };
  typedef vector< gt_det_t > vect_gt_det_t;
  typedef map< string, vect_gt_det_t > name_vect_dt_det_map_t;

  struct img_info_t
  {
    u32_pt_t size;
    uint32_t depth;
    name_vect_dt_det_map_t gt_dets;
    name_vect_scored_det_map_t scored_dets;
  };
  typedef shared_ptr< img_info_t > p_img_info_t;
  typedef map< string, p_img_info_t > id_to_img_info_map_t;

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
    p_img_info_t img_info( new img_info_t );
    
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
      gt_det.one_to_zero_coord_adj();
      assert_st( gt_det.is_strictly_normalized() );
      string const ann_obj_name( xml_must_decend( ann_fn, ann_obj, "name" ).child_value() );
      img_info->gt_dets[ann_obj_name].push_back(gt_det);
    }
    return img_info;
  }

  struct img_db_t 
  {
    path pascal_base_path;
    id_to_img_info_map_t id_to_img_info_map;
    img_db_t( void ) : pascal_base_path( "/home/moskewcz/bench/VOCdevkit/VOC2007" ) { }
    void load_ann_for_id( string const & img_id )
    {
      p_img_info_t & img_info = id_to_img_info_map[img_id];
      if( img_info ) { rt_err( "tried to load annotations multiple times for id '"+img_id+"'"); }
      img_info = read_pascal_annotations_for_id( pascal_base_path, img_id ); 
    }
    void add_scored_det( string const & class_name, string const & img_id, scored_det_t const & scored_det )
    {
      id_to_img_info_map_t::iterator img_info = id_to_img_info_map.find(img_id);
      if( img_info == id_to_img_info_map.end() ) { rt_err("tried to add detection for unloaded id '"+img_id+"'"); }
      img_info->second->scored_dets[class_name].push_back( scored_det );
    }
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
      string const img_id = parts[0];
      scored_det_t scored_det;
      scored_det.score = lc_str_d( parts[1] );
      scored_det.p[0].d[0] = lc_str_u32( parts[2] );
      scored_det.p[0].d[1] = lc_str_u32( parts[3] );
      scored_det.p[1].d[0] = lc_str_u32( parts[4] );
      scored_det.p[1].d[1] = lc_str_u32( parts[5] );
      scored_det.one_to_zero_coord_adj();
      assert_st( scored_det.is_strictly_normalized() );
      img_db->add_scored_det( class_name, img_id, scored_det );
    }
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

}
