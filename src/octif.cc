#include"boda_tu_base.H"
#include<iostream>
#include<octave/oct.h>
#include<octave/octave.h>
#include<octave/parse.h>
#include<octave/oct-map.h>
//#include<octave/mxarray.h>
//#include<octave/mexproto.h>
#include<fstream>
#include<boost/filesystem.hpp>

#include"model.h"
#include"geom_prim.H"
#include"str_util.H"
#include"img_io.H"
#include"results_io.H"
#include"octif.H"
#include"timers.H"

namespace boda 
{
  using namespace::std;

  void print_field( ostream & out, octave_scalar_map const & osm, string const & fn )
  {
    octave_value fv = osm.contents(fn.c_str());
    assert_st( !error_state && fv.is_defined() );
    out << fn << "="; fv.print(out);
  }

  void oct_init( void )
  {
    string_vector argv (2);
    argv(0) = "embedded";
    argv(1) = "-q";
    octave_main (2, argv.c_str_vec (), 1);
  }
  using boost::filesystem3::path;
  void test_oct( ostream & out, string const & mat_fn )
  {
    octave_value_list in;
    in(0) = octave_value( mat_fn );
    octave_value_list load_out = feval ("load", in, 1);
    assert_st( !error_state && (load_out.length() > 0) );
    //out << "load of ["  << in(0).string_value () << "] is "; out(0).print(out); out << endl;
    //out << "load ret="; out(0).print(out); out << endl;
    octave_scalar_map osm = load_out(0).scalar_map_value();
    assert_st( !error_state );
    octave_value mod = osm.contents("csc_model");
    assert_st( !error_state && mod.is_defined() );
    octave_scalar_map mod_osm = mod.scalar_map_value();
    assert_st( !error_state );
    out << "keys=";
    for (int i = 0; i < mod_osm.keys().length(); ++i)
    {
      out << mod_osm.keys()[i] << " ";
    }
    out << endl;
    print_field( out, mod_osm, "thresh" );
    print_field( out, mod_osm, "sbin" );
    print_field( out, mod_osm, "interval" );
#if 1
    mxArray mxa( mod );
    Model mod2( &mxa );
#endif
  }

  struct test_oct_t : virtual public nesi, public has_main_t // NESI(help="run simple octave interface test",bases=["has_main_t"], type_id="test_oct", hide=1 )
  {
    filename_t mat_fn; //NESI(default="%(boda_test_dir)/oct_test/car_final_cascade.mat",help="in matrix fn")
    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="output filename")
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    virtual void main( nesi_init_arg_t * nia ) { 
      p_ofstream out = ofs_open( out_fn.exp );  
      test_oct( *out, mat_fn.exp ); 
    }
  };

  void bs_matrix_to_dets( Matrix & det_boxes, uint32_t const img_ix, p_vect_scored_det_t scored_dets )
  {
    for( octave_idx_type i = 0; i < det_boxes.dim1(); ++i ) {
      assert( det_boxes.dim2() >= 5 ); // should have at least 4 columns to form detection bbox + 1 for score
      scored_det_t det;
      det.img_ix = img_ix;
      for( uint32_t p = 0; p < 2; ++p ) { // upper/lower
	for( uint32_t d = 0; d < 2; ++d ) { // x/y
	  det.p[p].d[d] = det_boxes(i,(p*2)+d);
	}
      }
      det.score = det_boxes(i,det_boxes.dim2()-1);
      det.from_pascal_coord_adjust();
      //printf( "det=%s\n", str(det).c_str() );
      if(scored_dets) { scored_dets->push_back( det ); }
    }
  }

  void oct_dfc( ostream & out, string const & dpm_fast_cascade_dir, p_vect_scored_det_t scored_dets, 
		string const & class_name, string const & image_fn, uint32_t const img_ix ) {
    boost::filesystem::path init_path = boost::filesystem::current_path();
    printf( "oct_dfc() class_name=%s image_fn=%s img_ix=%s\n", 
	    str(class_name).c_str(), str(image_fn).c_str(), str(img_ix).c_str() );
    p_img_t img( new img_t );
    img->load_fn( image_fn.c_str() );

    int parse_ret = 0;
    path dfc_vr = path(dpm_fast_cascade_dir) / "voc-release5";
    eval_string("cd "+dfc_vr.string(), 0, parse_ret);
    assert_st( !error_state );
    eval_string("pkg load image", 0, parse_ret);
    assert_st( !error_state );
    feval("startup" );
    assert_st( !error_state );

    string const mat_fn = (dfc_vr / (class_name+"_final.mat")).string();
    octave_value_list in;
    in(0) = octave_value( mat_fn );
    octave_value_list load_out = feval ("load", in, 1);
    assert_st( !error_state && (load_out.length() > 0) );
    //out << "load of ["  << in(0).string_value () << "] is "; load_out(0).print(out); out << endl;
    //out << "load ret="; load_out(0).print(out); out << endl;
    octave_scalar_map osm = load_out(0).scalar_map_value();
    assert_st( !error_state );
    octave_value mod = osm.contents("model");
    assert_st( !error_state && mod.is_defined() );
    octave_scalar_map mod_osm = mod.scalar_map_value();
    assert_st( !error_state );
    out << "keys=";
    for (int i = 0; i < mod_osm.keys().length(); ++i)
    {
      out << mod_osm.keys()[i] << " ";
    }
    out << endl;
    print_field( out, mod_osm, "thresh" );
    print_field( out, mod_osm, "sbin" );
    print_field( out, mod_osm, "interval" );
    assert_st( !error_state );
   
    in(0) = octave_value( image_fn );
    in(1) = mod;
    octave_value_list boda_if_ret;
    {
      timer_t t( "dfc_octave_top" );
      boda_if_ret = feval("boda_if", in, 2 );
    }
    assert_st( !error_state );
    assert_st( boda_if_ret.length() == 1);
    assert_st( boda_if_ret(0).is_matrix_type() );
    // unclear if this will always work and/or how to error check, since there are many 'matrix' types ...
    Matrix det_boxes = boda_if_ret(0).matrix_value(); 
    assert_st( !error_state );
    {
      timer_t t( "bs_matrix_to_dets" );
      bs_matrix_to_dets( det_boxes, img_ix, scored_dets );
    }
    boost::filesystem::current_path( init_path );
  }

  struct oct_dfc_t : virtual public nesi, public has_main_t // NESI(help="run dpm fast cascade over a single image file",bases=["has_main_t"], type_id="oct_dfc")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string image_fn; //NESI(help="input: image filename",req=1)
    string class_name; //NESI(help="name of object class",req=1)
    uint32_t img_ix; //NESI(default=0,help="internal use only: img_ix to put in results placed in results vector")
    string dpm_fast_cascade_dir; // NESI(help="dpm_fast_cascade base src dir, usually /parent/dirs/svn_work/dpm_fast_cascade",req=1)
    p_vect_scored_det_t scored_dets; // output, may be null to omit

    virtual void main( nesi_init_arg_t * nia ) {
      oct_dfc( cout, dpm_fast_cascade_dir, scored_dets, class_name, image_fn, img_ix );
    }
  };

  // example command lines: FIXME: fold iteration / filename generation into this class?
  // for cn in `cat ../../test/pascal_classes.txt`; do ../../lib/boda score --pil-fn=%(pascal_data_dir)/ImageSets/Main/${cn}_test.txt --res-fn=${cn}_hamming.txt --class-name=${cn}; done
  // for cn in `cat ../test/pascal_classes.txt`; do ../lib/boda mat_bs_to_pascal --mat-bs-fn=/home/moskewcz/bench/hamming/hamming_toplevel_bboxes_pascal2007/${cn}_boxes_test__hamming.mat --res-fn=${cn}_hamming.txt --class-name=${cn} --pil-fn=%(pascal_data_dir)/ImageSets/Main/${cn}_test.txt ; done

  struct convert_matlab_res_t : virtual public nesi, public has_main_t // NESI(help="convert matlab 'ds' results to pascal format",bases=["has_main_t"], type_id="mat_bs_to_pascal")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string mat_bs_fn; //NESI(help="input: name of matlab file with ds var with results",req=1)
    string class_name; //NESI(help="name of object class",req=1)
    string pil_fn; //NESI(help="input: name of pascal-VOC format image list file",req=1)
    string res_fn; //NESI(help="output: name of pascal-VOC format detection results file to write",req=1)
    p_img_db_t img_db; //NESI(default="()", help="image database")

    virtual void main( nesi_init_arg_t * nia ) {
      
      read_pascal_image_list_file( img_db, pil_fn, 0 );

      p_vect_scored_det_t scored_dets( new vect_scored_det_t );
      octave_value_list in;
      in(0) = octave_value( mat_bs_fn );
      octave_value_list load_out = feval ("load", in, 1);
      assert_st( !error_state && (load_out.length() > 0) );
      octave_scalar_map osm = load_out(0).scalar_map_value();
      assert_st( !error_state );
      Cell ds = osm.contents("ds").cell_value();
      assert_st( !error_state );

      //printf( "ds.dim2()=%s ds.columns()=%s\n", str(ds.dim2()).c_str(), str(ds.columns()).c_str() );
      for( octave_idx_type i = 0; i < ds.dim2(); ++i ) {
	octave_value det_boxes_val = ds(i);
	assert_st( !error_state && det_boxes_val.is_defined() );
	if( det_boxes_val.is_matrix_type() ) {
	  Matrix det_boxes = det_boxes_val.matrix_value();
	  bs_matrix_to_dets( det_boxes, i, scored_dets );
	}
	//img_db_show_dets( img_db, scored_dets, i );
	//scored_dets->clear();
	//break;
      }
      img_db_set_results( img_db, class_name, scored_dets );

      write_results_file( img_db, res_fn, class_name );
    }
  };


#include"gen/octif.cc.nesi_gen.cc"

}
