#include"boda_tu_base.H"
#include<iostream>
#include<octave/oct.h>
#include<octave/octave.h>
#include<octave/parse.h>
#include<octave/oct-map.h>
#include<octave/Cell.h>
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


  typedef vector< NDArray > vect_NDAarray;

  vect_string boda_octave_setup = { "warning ('off', 'Octave:undefined-return-values');",
				    "warning ('off', 'all');",
				    "pkg load image;",
				    };

  void oct_dfc_startup( string const & dpm_fast_cascade_dir ) {
    timer_t t( "oct_dfc_startup" );
    assert_st( !error_state );
    string const mwm_vocr5_dir = strprintf( "%s/voc-release5", dpm_fast_cascade_dir.c_str() );
    set_global_value( "mwm_vocr5_dir", octave_value( mwm_vocr5_dir ) );
    int parse_ret = 0;
    for( vect_string::const_iterator i = boda_octave_setup.begin(); i != boda_octave_setup.end(); ++i ) {
      eval_string( *i, 0, parse_ret );
      assert_st( !error_state && !parse_ret );
    }
    eval_string( "addpath( '" + mwm_vocr5_dir + "' );", 0, parse_ret );
    assert_st( !error_state && !parse_ret );
    feval( "startup" );
    assert_st( !error_state );
  }

  Matrix get_field_as_matrix( octave_scalar_map const & osm, string const & fn ) {
    octave_value fv = osm.contents(fn.c_str());
    assert_st( !error_state && fv.is_defined() );    
    assert_st( fv.is_matrix_type() );
    return fv.matrix_value();
  }

  void get_ndas_from_field( vect_NDAarray & out, octave_scalar_map const & osm, string const & fn ) {
    octave_value fv = osm.contents(fn.c_str());
    assert_st( !error_state && fv.is_defined() );
    assert_st( fv.is_cell() );
    Cell fv_cell = fv.cell_value();
    assert( fv_cell.numel() == fv_cell.dims().elem(0) ); // should be N(x1)* 
    for( uint32_t ix = 0; ix != uint32_t(fv_cell.dims().elem(0)); ++ix ) { 
      octave_value fv_cell_ix = fv_cell(ix);
      assert_st( fv_cell_ix.is_matrix_type() );
      out.push_back( fv_cell_ix.array_value() );
    }
    

  }

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
      scored_dets->push_back( det );
    }
  }

  void oct_dfc( ostream & out, string const & dpm_fast_cascade_dir, p_vect_scored_det_t scored_dets, 
		string const & image_fn, uint32_t const img_ix ) {

    oct_dfc_startup( dpm_fast_cascade_dir );

    string const & class_name = scored_dets->class_name;
    printf( "oct_dfc() class_name=%s image_fn=%s img_ix=%s\n", 
	    str(class_name).c_str(), str(image_fn).c_str(), str(img_ix).c_str() );
    p_img_t img( new img_t );
    img->load_fn( image_fn.c_str() );

    string const mat_fn = (path(dpm_fast_cascade_dir) / "voc-release5" / "VOC2007" / (class_name+"_final.mat")).string();
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
  }

  struct oct_dfc_t : virtual public nesi, public has_main_t // NESI(help="run dpm fast cascade over a single image file",bases=["has_main_t"], type_id="oct_dfc")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string image_fn; //NESI(help="input: image filename",req=1)
    string class_name; //NESI(help="name of object class",req=1)
    uint32_t img_ix; //NESI(default=0,help="internal use only: img_ix to put in results placed in results vector")
    string dpm_fast_cascade_dir; // NESI(help="dpm_fast_cascade base src dir, usually /parent/dirs/svn_work/dpm_fast_cascade",req=1)
    p_vect_scored_det_t scored_dets; // output

    virtual void main( nesi_init_arg_t * nia ) {
      scored_dets.reset( new vect_scored_det_t( class_name ) );
      oct_dfc( cout, dpm_fast_cascade_dir, scored_dets, image_fn, img_ix );
    }
  };

  // FIXME: example cmds out of date, iter moved inside:
  // for cn in `cat ../../test/pascal_classes.txt`; do ../../lib/boda score --pil-fn=%(pascal_data_dir)/ImageSets/Main/${cn}_test.txt --res-fn=${cn}_hamming.txt --class-name=${cn}; done
  // for cn in `cat ../test/pascal_classes.txt`; do ../lib/boda mat_bs_to_pascal --mat-bs-fn=/home/moskewcz/bench/hamming/hamming_toplevel_bboxes_pascal2007/${cn}_boxes_test__hamming.mat --res-fn=${cn}_hamming.txt --class-name=${cn} --pil-fn=%(pascal_data_dir)/ImageSets/Main/${cn}_test.txt ; done

  struct convert_matlab_res_t : virtual public nesi, public has_main_t // NESI(help="convert matlab 'ds' results to pascal format",bases=["has_main_t"], type_id="mat_bs_to_pascal")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t pascal_classes_fn; //NESI(default="%(boda_test_dir)/pascal_classes.txt",help="file with list of classes to process")
    filename_t mat_bs_fn; //NESI(default="%(bench_dir)/hamming/voc-release5_simpleHog_no_sqrt_11-27-13/2007/%%s_boxes_test_simpleHog.mat", help="input: format for filenames of matlab file with ds var with results: %%s will be replaced with the class name")
    filename_t pil_fn; //NESI(default="%(pascal_data_dir)/ImageSets/Main/%%s_test.txt",help="format for filenames of pascal image list files. %%s will be replaced with the class name.")
    filename_t res_fn; //NESI(default="%(bench_dir)/hamming/pf_shog/%%s_test.txt",help="output: format for filenames of pascal-VOC format detection results file to write")
    p_img_db_t img_db; //NESI(default="()", help="image database")

    virtual void main( nesi_init_arg_t * nia ) {
      p_vect_string classes = readlines_fn( pascal_classes_fn.exp );
      for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
	convert_class( *i, i != (*classes).begin() );
      }
    }

    void convert_class( string const & class_name, bool const check_ix_only ) {
      string c_pil_fn = strprintf( pil_fn.exp.c_str(), class_name.c_str() );
      read_pascal_image_list_file( img_db, c_pil_fn, 0, check_ix_only );
      p_vect_scored_det_t scored_dets( new vect_scored_det_t( class_name ) );
      octave_value_list in;
      
      string c_mat_bs_fn = strprintf( mat_bs_fn.exp.c_str(), class_name.c_str() );
      in(0) = octave_value( c_mat_bs_fn );
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
      string c_res_fn = strprintf( res_fn.exp.c_str(), class_name.c_str() );
      write_results_file( img_db, c_res_fn, scored_dets );
    }
  };

  void osm_print_keys( ostream & out, octave_value & ov ) {
    octave_scalar_map osm = ov.scalar_map_value();
    assert_st( !error_state );
    out << "keys=";
    for (int i = 0; i < osm.keys().length(); ++i)
    {
      out << osm.keys()[i] << " ";
    }
    out << endl;
  }

  void oct_featpyra_inner( p_img_t img, uint32_t const sbin, uint32_t const interval ) {
    timer_t t( "oct_featpyra_inner" );
    assert_st( sbin >= 2 ); // for sanity
    uint32_t min_pyra_isz = 5 * sbin; // 5 seems pretty arbirtary. needs to be integer and at least 1 i think though.
    if( img->min_dim() < min_pyra_isz ) {
      rt_err( strprintf("input image with WxH = %s is too small to make a feature pyramid", img->WxH_str().c_str() ) );
    }
    for( uint32_t i = 0; i < interval; ++i ) { // we create one 'primary' scaled image per internal (octave sub-step)
      p_img_t ids_img = img;
      if( i ) { // don't try to scale for 0 step (where scale == 1)
	uint32_t interval_scale = pow(2.0d, 16.0d - (double(i) / double(interval) ) ); // in 0.16 fixed point
	assert_st( interval_scale < (1U<<16) );
	printf( "interval_scale=%s\n", str(interval_scale).c_str() );
	ids_img = img->downsample( uint16_t(interval_scale) ); // note: 0.16 fixed point scale must be in [.5,1)
      }
      for( int32_t octave = 1; octave > -15; --octave )  { // scale = 2^octave, -15 bound is a sanity limit only
	uint32_t min_sz = (octave >= 0 ) ? (ids_img->min_dim() << octave) : (ids_img->min_dim() >> (-octave));
	if( min_sz < min_pyra_isz ) { break; } // this octave/interval is too small.
	double const scale = pow(2.0d, double(octave) - (double(i) / double(interval) ) );
	printf( "scale=%s\n", str(scale).c_str() );
	// we now handle all the factor-of-2 up and down samplings of the image in this loop
	if( octave >= 0 ) { // we handle octaves >= 0 (normal and upsampled octaves) by adjusting the feature sbin
	  uint32_t const feat_sbin = sbin >> octave;
	  assert_st( feat_sbin >= 2 ); // 1 might actually be okay for HOG, unclear, but it would be ... odd.
	  assert_st( (feat_sbin << octave) == sbin ); // sbin should be evenly disisible
	} else { // we handle the downsampled octaves by actually 2X downsampling the image and using the input sbin
	  ids_img = ids_img->downsample( 1 << 15 ); // note: input ids_img is released here	  
	}
      }
    }
  }

  p_nda_double_t create_p_nda_double_from_oct_NDArray( NDArray const & nda ) {
    dim_vector const & dv = nda.dims();
    dims_t dims; // boda nda stores dims in row-major order, so we reverse the octave dims as we copy them
    dims.resize_and_zero( dv.length() );
    for( uint32_t i = 0; i < uint32_t(dv.length()); ++i ) { dims.dims(i) = dv.elem(dv.length()-1-i); }
    p_nda_double_t ret( new nda_double_t );
    ret->set_dims( dims );
    assert_st( ret->elems.sz == (uint32_t)nda.numel() );
    double const * oct_data = nda.fortran_vec();
    double * data = &ret->elems[0]; // our data layout is now an exact match to the octave one ...
    for( uint32_t i = 0; i < ret->elems.sz; ++i ) { data[i] = oct_data[i]; } // ... so, just copy the elements flat
    return ret;
  }
  
  void write_scales_and_feats( p_ostream out, p_nda_double_t & scales, vect_p_nda_double_t & feats ) {
#if 0
    scales.reset( new nda_double_t );
    dims_t dims; 
    dims.resize_and_zero( 2 );
    dims.dims(1) = 20;
    dims.dims(0) = 1;
    scales->set_dims( dims );
    for( uint32_t i = 0; i < scales->elems.sz; ++i ) { scales->elems[i]  = 1.1; }
#endif
    bwrite( *out, boda_magic );
    bwrite_id( *out, string("scales") );
    bwrite( *out, string("p_nda_double_t") );
    bwrite( *out, scales );
    bwrite_id( *out, string("feats") );
    bwrite( *out, string("vect_p_nda_double_t") );
    bwrite( *out, feats );
    bwrite( *out, string("END_BODA") );
    
  }

  void oct_featpyra( ostream & out, string const & dpm_fast_cascade_dir, 
		     string const & image_fn, string const & pyra_out_fn ) {
    timer_t t( "oct_featpyra" );
    //out << strprintf( "oct_featpyra() image_fn=%s\n", str(image_fn).c_str() );

    p_img_t img( new img_t );
    img->load_fn( image_fn.c_str() );
    //oct_featpyra_inner( img, 8, 10 );
    //return;

    oct_dfc_startup( dpm_fast_cascade_dir );

    dim_vector img_dv(img->h, img->w, 3);
    NDArray img_oct( img_dv );
    for( uint32_t y = 0; y < img->h; ++y ) {
      for( uint32_t x = 0; x < img->w; ++x ) {
	for( uint32_t c = 0; c < 3; ++c ) {
	  img_oct(y,x,c) = img->pels.get()[y*img->row_pitch + x*img->depth + c];
	}
      }
    }

    octave_value_list in;
    in(0) = img_oct; //octave_value( image_fn );
    in(1) = octave_value( pyra_out_fn );
    octave_value_list boda_if_ret;
    {
      timer_t t( "oct_featpyra_boda_if_feat" );
      boda_if_ret = feval("boda_if_feat", in, 1 );
    }
    assert_st( !error_state );
    assert_st( boda_if_ret.length() == 1);
    //osm_print_keys( out, boda_if_ret(0) );
    assert_st( !error_state );

    p_nda_double_t scales;
    vect_p_nda_double_t feats;

    octave_scalar_map ret_osm = boda_if_ret(0).scalar_map_value();
    //print_field( out, ret_osm, "scales" );
    //print_field( out, ret_osm, "feat" );
    Matrix oct_scales = get_field_as_matrix( ret_osm, "scales" );
    assert( oct_scales.numel() == oct_scales.dims().elem(0) ); // should be N(x1)* 
    scales = create_p_nda_double_from_oct_NDArray( oct_scales );
    //printf( "scales=%s\n", str(scales).c_str() );
    vect_NDAarray oct_feats;
    get_ndas_from_field( oct_feats, ret_osm, "feat" );
    //printf( "scales.size()=%s\n", str(scales.size()).c_str() );
    //printf( "oct_feats.size()=%s\n", str(oct_feats.size()).c_str() );
    assert_st( scales->elems.sz == oct_feats.size() );
    for( vect_NDAarray::const_iterator i = oct_feats.begin(); i != oct_feats.end(); ++i ) {
      feats.push_back( create_p_nda_double_from_oct_NDArray( *i ) );
    }
    p_ostream bo = ofs_open( pyra_out_fn + ".boda" );
    write_scales_and_feats( bo, scales, feats );

#if 0
    for( uint32_t i = 0; i < feats.size(); ++i ) {
      printf( "scale=%s\n", str(scales->elems[i]).c_str() );
      printf( "feats=%s\n", str(feats[i]).c_str() );
    }
#endif

  }


  

#if 0

pyra.num_levels = length(pyra.feat);

td = model.features.truncation_dim;
for i = 1:pyra.num_levels
  % add 1 to padding because feature generation deletes a 1-cell
  % wide border around the feature map
  pyra.feat{i} = padarray(pyra.feat{i}, [pady+1 padx+1 0], 0);
  % write boundary occlusion feature
  pyra.feat{i}(1:pady+1, :, td) = 1;
  pyra.feat{i}(end-pady:end, :, td) = 1;
  pyra.feat{i}(:, 1:padx+1, td) = 1;
  pyra.feat{i}(:, end-padx:end, td) = 1;
end
pyra.valid_levels = true(pyra.num_levels, 1);
pyra.padx = padx;
pyra.pady = pady;
#endif


  struct oct_featpyra_t : virtual public nesi, public has_main_t // NESI(help="run dpm fast cascade over a single image file",bases=["has_main_t"], type_id="oct_featpyra")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t image_fn; //NESI(help="input: image filename",req=1)
    filename_t pyra_out_fn; //NESI(default="%(boda_output_dir)/pyra.oct",help="output: octave text output filename for feature pyramid")
    string dpm_fast_cascade_dir; // NESI(help="dpm_fast_cascade base src dir, usually /parent/dirs/svn_work/dpm_fast_cascade",req=1)
    virtual void main( nesi_init_arg_t * nia ) {
      oct_featpyra( cout, dpm_fast_cascade_dir, image_fn.exp, pyra_out_fn.exp );
    }
  };


#include"gen/octif.cc.nesi_gen.cc"

}
