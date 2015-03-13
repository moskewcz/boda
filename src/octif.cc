// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include<iostream>
#include<octave/oct.h>
#include<octave/octave.h>
#include<octave/toplev.h> // for clean_up_and_exit()
#include<octave/parse.h>
#include<octave/oct-map.h>
#include<octave/Cell.h>
#include<octave/mxarray.h>
//#include<octave/mexproto.h>
#include<fstream>
#include<boost/filesystem.hpp>

#include"ext/model.h"
#include"geom_prim.H"
#include"str_util.H"
#include"img_io.H"
#include"results_io.H"
#include"octif.H"
#include"timers.H"
#include"nesi.H"


namespace boda 
{
  using std::ostream;
  using std::endl;
  using std::cout;

  p_nda_double_t pad_nda( double const v, vect_uint32_t const & pad_n, vect_uint32_t const & pad_p, p_nda_double_t in ) {
    assert( pad_n.size() == in->dims.sz() );
    assert( pad_p.size() == in->dims.sz() );
    dims_t ret_dims( in->dims.sz() );
    for( uint32_t i = 0; i < ret_dims.sz(); ++i ) { ret_dims.dims(i) = in->dims.dims(i) + pad_n[i] + pad_p[i]; }
    p_nda_double_t ret( new nda_double_t( ret_dims ) );    
    for( uint32_t i = 0; i < ret->elems.sz; ++i ) { ret->elems[i] = v; } //  init
    for( dims_iter_t di( in->dims ) ; ; ) { ret->at(di.di,pad_n) = in->at(di.di);  if( !di.next() ) { break; } }
    return ret;
  }
  p_nda_double_t pad_nda( double const v, vect_uint32_t const & pad, p_nda_double_t in ) { 
    return pad_nda( v, pad, pad, in );  }


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

  NDArray get_field_as_NDArray( octave_scalar_map const & osm, string const & fn ) {
    octave_value fv = osm.contents(fn.c_str());
    assert_st( !error_state && fv.is_defined() );    
    assert_st( fv.is_matrix_type() );
    return fv.array_value();
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

  void oct_exit( int const retval )
  {
    // without this call, octave chokes (inf loop, double free, ???)
    // at process exit time. so, we seem to have little option other
    // than calling this, which will terminate our process. charming!
    clean_up_and_exit( retval ); 
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

  void bs_matrix_to_dets( NDArray & det_boxes, uint32_t const img_ix, p_per_class_scored_dets_t scored_dets )
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
      scored_dets->add_det( det );
    }
  }

  void oct_dfc( ostream & out, string const & dpm_fast_cascade_dir, p_per_class_scored_dets_t scored_dets, 
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
    NDArray det_boxes = boda_if_ret(0).array_value(); 
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
    p_per_class_scored_dets_t scored_dets; // output

    virtual void main( nesi_init_arg_t * nia ) {
      scored_dets.reset( new per_class_scored_dets_t( class_name ) );
      oct_dfc( cout, dpm_fast_cascade_dir, scored_dets, image_fn, img_ix );
    }
  };

  // FIXME: example cmds out of date, iter moved inside:
  // for cn in `cat ../../test/pascal_classes.txt`; do ../../lib/boda score --pil-fn=%(pascal_data_dir)/ImageSets/Main/${cn}_test.txt --res-fn=${cn}_hamming.txt --class-name=${cn}; done
  // for cn in `cat ../test/pascal_classes.txt`; do ../lib/boda mat_bs_to_pascal --mat-bs-fn=/home/moskewcz/bench/hamming/hamming_toplevel_bboxes_pascal2007/${cn}_boxes_test__hamming.mat --res-fn=${cn}_hamming.txt --class-name=${cn} --pil-fn=%(pascal_data_dir)/ImageSets/Main/${cn}_test.txt ; done

  struct convert_matlab_res_t : virtual public nesi, public load_pil_t // NESI(help="convert matlab 'ds' results to pascal format",bases=["load_pil_t"], type_id="mat_bs_to_pascal")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t mat_bs_fn; //NESI(default="%(bench_dir)/hamming/voc-release5_simpleHog_no_sqrt_11-27-13/2007/%%s_boxes_test_simpleHog.mat", help="input: format for filenames of matlab file with ds var with results: %%s will be replaced with the class name")
    filename_t res_fn; //NESI(default="%(bench_dir)/hamming/pf_shog/%%s_test.txt",help="output: format for filenames of pascal-VOC format detection results file to write")

    virtual void main( nesi_init_arg_t * nia ) {
      load_img_db( false );
      for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) { convert_class( *i ); }
    }

    void convert_class( string const & class_name ) {
      p_per_class_scored_dets_t scored_dets( new per_class_scored_dets_t( class_name ) );
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
	  NDArray det_boxes = det_boxes_val.array_value();
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

  p_nda_double_t process( p_nda_double_t const & mximage, int const sbin );

  p_nda_double_t create_p_nda_double_from_img( p_img_t img ) {
    dims_t dims(3);
    dims.dims(0) = 3;
    dims.dims(1) = img->sz.d[0];
    dims.dims(2) = img->sz.d[1];
    p_nda_double_t ret( new nda_double_t( dims ) );
    for( uint32_t y = 0; y < img->sz.d[1]; ++y ) {
      for( uint32_t x = 0; x < img->sz.d[0]; ++x ) {
	for( uint32_t c = 0; c < 3; ++c ) {
	  ret->at3(c,x,y) = img->pels.get()[y*img->row_pitch + x*img->depth + c];
	}
      }
    }
    return ret;
  }


  p_nda_double_t create_p_nda_double_from_oct_NDArray( NDArray const & nda ) {
    dim_vector const & dv = nda.dims();
    dims_t dims( dv.length() ); // boda nda stores dims in row-major order, so we reverse the octave dims as we copy them
    for( uint32_t i = 0; i < uint32_t(dv.length()); ++i ) { dims.dims(i) = dv.elem(dv.length()-1-i); }
    p_nda_double_t ret( new nda_double_t( dims ) );
    assert_st( ret->elems.sz == (uint32_t)nda.numel() );
    double const * oct_data = nda.fortran_vec();
    double * data = &ret->elems[0]; // our data layout is now an exact match to the octave one ...
    for( uint32_t i = 0; i < ret->elems.sz; ++i ) { data[i] = oct_data[i]; } // ... so, just copy the elements flat
    return ret;
  }

  // assumes nda's dims are channels, w, h
  p_img_t create_p_img_from_p_nda_double_from_img( p_nda_double_t nda ) {
    dims_t const & dims = nda->dims;
    assert( dims.dims(0) == 3 ); // RGB
    p_img_t img( new img_t );
    img->set_sz_and_alloc_pels( {dims.dims(1), dims.dims(2)} );
    for( uint32_t y = 0; y < img->sz.d[1]; ++y ) {
      for( uint32_t x = 0; x < img->sz.d[0]; ++x ) {
	for( uint32_t c = 0; c < 3; ++c ) {
	  img->pels.get()[y*img->row_pitch + x*img->depth + c] = nda->at3(c,x,y);
	}
	img->pels.get()[y*img->row_pitch + x*img->depth + 3] = uint8_t_const_max; // alpha
      }
    }
    return img;
  }

  
  void write_scales_and_feats( p_ostream out, p_nda_double_t & scales, vect_p_nda_double_t & feats ) {
#if 0
    dims_t dims( 2 );
    dims.dims(1) = 20;
    dims.dims(0) = 1;
    scales.reset( new nda_double_t( dims ) );
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

  p_nda_double_t oct_img_resize( NDArray const & img_oct, double const scale )
  {
    octave_value_list in;
    in(0) = img_oct; //octave_value( image_fn );
    in(1) = octave_value( scale );
    octave_value_list boda_if_ret;
    {
      timer_t t( "oct_resize" );
      boda_if_ret = feval("resize", in, 1 );
    }
    assert_st( !error_state );
    assert_st( boda_if_ret.length() == 1);
    //osm_print_keys( out, boda_if_ret(0) );
    assert_st( !error_state );

    assert_st( boda_if_ret(0).is_matrix_type() );
    NDArray oct_resize_out = boda_if_ret(0).array_value();
    p_nda_double_t resize_out = create_p_nda_double_from_oct_NDArray( oct_resize_out );
    return resize_out;
  }
  p_nda_double_t oct_img_resize( p_img_t img, double const scale )
  {
    //return;
    dim_vector img_dv(img->sz.d[1], img->sz.d[0], 3);
    NDArray img_oct( img_dv );
    for( uint32_t y = 0; y < img->sz.d[1]; ++y ) {
      for( uint32_t x = 0; x < img->sz.d[0]; ++x ) {
	for( uint32_t c = 0; c < 3; ++c ) {
	  img_oct(y,x,c) = img->pels.get()[y*img->row_pitch + x*img->depth + c];
	}
      }
    }
    return oct_img_resize( img_oct, scale );
  }
  p_nda_double_t oct_img_resize( p_nda_double_t img, double const scale )
  {
    assert( img->dims.sz() == 3 );
    assert( img->dims.dims(0) == 3 );
    //return;
    dim_vector img_dv(img->dims.dims(2), img->dims.dims(1), 3);
    NDArray img_oct( img_dv );
    for( uint32_t y = 0; y < img->dims.dims(2); ++y ) {
      for( uint32_t x = 0; x < img->dims.dims(1); ++x ) {
	for( uint32_t c = 0; c < 3; ++c ) {
	  img_oct(y,x,c) = img->cm_at3(y,x,c);
	}
      }
    }
    return oct_img_resize( img_oct, scale );
  }

  p_nda_double_t clone_from_corner( dims_t const & dims, p_nda_double_t in ) {
    p_nda_double_t ret( new nda_double_t( dims ) );
    //printf( "dims=%s in->dims=%s\n", str(dims).c_str(), str(in->dims).c_str() );
    for( dims_iter_t di( dims ) ; ; ) { ret->at(di.di) = in->at(di.di);  if( !di.next() ) { break; } }
    return ret;
  }

  p_nda_double_t oct_featpyra_inner( vect_p_nda_double_t & feats,
				     p_img_t img, uint32_t const sbin, uint32_t const interval ) {
    vect_double scales;
    //p_nda_double_t ids_nda; // NOTE: uncomment various ids_nda lines to use float/octave resize
    vect_uint32_t pad = { 0, 1, 1 };
    timer_t t( "oct_featpyra_inner" );
    assert_st( sbin >= 2 ); // for sanity
    uint32_t min_pyra_isz = 5 * sbin; // 5 seems pretty arbirtary. needs to be integer and at least 1 i think though.
    for( uint32_t i = 0; i < interval; ++i ) { // we create one 'primary' scaled image per internal (octave sub-step)
      p_img_t ids_img = img;
      if( i ) { // don't try to scale for 0 step (where scale == 1)
	double interval_scale = pow(2.0d, 0.0d - (double(i) / double(interval) ) ); // in 0.16 fixed point
	//printf( "interval_scale=%s\n", str(interval_scale).c_str() );
	//ids_nda = oct_img_resize( img, interval_scale ); // NOTE: also must (1 || i) above cond 
	ids_img = downsample_up_to_2x( img, interval_scale ); // note: scale must be in [.5,1)
      }
      uint32_t cur_scale = i;
      for( int32_t octave = 1; octave > -15; --octave )  { // scale = 2^octave, -15 bound is a sanity limit only
	//uint32_t const ids_min_dim = min(ids_nda->dims.dims(1),ids_nda->dims.dims(2));
	uint32_t const ids_min_dim = ids_img->sz.dims_min();
	uint32_t min_sz = (octave >= 0 ) ? (ids_min_dim << octave) : (ids_min_dim >> 1);
	if( min_sz < min_pyra_isz ) { break; } // this octave/interval is too small.
	double const scale = pow(2.0d, double(octave) - (double(i) / double(interval) ) );
	//printf( "scale=%s\n", str(scale).c_str() );
	uint32_t feat_sbin = sbin;
	// we now handle all the factor-of-2 up and down samplings of the image in this loop
	if( octave >= 0 ) { // we handle octaves >= 0 (normal and upsampled octaves) by adjusting the feature sbin
	  feat_sbin = sbin >> octave;
	  assert_st( feat_sbin >= 2 ); // 1 might actually be okay for HOG, unclear, but it would be ... odd.
	  assert_st( (feat_sbin << octave) == sbin ); // sbin should be evenly disisible
	} else { // we handle the downsampled octaves by actually 2X downsampling the image and using the input sbin
	  //ids_nda = oct_img_resize( ids_nda, .5 );
	  ids_img = downsample_2x( ids_img ); // note: input ids_img is released here	  
	}
	//p_nda_double_t d_im = ids_nda;
	p_nda_double_t d_im = create_p_nda_double_from_img( ids_img );
	//printf( "scale=%s d_im->dims=%s\n", str(scale).c_str(), str(d_im->dims).c_str() );
	p_nda_double_t scale_feats = process( d_im, feat_sbin );
	if( cur_scale >= scales.size() ) { 
	  scales.resize( cur_scale + 1 ); 
	  feats.resize( cur_scale + 1 );
	}
	scale_feats = pad_nda( 0.0d, pad, scale_feats );
	// fill in padding indicator feature
	dims_iter_t di( scale_feats->dims );
	assert( di.di.size() == 3 );
	assert( pad.size() == 3 );
	assert( di.e.front() );
	di.b.front() = di.e.front()-1; // only iterate over last elem of first dim
	for( di.di = di.b ; ; ) { 
	  bool in_pad = 0;
	  for( uint32_t d = 1; d < 3; ++d ) {  
	    if( di.di[d] < pad[d] ) { in_pad = 1; } 
	    if( (di.e[d] - di.di[d] - 1) < pad[d] ) { in_pad = 1; } 
	  }
	  if( in_pad ) { scale_feats->at(di.di) = 1; }
	  if( !di.next() ) { break; } 
	}
	
	scales[cur_scale] = scale;
	feats[cur_scale] = scale_feats;
	cur_scale += interval;
      }
    }
    if( scales.empty() ) {
      rt_err( strprintf("input image with WxH = %s is too small to make a feature pyramid", img->WxH_str().c_str() ) );
    }

    // convert scales to a p_nda_double_t
    dims_t scales_dims( 2 );
    scales_dims.dims(0) = 1;
    scales_dims.dims(1) = scales.size();
    p_nda_double_t scales_out( new nda_double_t( scales_dims ) );
    for( uint32_t i = 0; i < scales.size(); ++i ) { scales_out->cm_at1(i) = scales[i]; }
    return scales_out;
  }

  struct proc_img_file_list_t : virtual public nesi, public has_main_t // NESI(help="process a list of image files to produce an output text file",bases=["has_main_t"], is_abstract=1)
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t image_list_fn; //NESI(help="input: text list of image filenames, one per line",req=1)
    filename_t out_fn; //NESI(default="%(boda_output_dir)/oct_to_boda_comp.txt",help="output: text summary of differences between octave and boda resized images.")
    string dpm_fast_cascade_dir; // NESI(help="dpm_fast_cascade base src dir, usually /parent/dirs/svn_work/dpm_fast_cascade",req=1)

    void main( nesi_init_arg_t * nia ) {
      timer_t t( mode );
      p_ostream out = ofs_open( out_fn.exp );
      p_vect_string image_fns = readlines_fn( image_list_fn );
      for( vect_string::const_iterator i = image_fns->begin(); i != image_fns->end(); ++i ) { 
	(*out) << strprintf( "img_fn=%s\n", str(*i).c_str() );
	proc_img( *out, nesi_filename_t_expand( nia, *i ) ); 
      }
    }
    virtual void proc_img( ostream & out, string const & img_fn ) = 0;
  };

  struct oct_resize_t : virtual public nesi, public proc_img_file_list_t // NESI(help="compare resize using boda versus octave",bases=["proc_img_file_list_t"], type_id="oct_resize")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t write_images; //NESI(default=0,help="write out octave and boda resized images?")

    virtual void proc_img( ostream & out, string const & img_fn ) {
      p_img_t img( new img_t );
      img->load_fn( img_fn.c_str() );
      //oct_featpyra_inner( img, 8, 10 ); return;
      uint32_t const interval = 10;
      uint32_t const imax = interval;
      for( uint32_t i = 1; i <= imax; ++i ) { 
	double const scale = pow(2.0d, 0.0d - (double(i) / double(interval) ) );
	oct_dfc_startup( dpm_fast_cascade_dir );
	p_nda_double_t oct_resize_out = oct_img_resize( img, scale );
	p_img_t ds_img = downsample_up_to_2x( img, scale ); // note: input ids_img is released here	  
	double h_scale = scale;
	for( uint32_t h = 0; h < 5; ++h ) {
	  if( ds_img->sz.dims_min() < 2 ) { break; } // this octave/interval is too small.
	  if( h ) {
	    h_scale = h_scale / 2.0d;
	    ds_img = downsample_2x( ds_img ); // note: input ids_img is released here	  
	    oct_resize_out = oct_img_resize( oct_resize_out, .5 );
	  }
	  if( write_images ) {
	    ds_img->save_fn_png( out_fn.exp + "." + str(i) + "_" + str(h) + ".png" );
	    p_img_t oct_img = create_p_img_from_p_nda_double_from_img( oct_resize_out );
	    oct_img->save_fn_png( out_fn.exp + "." + str(i) + "_" + str(h) + ".oct.png" );	
	  }
	  p_nda_double_t resize_out = create_p_nda_double_from_img( ds_img );

	  if( !(oct_resize_out->dims == resize_out->dims) ) {
	    out << strprintf( "oct_resize_out->dims=%s resize_out->dims=%s\n", str(oct_resize_out->dims).c_str(), str(resize_out->dims).c_str() );
	    assert_st( resize_out->dims.fits_in( oct_resize_out->dims ) );
	    oct_resize_out = clone_from_corner( resize_out->dims, oct_resize_out );
	  } 
	  out << strprintf( "scale=%s ssds_str(oct,boda)=%s\n", str(h_scale).c_str(),
			    ssds_str(oct_resize_out,resize_out).c_str() );
	}
	//printf( "resize_out=%s oct_resize_out=%s\n", str(resize_out).c_str(), str(oct_resize_out).c_str() );
      }	
#if 0	
      p_ostream out = ofs_open( out_fn.exp );
      bwrite( *out, boda_magic );
      bwrite_id( *out, string("resize_out") );
      bwrite( *out, string("p_nda_double_t") );
      bwrite( *out, resize_out );
      bwrite( *out, string("END_BODA") );
      //printf( "resize_out=%s\n", str(resize_out).c_str() );
#endif
    }
  };



  struct oct_featpyra_t : virtual public nesi, public proc_img_file_list_t // NESI(help="compare HOG pyramid generated using boda versus octave",bases=["proc_img_file_list_t"], type_id="oct_featpyra")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    virtual void proc_img( ostream & out, string const & img_fn ) {
      p_img_t img( new img_t );
      img->load_fn( img_fn.c_str() );

      p_nda_double_t boda_scales;
      vect_p_nda_double_t boda_feats;
      boda_scales = oct_featpyra_inner( boda_feats, img, 8, 10 );
      assert_st( boda_scales->elems.sz == boda_feats.size() );

      p_nda_double_t scales;
      vect_p_nda_double_t feats;

      oct_dfc_startup( dpm_fast_cascade_dir );
      dim_vector img_dv(img->sz.d[1], img->sz.d[0], 3);
      NDArray img_oct( img_dv );
      for( uint32_t y = 0; y < img->sz.d[1]; ++y ) {
	for( uint32_t x = 0; x < img->sz.d[0]; ++x ) {
	  for( uint32_t c = 0; c < 3; ++c ) {
	    img_oct(y,x,c) = img->pels.get()[y*img->row_pitch + x*img->depth + c];
	  }
	}
      }
      octave_value_list in;
      in(0) = img_oct; //octave_value( image_fn );
      octave_value_list boda_if_ret;
      {
	timer_t t( "oct_featpyra_boda_if_feat" );
	boda_if_ret = feval("boda_if_feat", in, 1 );
      }
      assert_st( !error_state );
      assert_st( boda_if_ret.length() == 1);
      //osm_print_keys( out, boda_if_ret(0) );
      assert_st( !error_state );


      octave_scalar_map ret_osm = boda_if_ret(0).scalar_map_value();
      //print_field( out, ret_osm, "scales" );
      //print_field( out, ret_osm, "feat" );
      NDArray oct_scales = get_field_as_NDArray( ret_osm, "scales" );
      assert( oct_scales.numel() == oct_scales.dims().elem(0) ); // should be N(x1)* 
      scales = create_p_nda_double_from_oct_NDArray( oct_scales );
      //printf( "scales=%s\n", str(scales).c_str() );
      vect_NDAarray oct_feats;
      get_ndas_from_field( oct_feats, ret_osm, "feat" );
      //printf( "scales.size()=%s\n", str(scales.size()).c_str() );
      //printf( "oct_feats.size()=%s\n", str(oct_feats.size()).c_str() );
      assert_st( scales->elems.sz == oct_feats.size() );
      vect_uint32_t pad = {0,2,3};
      for( vect_NDAarray::const_iterator i = oct_feats.begin(); i != oct_feats.end(); ++i ) {
	feats.push_back( create_p_nda_double_from_oct_NDArray( *i ) );
      }

      assert_st( scales->elems.sz == boda_scales->elems.sz );
      assert_st( feats.size() == boda_feats.size() );
      assert_st( feats.size() == scales->elems.sz );
      
      for( uint32_t i = 0; i < feats.size(); ++i ) {
	out << strprintf( "scale=%s boda_scale=%s\n", str(scales->elems[i]).c_str(), 
			  str(boda_scales->elems[i]).c_str() );
	out << strprintf( "feats: ssds_str(oct,boda)=%s\n", ssds_str(feats[i],boda_feats[i]).c_str() );
      }

//      p_ostream bo = ofs_open( pyra_out_fn + ".boda" );
//      write_scales_and_feats( bo, scales, feats );
#if 0
      for( uint32_t i = 0; i < feats.size(); ++i ) {
	printf( "scale=%s\n", str(scales->elems[i]).c_str() );
	printf( "feats=%s\n", str(feats[i]).c_str() );
      }
#endif
    }
  };

  struct run_dfc_t : virtual public nesi, public load_pil_t // NESI(
		     // help="run dpm fast cascade over pascal-VOC-format image file list",
		     // bases=["load_pil_t"], type_id="run_dfc")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    filename_t dpm_fast_cascade_dir; //NESI(default="%(dpm_fast_cascade_dir)",help="dpm_fast_cascade base dir")

    filename_t res_fn; //NESI(default="%(boda_output_dir)/%%s.txt",help="output: format for filenames of pascal-VOC format detection results file to write")
    filename_t prc_txt_fn; //NESI(default="%(boda_output_dir)/prc_",help="output: text prc curve base filename")
    filename_t prc_png_fn; //NESI(default="%(boda_output_dir)/mAP_",help="output: png prc curve base filename")


    virtual void main( nesi_init_arg_t * nia ) {
      load_img_db( 1 );
      p_vect_p_per_class_scored_dets_t scored_dets( new vect_p_per_class_scored_dets_t );      
      for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
	scored_dets->push_back( p_per_class_scored_dets_t( new per_class_scored_dets_t( *i ) ) );
	for (uint32_t ix = 0; ix < img_db->img_infos.size(); ++ix) {
	  p_img_info_t img_info = img_db->img_infos[ix];
	  oct_dfc( cout, dpm_fast_cascade_dir.exp, scored_dets->back(), img_info->full_fn, img_info->ix );
	}
	string c_res_fn = strprintf( res_fn.exp.c_str(), (*i).c_str() );
	write_results_file( img_db, c_res_fn, scored_dets->back() );
      }
      img_db->score_results( scored_dets, prc_txt_fn.exp, prc_png_fn.exp, 0 );
    }
  };



#include"gen/octif.cc.nesi_gen.cc"

}
