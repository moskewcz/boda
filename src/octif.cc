#include"boda_tu_base.H"
#include<iostream>
#include<octave/oct.h>
#include<octave/octave.h>
#include<octave/parse.h>
#include<octave/oct-map.h>
//#include<octave/mxarray.h>
//#include<octave/mexproto.h>
#include<fstream>

#include"model.h"
#include"geom_prim.H"
#include"str_util.H"

namespace boda 
{
  using namespace::std;

  void print_field( octave_scalar_map const & osm, string const & fn )
  {
    octave_value fv = osm.contents(fn.c_str());
    assert_st( !error_state && fv.is_defined() );
    cout << fn << "="; fv.print(cout);
  }

  void oct_init( void )
  {
    //string const mat_fn = "/home/moskewcz/svn_work/dpm_fast_cascade/voc-release5/VOC2007/car_final.mat";
    string const mat_fn = "/home/moskewcz/svn_work/dpm_fast_cascade/voc-release5/car_final_cascade.mat";
    string_vector argv (2);
    argv(0) = "embedded";
    argv(1) = "-q";
    octave_main (2, argv.c_str_vec (), 1);
    octave_value_list in;
    in(0) = octave_value( mat_fn );
    octave_value_list out = feval ("load", in, 1);
    assert_st( !error_state && (out.length() > 0) );
    //cout << "load of ["  << in(0).string_value () << "] is "; out(0).print(cout); cout << endl;
    //cout << "load ret="; out(0).print(cout); cout << endl;
    octave_scalar_map osm = out(0).scalar_map_value();
    assert_st( !error_state );
    octave_value mod = osm.contents("csc_model");
    assert_st( !error_state && mod.is_defined() );
    octave_scalar_map mod_osm = mod.scalar_map_value();
    assert_st( !error_state );
    cout << "keys=";
    for (int i = 0; i < mod_osm.keys().length(); ++i)
    {
      cout << mod_osm.keys()[i] << " ";
    }
    cout << endl;
    print_field( mod_osm, "thresh" );
    print_field( mod_osm, "sbin" );
    print_field( mod_osm, "interval" );
#if 1
    mxArray mxa( mod );
    Model mod2( &mxa );
#endif
  }


  void oct_dfc( string const & class_name, string const & impath )
  {
    string_vector argv (2);
    argv(0) = "embedded";
    argv(1) = "-q";
    octave_main (2, argv.c_str_vec (), 1);
    int parse_ret = 0;
    eval_string("cd /home/moskewcz/svn_work/dpm_fast_cascade/voc-release5", 0, parse_ret);
    assert_st( !error_state );
    eval_string("pkg load image", 0, parse_ret);
    assert_st( !error_state );
    feval("startup" );
    assert_st( !error_state );


    string const mat_fn = "/home/moskewcz/svn_work/dpm_fast_cascade/voc-release5/VOC2007/"+class_name+"_final.mat";
    octave_value_list in;
    in(0) = octave_value( mat_fn );
    octave_value_list out = feval ("load", in, 1);
    assert_st( !error_state && (out.length() > 0) );
    //cout << "load of ["  << in(0).string_value () << "] is "; out(0).print(cout); cout << endl;
    //cout << "load ret="; out(0).print(cout); cout << endl;
    octave_scalar_map osm = out(0).scalar_map_value();
    assert_st( !error_state );
    octave_value mod = osm.contents("model");
    assert_st( !error_state && mod.is_defined() );
    octave_scalar_map mod_osm = mod.scalar_map_value();
    assert_st( !error_state );
    cout << "keys=";
    for (int i = 0; i < mod_osm.keys().length(); ++i)
    {
      cout << mod_osm.keys()[i] << " ";
    }
    cout << endl;
    print_field( mod_osm, "thresh" );
    print_field( mod_osm, "sbin" );
    print_field( mod_osm, "interval" );
    assert_st( !error_state );
   
    in(0) = octave_value( impath );
    in(1) = mod;
    octave_value_list boda_if_ret = feval("boda_if", in, 2 );
    assert_st( !error_state );
    assert_st( boda_if_ret.length() == 1);
    assert_st( boda_if_ret(0).is_matrix_type() );
    // unclear if this will always work and/or how to error check, since there are many 'matrix' types ...
    Matrix det_boxes = boda_if_ret(0).matrix_value(); 
    assert_st( !error_state );
    for( octave_idx_type i = 0; i < det_boxes.dim1(); ++i ) {
      assert( det_boxes.dim2() >= 5 ); // should have at least 4 columns to form detection bbox + 1 for score
      u32_box_t det;
      for( uint32_t p = 0; p < 2; ++p ) { // upper/lower
	for( uint32_t d = 0; d < 2; ++d ) { // x/y
	  det.p[p].d[d] = det_boxes(i,(p*2)+d);
	}
      }
      double const score = det_boxes(i,det_boxes.dim2()-1);
      det.from_pascal_coord_adjust();
      printf( "det=%s score=%s\n", str(det).c_str(), str(score).c_str() );
    }


  }
}
