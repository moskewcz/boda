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

namespace boda 
{
  using namespace::std;
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
    if (!error_state && out.length () > 0)
    {
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
      octave_value thresh = mod_osm.contents("thresh");
      assert_st( !error_state && thresh.is_defined() );
      cout << "thresh="; thresh.print(cout); cout << endl;
#if 1
      mxArray mxa( mod );
      Model mod2( &mxa );
#endif
      
    }
    else
    {
      std::cout << "invalid\n";
    }
  }
}
