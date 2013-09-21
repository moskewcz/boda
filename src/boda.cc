#include"boda_tu_base.H"
#include<numpy/arrayobject.h>
#include"str_util.H"
#include"results_io.H"
#include"pyif.H"

namespace boda
{
  // working example cline:
  // time ../lib/boda score ~/bench/VOCdevkit/VOC2007/ImageSets/Main/bicycle_test.txt ~/research/ffld/build/ffld_VOC2007_bicycle_test_out.txt bicycle

  int boda_main( int argc, char **argv )
  {
    Py_SetProgramName(argv[0]);
    Py_Initialize();
    py_path_setup();
    if( _import_array() < 0 ) { rt_err( "failed to import numpy" ); }
    if( argc < 5 )
    {
      printf("usage: boda score list_fn res_fn class_name\n");
      return 1;
    }
    std::string const mode = argv[1];
    std::string const list_fn = argv[2];
    std::string const res_fn = argv[3];
    std::string const class_name = argv[4];
    if(0) { }
    else if( mode == "score" ) { score_results_file( list_fn, res_fn, class_name ); }
    else { rt_err( "unknown mode '" + mode + "'" ); }
    Py_Finalize();
    return 0;
  }
  int boda_main_wrap( int argc, char **argv )
  {
    int ret = 1;
    try { ret = boda_main( argc, argv); }
    catch( rt_exception const & e ) { printstr( e.what_and_stacktrace() ); ret = e.get_ret_code(); }
    catch( std::exception const & e ) { printf( "exiting, top-level std::exception, what=%s\n", e.what() ); }
    catch( ... ) { printf( "exiting, top-level unknown exception\n" ); }
    return ret;
  }
}

int main( int argc, char **argv ) { return boda::boda_main_wrap( argc, argv ); }
