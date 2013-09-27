#include"boda_tu_base.H"
#include"str_util.H"
#include"results_io.H"
#include"pyif.H"
#include"octif.H"

namespace boda
{
  // working example cline:
  // time ../lib/boda score ~/bench/VOCdevkit/VOC2007/ImageSets/Main/bicycle_test.txt ~/research/ffld/build/ffld_VOC2007_bicycle_test_out.txt bicycle

  int boda_main( int argc, char **argv )
  {
    py_init( argv[0] );
    //oct_init();
    if( argc < 2 )
    {
      printf("usage: boda mode\n");
      printf("modes: score oct_init\n");
      return 1;
    }
    std::string const mode = argv[1];
    if(0) { }
    else if( mode == "score" ) 
    {
      if( argc != 5 ) { printf("usage: boda score list_fn res_fn class_name\n"); }
      else {
	std::string const list_fn = argv[2];
	std::string const res_fn = argv[3];
	std::string const class_name = argv[4];
	score_results_file( list_fn, res_fn, class_name ); 
      }
    }
    else if( mode == "oct_init" ) 
    {
      if( argc != 2 ) { printf("usage: boda oct_init\n"); }
      else {
	oct_init(); 
      }
    }
    else { rt_err( "unknown mode '" + mode + "'" ); }
    py_finalize();
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
