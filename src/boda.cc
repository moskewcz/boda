#include"boda_tu_base.H"
#include"str_util.H"
#include"results_io.H"

namespace boda
{
  int boda_main( int argc, char **argv )
  {
    if( argc < 3 )
    {
      printf("usage: boda mode arg\n");
      return 1;
    }
    std::string const mode = argv[1];
    std::string const fn = argv[2];
    if(0) { }
    else if( mode == "res" ) { read_results_file( fn ); }
    else if( mode == "il" ) { read_image_list_file( fn ); }
    else { rt_err( "unknown mode '" + mode + "'" ); }
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
