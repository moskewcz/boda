#include"boda_tu_base.H"
#include"str_util.H"
#include"results_io.H"

namespace boda
{
  int boda_main( int argc, char **argv )
  {
    if( argc < 2 )
    {
      printf("usage: boda arg\n");
      return 1;
    }
    read_results_file( std::string(argv[1]) );
    return 0;
  }
}

int main( int argc, char **argv ) { return boda::boda_main( argc, argv ); }
