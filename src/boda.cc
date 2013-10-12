#include"boda_tu_base.H"
#include"str_util.H"
#include"results_io.H"
#include"pyif.H"
#include"octif.H"
#include"lexp.H"

namespace boda
{
  // working example cline:
  // time ../lib/boda score ~/bench/VOCdevkit/VOC2007/ImageSets/Main/bicycle_test.txt ~/research/ffld/build/ffld_VOC2007_bicycle_test_out.txt bicycle
  void downsample_test( std::string const & fn );

  int boda_main( int argc, char **argv )
  {
    py_init( argv[0] );
    oct_init();
    if( argc < 2 )
    {
      printf("usage: boda mode\n");
      printf("modes: load_pil score oct_test oct_dfc ds_test\n");
      return 1;
    }
    std::string const mode = argv[1];
    if(0) { } 
    else if( mode == "lexp" ) 
    {
      if( argc != 3 ) { printf("test lexp parsing\nusage: boda lexp LEXP_STR\n"); }
      else {
	std::string const lexp_str = argv[2];
	p_lexp_t lexp = parse_lexp( lexp_str );
	printf( "*lexp=%s\n", str(*lexp).c_str() );
      }
    }
    else if( mode == "load_pil" ) 
    {
      if( argc != 3 ) { printf("load pascal image list file\nusage: boda load_pascal_img_list list_fn\n"); }
      else {
	std::string const list_fn = argv[2];
	read_pascal_image_list_file( list_fn, 1 );
      }
    }
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
    else if( mode == "run_dfc" ) 
    {
      if( argc != 5 ) { printf("usage: boda run_dfc list_fn res_fn class_name\n"); }
      else {
	std::string const list_fn = argv[2];
	std::string const res_fn = argv[3];
	std::string const class_name = argv[4];
	run_dfc( list_fn, res_fn, class_name ); 
      }
    }
    else if( mode == "oct_test" ) 
    {
      if( argc != 2 ) { printf("usage: boda oct_test\n"); }
      else {
	oct_test(); 
      }
    }
    else if( mode == "oct_dfc" ) 
    {
      if( argc != 4 ) { printf("usage: boda oct_dfc class_name impath\n"); }
      else {
	std::string const class_name = argv[2];
	std::string const impath = argv[3];
	oct_dfc( p_vect_scored_det_t(), class_name, impath, 0 ); 
      }
    }
    else if( mode == "ds_test" ) 
    {
      if( argc != 3 ) { printf("usage: boda ds_test img_fn\n"); }
      else {
	std::string const img_fn = argv[2];
	downsample_test( img_fn );
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
