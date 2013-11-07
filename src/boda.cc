#include"boda_tu_base.H"
#include"nesi.H"
#include"str_util.H"
#include"results_io.H"
#include"pyif.H"
#include"octif.H"
#include"lexp.H"
#include"has_main.H"
#include<ostream>

namespace boda
{
#include"nesi_decls.H"
  void nesi_struct_hier_help( cinfo_t const * const ci, std::ostream & os, string & prefix );
  void dump_nesi_struct_vars_help(std::ostream & top_ostream, nesi const & v);

  extern tinfo_t tinfo_has_main_t;
  extern cinfo_t cinfo_has_main_t;
  // working example cline:
  // time ../lib/boda score ~/bench/VOCdevkit/VOC2007/ImageSets/Main/bicycle_test.txt ~/research/ffld/build/ffld_VOC2007_bicycle_test_out.txt bicycle
  void downsample_test( std::string const & fn );

  int boda_main( int argc, char **argv )
  {
    py_init( argv[0] );
    oct_init();
    if( argc < 2 )
    {
      printf("usage: boda '(mode=mode_name,arg1=arg1_val,...)'\n");
      string prefix;
      nesi_struct_hier_help( &cinfo_has_main_t, std::cout, prefix );
      return 1;
    }
    std::string const mode = argv[1];
    if(0) { } 
    else if( mode == "test_nesi" ) 
    {
      if( argc != 3 ) { printf("automated tests for nesi\nusage: boda test_nesi arg\n"); }
      else { 
	std::string const lexp_str = argv[2];
	p_lexp_t lexp = parse_lexp( lexp_str );
	p_has_main_t has_main;
	void * pv = nesi_struct_make_p( &tinfo_has_main_t, &has_main, lexp.get() );
	nesi_struct_init( &tinfo_has_main_t, pv, lexp.get() );
	// check for unused fields in l
	vect_string path;
	lexp_check_unused( lexp.get(), path );

	printf( "*has_main=%s\n", str(*has_main).c_str() );
	dump_nesi_struct_vars_help( std::cout, *has_main );

	has_main->main();

      }
    }
    else if( mode == "test_lexp" ) 
    {
      if( argc != 2 ) { printf("automated tests for lexp\nusage: boda test_lexp\n"); }
      else { test_lexp(); }
    }
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
