#include"boda_tu_base.H"
#include"nesi.H"
#include"str_util.H"
#include"pyif.H"
#include"octif.H"
#include"lexp.H"
#include"has_main.H"
#include<ostream>

namespace boda
{
#include"nesi_decls.H"
  void nesi_struct_hier_help( cinfo_t const * const ci, string * os, string & prefix, bool const show_all );
  void nesi_struct_nesi_help( tinfo_t const * tinfo, void * o, void * os_, 
			      bool const show_all, void * help_args_, uint32_t help_ix );

  extern tinfo_t tinfo_has_main_t;
  extern cinfo_t cinfo_has_main_t;
  // working example cline:
  // time ../lib/boda score ~/bench/VOCdevkit/VOC2007/ImageSets/Main/bicycle_test.txt ~/research/ffld/build/ffld_VOC2007_bicycle_test_out.txt bicycle

  char const * const boda_help_usage = "boda help '(mode=mode_name[,sub_mode=sub_mode_name])' [field] [sub_field] [sub_sub_field] [...] ";
  char const * const boda_help_all_note = "(note: use help_all instead of help to show hidden things)";

  int boda_main( int argc, char **argv ) {
    assert_st( argc >= 0 );
    py_init( argv[0] );
    oct_init();
    string prefix; // empty prefix for printing usage
    string out; // output string for printing usage
    if( argc < 2 ) {
      printf("   usage:   boda '(mode=mode_name,arg1=arg1_val,...)'\n");
      printf("for help:   %s\n%s\n\n", boda_help_usage, boda_help_all_note );
      nesi_struct_hier_help( &cinfo_has_main_t, &out, prefix, 0 ); printstr( out );
      return 1;
    }
    std::string const mode = argv[1];
    if(0) { } 
    else if( mode == "help" || mode == "help_all" ) {
      bool show_all = (mode == "help_all");
      if( argc < 3 ) { printf("error: boda help/help_all takes at least 1 argument, but got %s\nfor help:   %s\n%s\n\n",
			       str(argc-2).c_str(), boda_help_usage, boda_help_all_note);
	nesi_struct_hier_help( &cinfo_has_main_t, &out, prefix, show_all ); printstr( out );
	return 1;
      } else { 
	std::string const lexp_str = argv[2];
	p_lexp_t lexp = parse_lexp( lexp_str );
	p_has_main_t has_main;
	nesi_struct_make_p( &tinfo_has_main_t, &has_main, lexp.get() );
	vect_string help_args;
	for( uint32_t i = 3; i < (uint32_t)argc; ++i ) { help_args.push_back( string( argv[i] ) ); }
	nesi_struct_nesi_help( &tinfo_has_main_t, has_main.get(), &out, show_all, &help_args, 0 ); printstr( out );
      }
    }
    else if( mode == "test_nesi" || mode == "test_nesi_xml" ) {
      bool const xml = ( mode == "test_nesi_xml" );
      if( argc != 3 ) { printf("automated tests for nesi\nusage: boda test_nesi arg\n"); }
      else { 
	std::string const arg_str = argv[2];
	p_lexp_t lexp = xml ? parse_lexp_xml_file( arg_str ) : parse_lexp( arg_str );
	p_has_main_t has_main;
	void * pv = nesi_struct_make_p( &tinfo_has_main_t, &has_main, lexp.get() );
	nesi_struct_init( &tinfo_has_main_t, pv, lexp.get() );
	// check for unused fields in l
	vect_string path;
	lexp_check_unused( lexp.get(), path );
	printf( "*has_main=%s\n", str(*has_main).c_str() );
	has_main->main();

      }
    }
    else if( mode == "test_lexp" ) {
      if( argc != 2 ) { printf("automated tests for lexp\nusage: boda test_lexp\n"); }
      else { test_lexp(); }
    }
    else if( mode == "lexp" ) {
      if( argc != 3 ) { printf("test lexp parsing\nusage: boda lexp LEXP_STR\n"); }
      else {
	std::string const lexp_str = argv[2];
	p_lexp_t lexp = parse_lexp( lexp_str );
	printf( "*lexp=%s\n", str(*lexp).c_str() );
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
