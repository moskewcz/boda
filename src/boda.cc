#include"boda_tu_base.H"
#include"nesi.H"
#include"str_util.H"
#include"pyif.H"
#include"octif.H"
#include"lexp.H"
#include"has_main.H"
#include"timers.H"
#include<ostream>


// boda.cc is the top level of control flow. most of the code/logic here is for the top level UI:
// help, cli-format arugment processing, top-level xml file processing. this is also where various
// init/finalize functions are called for the python interface, octave interface, and logging-timers
// functionality (from timers.H). finally, main() and the top-level exception handling wrappers are
// here as well.
// FIXME: factor NESI related help/cli/xml handling code out of here?
namespace boda
{
#include"nesi_decls.H"
  void nesi_struct_hier_help( cinfo_t const * const ci, string * os, string & prefix, bool const show_all );
  void nesi_struct_nesi_help( tinfo_t const * tinfo, void * o, void * os_, 
			      bool const show_all, void * help_args_, uint32_t help_ix );

  extern tinfo_t tinfo_has_main_t;
  extern cinfo_t cinfo_has_main_t;

  char const * const boda_help_usage = "boda help mode [field] [sub_field] [sub_sub_field] [...] ";
  char const * const boda_help_all_note = "(notes: use help_all instead of help to show hidden things. use help_all_ex to use a full lexp as the mode if you need on help on deeply polymophic structs)";
  char const * const boda_xml_usage = "boda xml_command_file.xml[:element][:subelement][:...]";

  int boda_main_arg_proc( int argc, char **argv ) {
    string prefix; // empty prefix for printing usage
    string out; // output string for printing usage
    std::string const mode = argv[1];
    if(0) { } 
    // low-level help/testing modes that either cannot rely on NESI
    // support or where it seem to make little sense to use it
    else if( mode == "help" || mode == "help_all" || mode == "help_all_ex" ) {
      bool show_all = startswith( mode, "help_all" );
      bool help_ex =  endswith( mode, "_ex" );
      if( argc < 3 ) { printf("error: boda help/help_all takes at least 1 argument, but got %s\nfor help:   %s\n%s\n\n",
			       str(argc-2).c_str(), boda_help_usage, boda_help_all_note);
	nesi_struct_hier_help( &cinfo_has_main_t, &out, prefix, show_all ); printstr( out );
	return 1;
      } else { 
	std::string const help_for_mode = argv[2];
	p_lexp_t lexp = help_ex ? parse_lexp( help_for_mode ) : make_list_lexp_from_one_key_val( "mode", help_for_mode );
	p_has_main_t has_main;
	nesi_struct_make_p( &tinfo_has_main_t, &has_main, lexp.get() );
	vect_string help_args;
	for( uint32_t i = 3; i < (uint32_t)argc; ++i ) { help_args.push_back( string( argv[i] ) ); }
	nesi_struct_nesi_help( &tinfo_has_main_t, has_main.get(), &out, show_all, &help_args, 0 ); printstr( out );
      }
    }
    else if( mode == "xml" ) {
      if( argc != 3 ) { printf("run command from xml file\nusage: %s\n", boda_xml_usage); }
      else { create_and_run_has_main_t( parse_lexp_xml_file( argv[2] ) ); }
    // otherwise, in the common/main case, treat first arg as mode for has_main_t, with remaining
    // args uses as fields in cli-syntax: each arg must start with '--' (which is ignored), and
    // names a field (with "-"->"_"). '=' can used to split key from value in single arg, otherwise
    // the next arg is consumed as its value.
    } else {    
      p_lexp_t lexp = make_list_lexp_from_one_key_val( "mode", mode );
      assert_st( lexp->kids.size() == 1 ); // should just be mode
      if( !lexp->kids[0].v->leaf_val.exists() ) {
	rt_err( "specified mode name '"+mode+"' parses as a list, and it must not be a list." );
      }
      for( int32_t ai = 2; ai != argc; ++ai ) {
	string arg( argv[ai] );
	if( !startswith( arg, "--" ) ) { rt_err( strprintf("expected option, but argument '%s' does not start with '--'",
							   arg.c_str() ) ); }
	string key;
	string val;
	bool key_had_eq = 0;
	for (string::const_iterator i = arg.begin() + 2; i != arg.end(); ++i) {
	  if( (*i) == '=' ) { 
	    if( (i+1) == arg.end() ) { printf( "warning empty/missing value after '=' in option '%s'\n", arg.c_str() ); }
	    val = string( i+1, string::const_iterator( arg.end() ) ); 
	    key_had_eq = 1;
	    break; 
	  }
	  key.push_back( ((*i)=='-')? '_' : (*i) );
	}
	if( !key_had_eq ) {
	  if( (ai + 1) == argc ) { rt_err( strprintf("missing value for option '%s': no '=' present, and no more args",
						     arg.c_str() ) ); }
	  val = string( argv[ai+1] );
	  ++ai;
	  if( startswith( val, "--" ) ) { 
	    printf("warning: option '%s's value '%s' starts with '--', did you forget a value?\n", 
		   arg.c_str(), val.c_str() );
	  }
	  if( val.empty() ) { printf("warning: option '%s's value '' is empty.\n", arg.c_str() ); }
	}
	lexp->add_key_val( key, val ); 
      }
      create_and_run_has_main_t( lexp );
    }
    return 0;
  }
  int boda_main( int argc, char **argv ) {
    assert_st( argc >= 0 );
    py_init( argv[0] );
    oct_init();
    string prefix; // empty prefix for printing usage
    string out; // output string for printing usage
    if( argc < 2 ) {
      printf("   usage:   boda mode [--mode-arg=mode_val]*\n");
      printf("   usage:   %s\n", boda_xml_usage );
      printf("for help:   %s\n%s\n\n", boda_help_usage, boda_help_all_note );
      nesi_struct_hier_help( &cinfo_has_main_t, &out, prefix, 0 ); printstr( out );
      return 1;
    }
    int const ret = boda_main_arg_proc( argc, argv ); // split out for unit testing
    global_timer_log_finalize();
    py_finalize();
    return ret;
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
