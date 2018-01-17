// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"nesi.H"
#include"str_util.H"
#include"pyif.H"
#include"octif.H"
#include"lexp.H"
#include"has_main.H"
#include"timers.H"
#include<ostream>
#include<iostream>

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

  using std::ostream;

  char const * const boda_help_usage = "boda help mode [field] [sub_field] [sub_sub_field] [...] ";
  char const * const boda_help_all_note = "(notes: use help_all instead of help to show hidden things. use help_all_ex to use a full lexp as the mode if you need on help on deeply polymophic structs)";
  char const * const boda_xml_usage = "boda xml command_file.xml[:element][:subelement][:...]";

  // scan all arguments (execpt 0) for anything that looks like a 'cry for help', so we can force help-mode if found 
  void check_has_help_arg_and_setup_help( int const argc, char const * const * const argv,
					  bool & do_help, bool & help_ex, bool & help_all, vect_string & help_args ) 
  {
    assert( (!do_help) && (!help_ex) && (!help_all) );
    assert( help_args.empty() );
    assert( argc > 0 );
    if( argc == 1 ) { do_help = 1; } // no arguments case is the same as a bare -h or --help option: print usage
    for( int32_t i = 1; i < argc; ++i ) {
      string arg = argv[i];
      strip_prefixes( arg, "--" );
      if( arg == "-h" ) { do_help = 1; } // it's not possible to enable help_ex / help_all with short opt form
      else if( startswith( arg, "help" ) ) { 
	do_help = 1;
	help_all |= startswith( arg, "help_all" );
	help_ex |=  endswith( arg, "_ex" );
      }
      if( do_help ) { break; }
      help_args.push_back( arg ); // note: stripped of "--"'s ...
    }
  }

  int boda_main_arg_proc( ostream & os, int argc, char **argv ) {
    string prefix; // empty prefix for printing usage
    string out; // output string for printing usage
    bool do_help = 0, help_ex = 0, help_all = 0;
    vect_string help_args;
    check_has_help_arg_and_setup_help( argc, argv, do_help, help_ex, help_all, help_args );
    // low-level help/testing modes that either cannot rely on NESI
    // support or where it seem to make little sense to use it
    if( do_help ) {
      if( help_args.empty() ) { 
	os << strprintf("   usage:   boda mode [--mode-arg=mode_val]*\n");
	os << strprintf("   usage:   %s\n", boda_xml_usage );
	os << strprintf("for help:   %s\n%s\n\n", boda_help_usage, boda_help_all_note );
	nesi_struct_hier_help( &cinfo_has_main_t, &out, prefix, help_all ); os << out;
      } else { 
	std::string const help_for_mode = help_args.front();
	help_args.erase( help_args.begin() );
	p_lexp_t lexp = help_ex ? parse_lexp( help_for_mode ) : make_list_lexp_from_one_key_val( "mode", help_for_mode );
	p_has_main_t has_main;
	lexp_name_val_map_t nvm( lexp );
	nesi_struct_make_p( &nvm, &tinfo_has_main_t, &has_main );
	nesi_struct_nesi_help( &tinfo_has_main_t, has_main.get(), &out, prefix, help_all, &help_args, 0 ); os << out;
      }
      return 1;
    }
    assert( argc > 1 );
    std::string const mode = argv[1];
    vect_string vs_argv;
    for( int32_t ai = 0; ai != argc; ++ai ) { string arg( argv[ai] ); vs_argv.push_back( arg ); }
    if( mode == "xml" ) {
      if( argc < 3 ) { os << strprintf("run command from xml file\nusage: %s\n", boda_xml_usage); }
      else {
        p_lexp_t xml_lexp = parse_lexp_xml_file( argv[2] );
        add_argv_options_to_lexp( xml_lexp, 0, os, vs_argv.begin() + 3, vs_argv.end() );
        create_and_run_has_main_t( xml_lexp );
      }
    // otherwise, in the common/main case, treat first arg as mode for has_main_t, with remaining
    // args uses as fields in cli-syntax: each arg must start with '--' (which is ignored), and
    // names a field (with "-"->"_"). '=' can used to split key from value in single arg, otherwise
    // the next arg is consumed as its value.
    } else {
      p_lexp_t lexp = get_lexp_from_argv( vs_argv, os ); 
      create_and_run_has_main_t( lexp );
    }
    return 0;
  }
  void boda_asserts( void ); // language/portability checks
  int boda_main( int argc, char **argv ) {
    boda_asserts();
    assert_st( argc > 0 );
    boda_dirs_init();
    py_init( argv[0] );
    oct_init();
    int const ret = boda_main_arg_proc( std::cout, argc, argv ); // split out for unit testing
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

int main( int argc, char **argv ) { 
  int const retval = boda::boda_main_wrap( argc, argv ); 
  // gee, who doesn't want octave to terminate the process for them in
  // some unknown way? note: if octave support is disabled, the stub
  // version of this function does nothing and we fall through to the
  // return below.
  boda::oct_exit( retval ); 
  return retval;
}
