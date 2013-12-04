#include"boda_tu_base.H"
#include"has_main.H"
#include"pyif.H"
#include"str_util.H"
#include"xml_util.H"
#include"lexp.H"
#include"nesi.H"
#include"timers.H"
#include<boost/regex.hpp>
#include<boost/filesystem.hpp>

namespace boda 
{
  using pugi::xml_node;
  using pugi::xml_document;

  using boost::regex;
  using boost::regex_search;

  using boost::filesystem::path;
  using boost::filesystem::filesystem_error;

  struct various_stuff_t;
  typedef shared_ptr< various_stuff_t > p_various_stuff_t;
  typedef vector< p_various_stuff_t > vect_p_various_stuff_t;
  typedef vector< double > vect_double;
  typedef vector< uint64_t > vect_uint64_t;
  typedef shared_ptr< double > p_double;
  typedef shared_ptr< string > p_string;
  struct one_p_string_t : public virtual nesi // NESI(help="struct with one p_string")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    p_string s; // NESI(help="foo")
  };
  typedef vector< one_p_string_t > vect_one_p_string_t;
  struct various_stuff_t : public virtual nesi, public has_main_t // NESI(help="test of various base types in nesi", bases=["has_main_t"], type_id="vst", hide=1 )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint64_t u64; //NESI(help="a u64 with a default",default="345")
    double dpf; //NESI(req=1)
    double dpf_nr; //NESI(default="233.5")
    vect_double vdpf; //NESI()
    p_double pdpf; //NESI()
    vect_uint64_t vu64; //NESI()
    vect_p_various_stuff_t vvs; //NESI()
    vect_one_p_string_t vops; //NESI()
    one_p_string_t ops; //NESI()

    virtual void main( nesi_init_arg_t * nia ) {
      printf("vst::main()\n");
    }
  };

  struct sub_vst_t : public virtual nesi, public various_stuff_t // NESI(help="sub type of vst", bases=["various_stuff_t"], type_id="sub_vst")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

  };

  struct sub_vst_2_t : public virtual nesi, public various_stuff_t // NESI(help="sub type of vst", bases=["various_stuff_t"], type_id="sub_vst_2",tid_vn="sub_mode")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string sub_mode; //NESI(help="name of sub_mode to run",req=1)
  };

  struct sub_sub_vst_2_t : public virtual nesi, public various_stuff_t // NESI(help="sub type of vst", bases=["sub_vst_2_t"], type_id="sub_sub_vst_2")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
  };


  struct nesi_test_t : public virtual nesi // NESI(help="nesi test case")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string test_name; //NESI(help="name of test",req=1)
    p_has_main_t command; //NESI(help="input",req=1)
  };
  typedef vector< nesi_test_t > vect_nesi_test_t; 
  typedef shared_ptr< nesi_test_t > p_nesi_test_t; 
  typedef vector< p_nesi_test_t > vect_p_nesi_test_t;

  string tp_if_rel( string const & fn ) {
    assert_st( !fn.empty() );
    if( fn[0] == '/' ) { return fn; }
    return py_boda_test_dir() + "/" + fn;
  }

#include"nesi_decls.H"

  void run_system_cmd( string const &cmd, bool const verbose ) {
    if( verbose ) { printstr( cmd + "\n" ); }
    int const sys_ret = system( cmd.c_str() );
    assert_st( sys_ret == 0 );
  }

  extern tinfo_t tinfo_vect_p_nesi_test_t;
  struct test_modes_t : public virtual nesi, public has_main_t // NESI(help="test of modes in various configurations", bases=["has_main_t"], type_id="test_modes" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string xml_fn; //NESI(default="modes_tests.xml",help="xml file containing list of tests. relative paths will be prefixed with the boda test dir.")
    vect_p_nesi_test_t tests; //NESI(help="populated via xml_fn")
    string filt; //NESI(default=".*",help="regexp over test name of what tests to run (default runs all tests)")
    uint32_t verbose; //NESI(default=0,help="if true, print each test lexp before running it")
    uint32_t update_failing; //NESI(default=0,help="if true, update archives for all run tests that fail.")
    
    string boda_test_dir; //NESI(help="boda base test dir (generally set via boda_cfg.xml)",req=1)

    virtual void main( nesi_init_arg_t * nia ) {
      set_string seen_test_names;
      seen_test_names.insert( "good" ); // reserved sub-dir to hold known good results

      path good_dir = path(boda_output_dir.exp) / "good";
      ensure_is_dir( good_dir, 1 );

      regex filt_regex( filt );
      lexp_name_val_map_t nvm;
      nvm.parent = nia;
      nvm.insert_leaf( "boda_output_dir", "%(test_name)" );
      //p_lexp_t boda_test_cfg = parse_lexp_xml_file( tp_if_rel( "boda_test_cfg.xml" ) ); // unneeded complexity?
      //nvm.populate_from_lexp( boda_test_cfg.get() );

      string const full_xml_fn = tp_if_rel(xml_fn);
      nesi_init_and_check_unused_from_xml_fn( &nvm, &tinfo_vect_p_nesi_test_t, &tests, full_xml_fn );
      for (vect_p_nesi_test_t::iterator i = tests.begin(); i != tests.end(); ++i) {
	bool const seen_test_name = !seen_test_names.insert( (*i)->test_name ).second;
	if( seen_test_name ) { rt_err( "duplicate or reserved (e.g. 'good') test name:" + (*i)->test_name ); }
	if( regex_search( (*i)->test_name, filt_regex ) ) {
	  if( verbose ) { std::cout << (**i) << std::endl; }
	  timer_t t("mode_test");
	  // note: no test may be named 'good'
	  path gen_test_out_dir = path(boda_output_dir.exp) / (*i)->test_name;
	  // first, remove gen_test_out_dir if it exists.
	  if( exists( gen_test_out_dir ) ) {
	    assert_st( is_directory( gen_test_out_dir ) );
	    uint32_t const num_rem = remove_all( gen_test_out_dir );
	    assert( num_rem );
	  }
	  (*i)->command->base_setup(); // note: sets command->boda_output_dir *and* creates it
	  (*i)->command->main( &nvm );	

	  // note: test_out_dir should equivalent to gen_test_out_dir (but not ==). we check that:
	  path const test_out_dir( (*i)->command->boda_output_dir.exp );
	  assert_st( equivalent( test_out_dir, gen_test_out_dir ) );
	  if( !exists( test_out_dir ) ) { // test must create its output dir
	    rt_err( strprintf( "test '%s' did not create its expected output directory '%s'.", 
			       (*i)->test_name.c_str(), test_out_dir.c_str() ) );
	  }

 	  // note: test_good_dir will be relative to the *test_modes* output_dir, which is usually '.'
	  path const test_good_dir = good_dir / (*i)->test_name; 
	  path const test_good_arc = path(boda_test_dir) / "mt_good" / ( (*i)->test_name + ".tbz2");
	  bool update_archive = 0;
	  if( !exists( test_good_arc ) ) {
	    printf("no existing good results archive for test %s.\n",(*i)->test_name.c_str());
	    update_archive = 1;
	  } else { // achive exists, unpack it
	    assert_st( is_regular_file( test_good_arc ) );
	    // first, remove test_good_dir if it exists.
	    if( exists( test_good_dir ) ) {
	      assert_st( is_directory( test_good_dir ) );
	      uint32_t const num_rem = remove_all( test_good_dir );
	      assert_st( num_rem );
	    }
	    bool const did_create = ensure_is_dir( test_good_dir, 1 ); // create good dir, must not exists
	    assert_st( did_create );
	    run_system_cmd( strprintf("tar -C %s -xjf %s",
				      test_good_dir.string().c_str(),test_good_arc.c_str()), 0 );
	    // compare good and test directories
	    bool output_good = 0;

	    if( (!output_good) && update_failing ) { 
	      printf("AUTOUPDATE: test %s failed, will update.\n",(*i)->test_name.c_str());
	      update_archive = 1; 
	    }

	  }
	  
	  if( update_archive ) {
	    printf("UPDATING good results archive for test %s.\n",(*i)->test_name.c_str());
	    run_system_cmd( strprintf("tar -C %s -cjf %s .",
				      test_out_dir.string().c_str(),test_good_arc.c_str()), 0 );
	  }
	}
      }
    }
  };

#include"gen/test_nesi.cc.nesi_gen.cc"
}

