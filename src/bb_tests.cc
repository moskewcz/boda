// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"test_base.H"
#include"lexp.H"
#include"str_util.H"
#include"has_main.H"
#include<boost/filesystem.hpp>
#include"pyif.H"
#include"img_io.H"
#include"xml_util.H"
#include"test_base.H"
#include<sstream>

namespace boda {
  using std::ostream;
  using namespace pugi;

  // all c++ programmers must use pointer-to-member at least once per decade:
  struct boda_base_test_run_t;
  typedef void (boda_base_test_run_t::*test_func_t)( void );
  void boda_main_arg_proc( ostream &os, int argc, char **argv ); // from boda.cc
  // from geom_prim.cc
  void u32_box_t1( void ); 
  void u32_box_t2( void );

  vect_string bb_test_fns = { "%(boda_test_dir)/regfile.txt", "/etc/passwd", "/etc/shadow", "/dev", "/dev/null", "/dev/null/baz", "/bin/sh", "fsdlkfsjdflksjd234234" };
  vect_string bb_img_fns = { "%(boda_test_dir)/valid.png", "%(boda_test_dir)/valid.jpg", 
			     "%(boda_test_dir)/invalid.png", "%(boda_test_dir)/invalid.jpg", 
			     "%(boda_test_dir)/unk_img_type.idk" };
  vect_string bb_xml_fns = { "%(boda_test_dir)/valid.xml", "%(boda_test_dir)/valid2.xml", 
			     "%(boda_test_dir)/invalid.xml" };
  vect_string bb_lc_strs = { "12.0", "234", "-1234", "sdfsd", "12s", "123e45", "0.0123e-12" };
			     

  struct boda_base_test_run_t : public test_run_t, public virtual nesi, public has_main_t // NESI(help="NESI wrapper for low-level boda tests; use test_boda_base() global func to run w/o NESI", bases=["has_main_t"], type_id="test_boda_base", hide=1 )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    char const * cur_tn;
    char const * cur_fn;
    string cur_str_out;
    void test_print( void ) {
      assert_st( cur_tn );
      printf( "tix=%s %s\n", str(tix).c_str(), cur_tn );
    }

    void test_run( char const * tn, test_func_t tf, char const * exp_err, char const * exp_out = 0 )
    {
      assert( ! ( exp_err && exp_out ) ); // can't expect both an error and output
      cur_tn = tn; // for test_print()
      bool had_err = 1;
      try {
	(this->*tf)();
	had_err = 0;
      } catch( rt_exception const & rte ) {
	if( !exp_err ) { test_fail_err( rte.err_msg ); } // expected no error, but got one
	else { 	// check if error is correct one
	  if( rte.err_msg != string(exp_err) ) { test_fail_wrong_err( 
	      strprintf( "  %s\nexpected:\n  %s\n", str(rte.err_msg).c_str(), exp_err ) ); 
	  }
	}
      }
      if( !had_err ) {
	if( exp_err ) { test_fail_no_err( exp_err ); }
	else if ( exp_out ) {
	  if( cur_str_out != string(exp_out) ) { test_fail_wrong_res( 
	      strprintf( "%s\nexpected:\n  %s\n", cur_str_out.c_str(), exp_out ) ); 
	  }
	}
	// note: further !had_err && !exp_err case handling (i.e. checking correct output) may be inside test function.
      }
      ++tix;
    }
    void remap_fns( vect_string & fns ) {
      for( uint32_t fn_ix = 0; fn_ix < fns.size(); ++fn_ix ) { 
	if( startswith( fns[fn_ix], "%(boda_test_dir)" ) ) {
	  fns[fn_ix] = py_boda_test_dir() + string( fns[fn_ix], 16 );
	}
      }
    }

    void test_run_fns( vect_string const & fns, char const * tn_base, test_func_t tf, 
		       char const * fn_mask, char const * err_fmt ) {
      assert_st( strlen(fn_mask) == fns.size() );
      for( uint32_t fn_ix = 0; fn_ix < fns.size(); ++fn_ix ) { 
	char const tt = fn_mask[fn_ix];
	if( tt == '-' ) { continue; } // skip fn
	cur_fn = fns[fn_ix].c_str();
	string exp_err;
	if( tt != '0' ) { exp_err = strprintf( err_fmt, cur_fn ); }
	test_run( (string(tn_base)+str(fn_ix)).c_str(), tf, (tt=='0')?0:exp_err.c_str() );
      }
      cur_fn = 0;
    }

    void test_run_tfns( char const * tn_base, test_func_t tf, char const * fn_mask, char const * err_fmt ) {
      test_run_fns( bb_test_fns, tn_base, tf, fn_mask, err_fmt  );
    }
    void test_run_ifns( char const * tn_base, test_func_t tf, char const * fn_mask, char const * err_fmt ) {
      test_run_fns( bb_img_fns, tn_base, tf, fn_mask, err_fmt );
    }

    void lc_str_d_tst( void ) { lc_str_d( cur_fn ); }
    void lc_str_u32_tst( void ) { lc_str_u32( cur_fn ); }
    void ma_p_t1( void ) { p_uint8_t p = ma_p_uint8_t( 100, 3 ); assert_st( p ); }
    void ma_p_t2( void ) { p_uint8_t p = ma_p_uint8_t( 100, 1024 ); assert_st( p ); }
    void eid_t1( void ) { ensure_is_dir( "/dev/null", 0 ); }
    //void eid_t2( void ) { ensure_is_dir( "<something that gives fs error?>", 0 ); }
    void eid_t3( void ) { ensure_is_dir( "/dev/null", 1 ); }
    void eid_t4( void ) { ensure_is_dir( "/dev/null/baz", 1 ); } // note: error isn't nice/correct. hmm.
    void eid_t5( void ) { ensure_is_dir( "/dev", 0 ); }
    void eirf_tst( void ) { ensure_is_regular_file( cur_fn ); }
    void ifso_tst( void ) { ifs_open( cur_fn ); } 
    void ofso_tst( void ) { ofs_open( cur_fn ); } 
    void mapfnro_tst( void ) { map_file_ro( cur_fn ); } 
    void boda_main_wrap( vect_string args ) {
      vector< char const * > argv;
      for( vect_string::const_iterator i = args.begin(); i != args.end(); ++i ) {
	argv.push_back( i->c_str() );
      }
      // FIXME: ignoring the output of main makes our tests less
      // brittle, but isn't ideal. we should check that the output is
      // correct in at least a few more cases?
      std::ostringstream oss;
      boda_main_arg_proc( oss, argv.size(), (char**)&argv.front() );
      cur_str_out = oss.str();
    }
    void boda_main_t1( void ) { boda_main_wrap( { "boda", "(foo=biz)" } ); }
    void boda_main_t2( void ) { boda_main_wrap( { "boda", "foo", "--foo" } ); }
    void boda_main_t3( void ) { boda_main_wrap( { "boda", "vst", "bar", "--dpf=5.0" } ); } 
    void boda_main_t4( void ) { boda_main_wrap( { "boda", "foo", "--bar=biz" } ); }
    void boda_main_t5( void ) { boda_main_wrap( { "boda", "foo", "--bar", "biz" } ); }
    void boda_main_t6( void ) { boda_main_wrap( { "boda", "help" } ); } 
    void boda_main_t7( void ) { boda_main_wrap( { "boda", "help_all" } ); } 
    void boda_main_t8( void ) { boda_main_wrap( { "boda", "help_all_ex" } ); } 
    void boda_main_t9_orig( void ) { boda_main_wrap( { "boda", "help", "vst", "boozle" } ); } 
    void boda_main_t9( void ) { boda_main_wrap( { "boda", "vst", "boozle", "help" } ); } 
    void boda_main_t10( void ) { boda_main_wrap( { "boda", "help", "vst", "dpf" } ); } 
    void boda_main_t11( void ) { boda_main_wrap( { "boda", "vst", "dpf", "baz", "help" } ); } 
    void u32_box_t1_( void ) { u32_box_t1(); }
    void u32_box_t2_( void ) { u32_box_t2(); }
    void load_img_tst( void ) { img_t img; img.load_fn( cur_fn ); }
    void za_img_tst( void ) { img_t img; img.set_sz_and_alloc_pels( {2,0} ); }
    void xml_gr_tst( void ) { xml_document doc; xml_node xn = xml_file_get_root( doc, cur_fn ); 
      xml_must_decend( cur_fn, xn, "bar" ); }
    void lexp_unused_t1( void ) { p_lexp_t l = parse_lexp( "(foo=bar)" ); ++l->use_cnt; ++l->kids.front().v->use_cnt;
      vect_string path; lexp_check_unused( l.get(), path ); }
    void lexp_unused_t2( void ) { p_lexp_t l = parse_lexp( "(foo=bar)" ); ++l->use_cnt; 
      vect_string path; lexp_check_unused( l.get(), path ); }
    void lexp_nvm_base( string const & s ) { 
      p_lexp_t l = parse_lexp( s ); 
      lexp_name_val_map_t nvm( l );
      nvm.init_nvm();
    }
    void lexp_t3( void ) { lexp_nvm_base( "foo" ); }
    void lexp_t4( void ) { lexp_nvm_base( "(foo=bar)" ); }
    void lexp_t5( void ) { lexp_nvm_base( "(foo=bar,foo=biz)" ); }

#ifdef TND
#error "TND already defined. oops."
#else
#define TND(s) #s, &boda_base_test_run_t::s
    void main( nesi_init_arg_t * nia ) {
      num_fail = 0;
      cur_tn = 0;
      tix = 0;
      remap_fns( bb_test_fns );
      remap_fns( bb_img_fns );
      remap_fns( bb_xml_fns );
      char const * expected_regfile_fmt = "error: expected path '%s' to be a regular file, but it is not.";

      test_run( TND(ma_p_t1), "error: posix_memalign( p, 3, 100 ) failed, ret=22" );
      test_run( TND(ma_p_t2), 0 );
      test_run( TND(eid_t1), "error: expected path '/dev/null' to be a directory, but it is not.");
      test_run( TND(eid_t3), "error: error while trying to create '/dev/null' directory: boost::filesystem::create_directory: File exists: \"/dev/null\"" );
      test_run( TND(eid_t4), "error: error while trying to create '/dev/null/baz' directory: boost::filesystem::create_directory: Not a directory: \"/dev/null/baz\"" ); // note: error not the best ...
      test_run( TND(eid_t5), 0 ); 
      test_run_tfns( TND(eirf_tst), "00011101", expected_regfile_fmt );
      test_run_tfns( TND(ifso_tst), "00-11101", expected_regfile_fmt );
      test_run_tfns( TND(ifso_tst), "--1-----", "error: can't open file '%s' for reading" );
      test_run_tfns( TND(ofso_tst), "-111011-", "error: can't open file '%s' for writing" );
      test_run_tfns( TND(mapfnro_tst), "00111101", "error: failed to open/map file '%1$s' (expanded: '%1$s') for reading" );
      test_run( TND(boda_main_t1), "error: specified mode name '(foo=biz)' parses as a list, and it must not be a list." );
      test_run( TND(boda_main_t2), "error: missing value for option '--foo': no '=' present, and no more args" );
      //test_run( TND(boda_main_t3), "error: expected option, but argument 'bar' does not start with '--'" ); // for now, pos args are allowed, so we get a different error
      test_run( TND(boda_main_t3), "error: unused input: pos_args:(0=bar)" );
      char const * bad_mode_foo_err = "error: type id str of 'foo' did not match any derived class of has_main_t\n";
      test_run( TND(boda_main_t4), bad_mode_foo_err ); // AKA success (no prior errors)
      test_run( TND(boda_main_t5), bad_mode_foo_err ); // AKA success (no prior errors)
      test_run( TND(boda_main_t6), 0 ); 
      test_run( TND(boda_main_t7), 0 ); 
      test_run( TND(boda_main_t8), 0 ); 
      test_run( TND(boda_main_t9_orig), 0, 0 ); // return usage, ignore args after help
      test_run( TND(boda_main_t9), 0, "struct 'various_stuff_t' has no field 'boozle', so help cannot be provided for it.\n" ); 
      test_run( TND(boda_main_t10), 0 ); 
      test_run( TND(boda_main_t11), 0, 
		"DESCENDING TO DETAILED HELP FOR field 'dpf' of type=double of struct 'various_stuff_t'\n" 
		"leaf type 'double' has no fields at all, certainly not 'baz', so help cannot be provided for it.\n" ); 
      test_run( TND(u32_box_t1_), "error: during from_pascal_coord_adjust(), box had 0 coord, expected >= 1" );
      test_run( TND(u32_box_t2_), 0 );
      test_run_ifns( TND(load_img_tst), "00---", "%s" );
      test_run_ifns( TND(load_img_tst), "--1--", "error: failed to load image '%s': lodepng decoder error 27: PNG file is smaller than a PNG header" );
      test_run_ifns( TND(load_img_tst), "---1-", "error: failed to load image '%s': tjDecompressHeader2 failed:Not a JPEG file: starts with 0x64 0x73" );
      test_run_ifns( TND(load_img_tst), "----1", "error: failed to load image '%s': could not auto-detect file-type from extention. known extention/types are: '.jpg':jpeg '.png':png" );
      test_run( TND(za_img_tst), "error: can't create zero-area image. requests WxH was 2x0" );
      test_run_fns( bb_xml_fns, TND(xml_gr_tst), "0-1", "error: loading xml file '%s' failed: Error parsing start element tag" );
      test_run_fns( bb_xml_fns, TND(xml_gr_tst), "-1-", "error: error: parsing xml file: '%s': expected to find child named 'bar' from node with name 'root'" );
      test_run( TND(lexp_unused_t1), 0 );
      test_run( TND(lexp_unused_t2), "error: unused input: foo:bar" );
      test_run( TND(lexp_t3), "error: invalid attempt to use string as name/value list. string was:foo" );
      test_run( TND(lexp_t4), 0 );
      test_run( TND(lexp_t5), "error: invalid duplicate name 'foo' in name/value list" );
      test_run_fns( bb_lc_strs, TND(lc_str_u32_tst), "1001111", "error: can't convert '%s' to uint32_t." );
      test_run_fns( bb_lc_strs, TND(lc_str_d_tst), "0001100", "error: can't convert '%s' to double." );

      if( num_fail ) { printf( "test_boda_base num_fail=%s\n", str(num_fail).c_str() ); }
    }
#undef TND
#endif
  };

#include"gen/bb_tests.cc.nesi_gen.cc"

}
