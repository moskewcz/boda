// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"build_info.H"
#include"has_main.H"
#include"pyif.H"
#include"str_util.H"
#include"xml_util.H"
#include"lexp.H"
#include"nesi.H"
#include"timers.H"
#include<boost/regex.hpp>
#include<boost/filesystem.hpp>
#include<boost/iostreams/device/mapped_file.hpp>
#include<boost/iostreams/stream.hpp>         
#include<boost/program_options/parsers.hpp> // for split_unix()

#include"dtl/dtl.hpp"
#include"img_io.H"
#include"test_base.H"

namespace boda 
{
  using pugi::xml_node;
  using pugi::xml_document;

  using boost::regex;
  using boost::regex_search;

  using boost::filesystem::path;
  using boost::filesystem::exists;
  using boost::filesystem::is_directory;
  using boost::filesystem::is_regular_file;
  using boost::filesystem::filesystem_error;
  using boost::filesystem::recursive_directory_iterator;
  using boost::filesystem::file_size;

  typedef boost::iostreams::stream<mapped_file_source> mfs_istream;

  struct cmd_test_t : public virtual nesi // NESI(help="nesi test case")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string test_name; //NESI(help="name of test",req=1)
    p_string needs; //NESI(help="comma-seperated list of boda feature names needed for this test. if not avail, will skip.")
    uint32_t slow; //NESI(default=0,help="is test slow to run (and thus disabled by default)?")
    uint32_t ignore_missing_outputs; //NESI(default=0,help="if 1, don't treat missing outputs (aka file only in known-good) as failures (but still warn)")
    p_string err; //NESI(help="expected error (if any)")
    p_has_main_t command; //NESI(help="input")
    p_string cli_str; //NESI(help="cli-string-format command")
  };
  typedef vector< cmd_test_t > vect_cmd_test_t; 
  typedef shared_ptr< cmd_test_t > p_cmd_test_t; 
  typedef vector< p_cmd_test_t > vect_p_cmd_test_t;

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
    vect_one_p_string_t vops; //NESI()
    one_p_string_t ops; //NESI()
    vect_string vstr; //NESI()
    filename_t fn; //NESI(default="yo.mom")
    vect_p_cmd_test_t vct; //NESI()
    p_nda_t nda; //NESI()
    filename_t out_fn; //NESI(default="%(boda_output_dir)/test_vst.txt",help="output: print out nda, if specified.")

    virtual void main( nesi_init_arg_t * nia ) {
      p_ostream out = ofs_open( out_fn.exp );
      (*out) << strprintf("vst::main()\n");
      if( nda ) { (*out) << strprintf( "nda=%s\n", str(nda).c_str() ); }
    }
  };

  struct comp_ndas_t : public virtual nesi, public has_main_t // NESI(help="compare two nda's", bases=["has_main_t"], type_id="comp-ndas", hide=1 )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    nda_t a; //NESI(req=1)
    nda_t b; //NESI(req=1)
    virtual void main( nesi_init_arg_t * nia ) {
      printf( "a=%s\n", str(a).c_str() );
      printf( "b=%s\n", str(b).c_str() );
      printf( "a<b=%s\n", str(a<b).c_str() );
      printf( "b<a=%s\n", str(b<a).c_str() );
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


  struct nesi_test_t
  {
    string name;
    string desc;
    string in;
    char const * err_fmt;
  };

  string const ntb( "(mode=vst,boda_output_dir=." ); // note unmatched open paren ...
  string const ntb_b = ntb + ",dpf=3.4,";
  string const ntb_fn = ntb_b + "fn=";
  nesi_test_t nesi_tests[] = {
    { "bad_mode", "", "(mode=foozledefoo)", "error: type id str of 'foozledefoo' did not match any derived class of has_main_t\n"},
    { "vect_init_t1", "", ntb+",dpf=3.4,vdpf=(li_0=23.4))", 0},
    { "vect_init_t2", "", ntb+",dpf=3.4,vdpf=23.4)", "var 'vdpf': error: invalid attempt to use string as name/value list for vector init. string was:23.4"},
    { "no_req_val_t3", "", ntb+")", "error: missing required value for var 'dpf'"},
    { "bad_list_as_val_t1", "", ntb+",dpf=(li_0=3.4,li_1=34.0))", "var 'dpf': error: invalid attempt to use name/value list as double (double precision floating point number) value. list was:(li_0=3.4,li_1=34.0)"},
    { "bad_val_t1", "", ntb+",dpf=2jj2)", "var 'dpf': error: can't convert '2jj2' to double (double precision floating point number)."},
    // nda_t error cases (there are a bunch ...)
    { "bad_nda_1", "", ntb_b+"nda=2jj2)", "var 'nda': error: invalid attempt to use string as name/value list for nda_t (N-D Array) init. string was:2jj2"},
    { "bad_nda_2", "", ntb_b+"nda=())", "var 'nda': error: nda_t has neither 'tn' nor 'dims' field, can't determine type; set at least one."},
    { "bad_nda_3", "", ntb_b+"nda=(jimmy=no-fun))", "var 'nda': error: nda_t has no field named jimmy. valid fields are 'tn' (string; type name), 'dims' (dims_t), and 'v' (string; space-or-colon-seperated list of values" },
    { "bad_nda_4", "", ntb_b+"nda=(dims=()))", "var 'nda': error: nda_t 'dims' field specified empty dims." },
    { "bad_nda_5", "", ntb_b+"nda=(dims=jimbo))", "var 'nda': nda_t dims: error: invalid attempt to use string as name/value list for dims_t (N-D Array Dimentions) init. string was:jimbo" },
    { "bad_nda_6", "", ntb_b+"nda=(tn=uint32_t,dims=(v=2)))", 0 }, // empty ndas allowed now; error was: "var 'nda': error: nda_t has no 'v' field; currently, only non-empty nda_t's may be specified." },
    { "bad_nda_7", "", ntb_b+"nda=(tn=foo,v=baz))", "var 'nda': error: missing key:foo" }, // not ideal! means 'bad type'
    { "bad_nda_8", "", ntb_b+"nda=(tn=float,v=1.2:3.4:5.6))", "var 'nda': error: nda_t 'v' field has 3 elements/parts, but 'dims' was omitted; dims may onlybe omitted for scalars (1 element)." },
    { "bad_nda_8_ok", "", ntb_b+"nda=(dims=(v=3),v=1.2:3.4:5.6))", 0 },
    { "bad_nda_9", "", ntb_b+"nda=(dims=(v=4),v=1.2:3.4:5.6))", "var 'nda': error: nda_t 'v' field has 3 elements/parts, but user-specified 'dims' say there should be 4." },
    //{ "bad_nda_10", "", ntb_b+"nda=(tn=uint32_t))", "var 'nda': error: nda_t has neither 'v' nor 'dims' field, can't determine size; set at least one." }, // no longer an error, treated as scalar
    { "bad_nda_10", "", ntb_b+"nda=(tn=uint32_t))", 0 }, 

    // for the moment we're allowing this case:
    //{ "bad_val_t2", "", ntb+",dpf=23.1,vstr=(li_0=sdf,li_1=(li_0=biz)))", "var 'vstr': list elem 1: error: invalid attempt to use name/value list as string value. list was:(li_0=biz)"},
    { "fn_t1", "", ntb_fn+"foo.txt)", 0},
    { "fn_t2", "", ntb_fn+"%(boda_output_dir)/foo.txt)", 0},
    { "fn_t3", "", ntb_fn+"%(boda_test_dir)/foo.txt)", "var 'fn': error: unable to expand ref 'boda_test_dir' in filename, ref not found"}, // note: boda_test_dir is generally valid, but should indeed be unavailable when run in this context
    { "fn_t4", "ref_not_found", ntb_fn+"%(higgy_ma_jiggy)/foo.txt)", "var 'fn': error: unable to expand ref 'higgy_ma_jiggy' in filename, ref not found"},
    { "fn_t5", "escaped_percent", ntb_fn+"20%%_cooler.txt)", 0},
    { "fn_t6", "bad_percent", ntb_fn+"20%_cooler.txt)", "var 'fn': error: '_' after '%', expected '(' or '%'."},
    { "fn_t7", "percent_at_end", ntb_fn+"20%)", "var 'fn': error: end of input after '%', expected '(' or '%'."},
    { "fn_t8", "percent_op_at_end", ntb_fn+"20%\\()", "var 'fn': error: end of input after '%(', expected ')' to terminate ref"}, // yeah, good luck getting this error in the wild.
    { "fn_t9", "", ntb_fn+"%(av)/foo.txt,av=(li_0=foo))", "var 'fn': error: invalid attempt to use name/value list as filename ref 'av' value. list was:(li_0=foo)" },
    
  };

  extern tinfo_t tinfo_p_has_main_t;

  struct nesi_test_run_t : public test_run_t, public virtual nesi, public has_main_t // NESI(help="NESI initialization tests (error handling and correct usages).", bases=["has_main_t"], type_id="test_nesi", hide=1 )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    void test_print( void ) {
      printf( "tix=%s name=%s desc=%s\n", str(tix).c_str(), str(nesi_tests[tix].name).c_str(), str(nesi_tests[tix].desc).c_str() );
    }

    void nesi_test_run( void )
    {
      // note: global parse_lexp() function is inlined here for customization / error generation
      nesi_test_t const & lt = nesi_tests[tix];
      p_lexp_t lexp = parse_lexp( lt.in ); // nesi tests should be valid lexps (otherwise they should be a lexp test)
      bool no_error = 0;
      p_has_main_t has_main;
      try {
	lexp_name_val_map_t nvm( lexp );
	nesi_init_and_check_unused_from_nia( &nvm, &tinfo_p_has_main_t, &has_main ); 
	no_error = 1;
      } catch( rt_exception const & rte ) {
	assert_st( !no_error );
	if( !lt.err_fmt ) { // expected no error, but got one
	  test_fail_err( rte.err_msg ); 
	  // (*ofs_open("/tmp/eib.txt")) << rte.err_msg; // poor-man's auto-update
	} 
	else { 	// check if error is correct one
	  string const exp_err_msg = string(lt.err_fmt); 
	  if( rte.err_msg != exp_err_msg ) { 
	    test_fail_wrong_err( strprintf( "  '%s'\nexpected:\n  '%s'\n", 
					    str(rte.err_msg).c_str(), str(exp_err_msg).c_str() ) );
	    // (*ofs_open("/tmp/eib.txt")) << rte.err_msg; // poor-man's auto-update
	  }
	}
      }
      // (insert-file "/tmp/eib.txt")
      if( no_error ) {
	assert_st( has_main );
	if( lt.err_fmt ) { test_fail_no_err( string(lt.err_fmt) ); }
	else { 
	  // no error expected, no error occured. check that the string 
	  string const nesi_to_str( str( *has_main ) );
	  if( nesi_to_str != lt.in ) { test_fail_wrong_res( strprintf( "nesi_to_str=%s != lt.in=%s\n", 
								       str(nesi_to_str).c_str(), str(lt.in).c_str() ) );
	  }	
	}
      }
    }
    void main( nesi_init_arg_t * nia ) {
      num_fail = 0;
      for( tix = 0; tix < ( sizeof( nesi_tests ) / sizeof( nesi_test_t ) ); ++tix ) { nesi_test_run(); }
      if( num_fail ) { printf( "nesi_test num_fail=%s\n", str(num_fail).c_str() ); }
    }
  };


#include"nesi_decls.H"

  void run_system_cmd( string const &cmd, bool const verbose ) {
    timer_t t( "run_system_cmd" );
    if( verbose ) { printstr( cmd + "\n" ); }
    int const sys_ret = system( cmd.c_str() );
    assert_st( sys_ret == 0 );
  }

  // get size() of a path by iterating
  uint32_t num_elems( path const & p ) {
    uint32_t p_sz = 0; for( path::iterator pi = p.begin(); pi != p.end(); ++pi ) { ++p_sz; }
    return p_sz;
  }

  // return a new path that is p with the p_sz first elements removed. p must have at least p_sz elements.
  path strip_prefix( uint32_t const p_sz, path const & p ) {
    path rp; 
    path::iterator pi = p.begin();
    for( uint32_t pix = 0; pix != p_sz; ++pix ) { assert_st( pi != p.end() ); ++pi; } 
    for( ; pi != p.end(); ++pi ) { rp /= *pi; }
    return rp;
  }    

  void tag_dir_files( map_str_ziu32_t & tags, path const & p, uint32_t tag ) {
    assert_st( tag < 32 ); // 32 tags supported
    uint32_t const p_sz = num_elems(p);
    for( recursive_directory_iterator i(p); i != recursive_directory_iterator(); ++i ) {
      tags[strip_prefix( p_sz, i->path() ).string()].v |= (1<<tag);
    }
  }


  // this has gotta be in the c++ std lib or boost somewhere ...
  template< typename T >
  struct range {
    typedef T * iterator;
    typedef T const * const_iterator;
    T *b;
    T *e;
    T *begin( void ) { return b; }
    T *end( void ) { return e; }
    uint32_t size( void ) const { return e-b; }
    T &operator[]( size_t const & i ) { return b[i]; }
    range( T * const & b_, T * const & e_ ) : b(b_), e(e_) { }
    bool operator == ( range const & o ) const { if( size()!=o.size() ) { return 0; } return std::equal( b, e, o.b ); }
    bool operator != ( range const & o ) const = delete;
  };
  typedef range< uint8_t > range_uint8_t;
  typedef range< char const > range_char;
  typedef vector< range_char > vect_range_char;
  std::ostream & operator <<(std::ostream & os, range_char const & v) { os.write( v.b, v.size()); os.flush(); return os; }
  

  // split s at each newline. output will have (# newlines in s) + 1 elements. removes newlines.
  void getlines( vect_range_char & lines, range_char & s ) {
    char const * cur_b = s.begin();
    for( char const * c = s.begin(); c != s.end(); ++c ) {
      if( *c == '\n' ) { lines.push_back( range_char( cur_b, c ) ); cur_b = c+1; } // omit newline
    }
    lines.push_back( range_char( cur_b, s.end() ) ); // note: final elem may be empty and never has a newline
  }

  template< typename DT, typename T > void check_eq( DT & diff, std::istream & in, T const & o ) {
    T o2;
    bread( in, o2 );
    diff( o, o2 );
  }

  using std::istream;
  using std::cout;
  using std::endl;

  template< typename T >
  struct eq_diff {
    bool & has_diff;
    bool do_print;
    eq_diff( bool & has_diff_, bool const do_print_ = 1 ) : has_diff(has_diff_), do_print(do_print_) { }
    void operator ()( T const & o1, T const & o2 ) {
      if( !(o1 == o2) ) {
	has_diff = 1;
	if( do_print ) { cout << "DIFF " << typeid(o1).name() << " o1=" << o1 << " o2=" << o2 << endl; }
      }
    }
  };

  template< >
  struct eq_diff< p_nda_double_t > {
    bool & has_diff;
    eq_diff( bool & has_diff_ ) : has_diff(has_diff_) { }
    void operator ()( p_nda_double_t const & o1, p_nda_double_t const & o2 ) {
      eq_diff< dims_t > dims_eq( has_diff );
      dims_eq( o1->dims, o2->dims ); 
      if( has_diff ) { return; }
      assert_st( o1->elems_sz() == o2->elems_sz() ); // dims are same implies this
      for( uint64_t i = 0; i < o1->elems_sz(); ++i ) { if( o1->elems_ptr()[i] != o2->elems_ptr()[i] ) { has_diff = 1; } }
      if( has_diff ) {
	ssds_diff_t const ssds_diff(o1,o2);
	printf( "DIFF: nda_double_t differing elems %s", str( ssds_diff ).c_str() );
      }
      
    }
  };

  template< typename T >
  struct eq_diff< vector< T > > {
    bool & has_diff;
    eq_diff( bool & has_diff_ ) : has_diff(has_diff_) { }
    void operator ()( vect_p_nda_double_t const & o1, vect_p_nda_double_t const & o2 ) {
      if( o1.size() != o2.size() ) {
	has_diff = 1;
	cout << "DIFF vect size: o1=" << o1.size() << " o2=" << o2.size() << endl;
	return;
      }
      for( uint32_t ix = 0; ix < o1.size(); ++ix ) { 
	bool item_diff = 0;
	eq_diff< T > eq_ix( item_diff );
	eq_ix( o1[ix], o2[ix] ); 
	if( item_diff ) { has_diff = 1; }
      }
    }
  };

  template<> struct eq_diff< p_nda_digest_t > {
    bool & has_diff;
    string last_id;
    eq_diff( bool & has_diff_ ) : has_diff(has_diff_) { }
    void operator ()( p_nda_digest_t const & o1, p_nda_digest_t const & o2 ) {
      // note: we use the self_cmp_mrd/tolerance from the under-test nda_digest, *not* from the known-good digest. this
      // is perhaps arbitrary, but the general idea is that the known-good value might have some known tolerance, but in
      // general one might use a single known-good value against many under-test values with different tolerenances, so
      // that's where the tolerance must come from.
      string const mrd_diff = o1->mrd_comp( o2, o2->self_cmp_mrd );
      if( !mrd_diff.empty() ) {
        has_diff = 1;
        printstr( "DIFF: " + last_id + " " + mrd_diff );
      }
    }
  };

  // note: in test usages, i1 is the known-good stream, i2 is the under-test one
  bool diff_boda_streams( istream & i1, istream & i2 ) {
    uint32_t maybe_boda_magic = 0;
    bread( i1, maybe_boda_magic );
    assert_st( maybe_boda_magic == boda_magic );
    maybe_boda_magic = 0;
    bread( i2, maybe_boda_magic );
    assert_st( maybe_boda_magic == boda_magic );

    bool stream_diff = 0;
    bool data_diff = 0;
    eq_diff<string> eq_str( stream_diff );
    eq_diff<p_nda_double_t> eq_p_nda_double_t( data_diff );
    eq_diff<vect_p_nda_double_t> eq_vect_p_nda_double_t( data_diff );
    eq_diff<p_nda_digest_t> eq_p_nda_digest_t( data_diff );

    string id;
    string cmd;
    while( 1 ) {
      bread( i1, cmd );
      check_eq( eq_str, i2, cmd );
      //printf( "cmd=%s\n", str(cmd).c_str() );
      if( stream_diff ) { break; }
      if( cmd == "END_BODA" ) { break; }
      else if( cmd == "id" ) {
	bread( i1, id );
	check_eq( eq_str, i2, id );
	if( stream_diff ) { break; }
        eq_p_nda_digest_t.last_id = id; // FIXME: a hack, but expedient/practical/nice?
      }
      else if( cmd == "p_nda_double_t" ) {
	p_nda_double_t o;
	bread( i1, o );
	check_eq( eq_p_nda_double_t, i2, o );
      }
      else if( cmd == "vect_p_nda_double_t" ) {
	vect_p_nda_double_t o;
	bread( i1, o );
	check_eq( eq_vect_p_nda_double_t, i2, o );
      }
      else if( cmd == "p_nda_digest_t" ) {
	p_nda_digest_t o;
	bread( i1, o );
	check_eq( eq_p_nda_digest_t, i2, o );
      } else {
        rt_err( strprintf( "unknown cmd '%s' in boda stream (corrupt/invalid stream?)", str(cmd).c_str() ) );
      }
    }
    if( stream_diff ) { printf("DIFF: boda stream differs. last seen: cmd=%s id=%s\n", cmd.c_str(), id.c_str() ); }
    if( data_diff ) { printf("DIFF: boda data in stream differed.\n"); }
    bool const has_diff = stream_diff || data_diff;
    return has_diff;
  }


  // returns 1 if files differ
  bool diff_file( path const & good, path const & test, string const & fn ) {
    string const good_fn = (good / fn).string();
    string const test_fn = (test / fn).string();
    assert_st( exists( good_fn ) && exists( test_fn ) );
    if( is_directory( good_fn ) != is_directory( test_fn ) ) {
      printf( "DIFF: directory / non-directory mismatch for file '%s'.", fn.c_str() );
      return 1;
    }
    if( is_directory( good_fn ) ) { return 0; } // both are directories, so that's all fine and well.
    // we can only handle regular files and directories, so check for that:
    assert_st( is_regular_file( good_fn ) && is_regular_file( test_fn ) ); 

    // since map_file_ro chokes on empty files, we handle them manually ...
    bool const good_empty = file_size( good_fn ) == 0;
    bool const test_empty = file_size( test_fn ) == 0;
    if( good_empty || test_empty ) {
      if( good_empty == test_empty ) { return 0; } // both files empty
      printf( "DIFF: EMPTY/NONEMPTY mismatch: good_empty=%s test_empty=%s\n", 
	      str(good_empty).c_str(), str(test_empty).c_str() );
      return 1;
    }

    p_mapped_file_source good_map = map_file_ro( good_fn );
    p_mapped_file_source test_map = map_file_ro( test_fn );
    range_char good_range( good_map->data(), good_map->data() + good_map->size() );
    range_char test_range( test_map->data(), test_map->data() + test_map->size() );
    if( endswith(fn, ".txt" ) || endswith(fn, ".oct") ) { // do line-by-line diff
      vect_range_char good_lines; getlines( good_lines, good_range );
      vect_range_char test_lines; getlines( test_lines, test_range );
      if( endswith(fn,".oct") ) { // for octave files, remove/ignore the first line, as it has a timestamp
	if( good_lines.size() && test_lines.size() ) { // do nothing if both files don't have at least one line
	  good_lines.erase( good_lines.begin() ); 
	  test_lines.erase( test_lines.begin() ); 
	}
      }
      dtl::Diff< range_char, vect_range_char > d( good_lines, test_lines );
      d.compose();
      if( d.getEditDistance() ) {
	printf( "DIFF: text file '%s' edit distance:%s\n", fn.c_str(), str(d.getEditDistance()).c_str() );
	//d.printSES();
	d.composeUnifiedHunks(); 
	d.printUnifiedFormat();
	return 1;
      } else { 
	assert_st( good_lines.size() == test_lines.size() );
	return 0; 
      }
    } else if( endswith( fn, ".png" ) || endswith( fn, ".jpg" ) ) { // image diff
      if( !( good_range == test_range ) ) { // if not binary identical
	img_t good_img;
	img_t test_img;
	good_img.load_fn( good_fn );
	test_img.load_fn( test_fn );

	dtl::Diff< uint8_t, range_uint8_t > d( 
	  range_uint8_t( good_img.pels.get(), good_img.pels.get()+good_img.sz_raw_bytes() ),
	  range_uint8_t( test_img.pels.get(), test_img.pels.get()+test_img.sz_raw_bytes() ) );
	d.compose();
	printf( "DIFF: image file '%s' edit distance (padded raw color bytes, inexact):%s\n", 
		fn.c_str(), str(d.getEditDistance()).c_str() );
	return 1;
      }
      else { return 0; }
    } else if( endswith( fn, ".boda" ) ) { // BODA stream diff
      mfs_istream good_is;
      good_is.open( *good_map );
      mfs_istream test_is;
      test_is.open( *test_map );
      return diff_boda_streams( good_is, test_is );
    } else { // bytewise binary diff
      dtl::Diff< char, range_char > d( good_range, test_range );
      d.compose();
      if( d.getEditDistance() ) {
	printf( "DIFF: binary file '%s' edit distance:%s\n", fn.c_str(), str(d.getEditDistance()).c_str() );
	return 1;
      }
      else { return 0; }
    }    
  }




  void maybe_remove_dir( path const & dir ) {
    if( exists( dir ) ) {
      assert_st( is_directory( dir ) );
      uint32_t const num_rem = remove_all( dir );
      assert( num_rem );
    }
  }

  void gen_test_compute_tests( p_ostream & out ); // from test_compute.cc
  void gen_ops_prof_tests( p_ostream & out, bool const & run_slow ); // from test_compute.cc

  extern tinfo_t tinfo_p_cmd_test_t;
  struct test_cmds_t : public test_run_t, public virtual nesi, public has_main_t // NESI(help="test of modes in various configurations", bases=["has_main_t"], type_id="test_cmds" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t xml_fn; //NESI(default="%(boda_test_dir)/test_cmds.xml",help="xml file containing list of tests.")
    string filt; //NESI(default=".*",help="regexp over test name of what tests to run (default runs all tests)")
    uint32_t run_slow; //NESI(default=0,help="run slow tests too?")
    uint32_t no_diff; //NESI(default=0,help="disable diff?")
    uint32_t verbose; //NESI(default=0,help="if true, print each test lexp before running it")
    uint32_t update_failing; //NESI(default=0,help="if true, update archives for all run tests that fail.")
    
    filename_t boda_test_dir; //NESI(help="boda base test dir (generally set via boda_cfg.xml)",req=1)

    p_cmd_test_t cur_test;
    void test_print( void ) {
      printf( "tix=%s %s\n", str(tix).c_str(), str(*cur_test).c_str() );
    }

    // returns 1 if command ran with no error and no error was expected (and thus diff should be run)
    bool run_command( p_cmd_test_t cmd, nesi_init_arg_t * nia ) {
      bool do_diff = 0;
      bool write_err = 1;
      bool no_error = 0;
      try {
	cmd->command->base_setup(); // note: sets command->boda_output_dir *and* creates it
	cmd->command->main( nia );	
	cmd->command.reset(); // destory command. might matter if it's holding open output files or the like
	no_error = 1;
      } catch( rt_exception const & rte ) {
	assert_st( !no_error );
	if( !cmd->err ) { // expected no error, but got one
	  test_fail_err( rte.err_msg ); 
	  if( write_err ) { (*ofs_open("/tmp/eib.txt")) << rte.err_msg; } // poor-man's auto-update 
	} 
	else { 	// check if error is correct one
	  string const exp_err_msg = *cmd->err;
	  if( rte.err_msg != exp_err_msg ) { 
	    test_fail_wrong_err( strprintf( "  '%s'\nexpected:\n  '%s'\n", 
					    str(rte.err_msg).c_str(), str(exp_err_msg).c_str() ) );
	    if( write_err ) { (*ofs_open("/tmp/eib.txt")) << rte.err_msg; } // poor-man's auto-update
	  }
	}
      }
      // (insert-file "/tmp/eib.txt")
      if( no_error ) {
	if( cmd->err ) { test_fail_no_err( *cmd->err ); }
	else { do_diff = 1; }
      }
      return do_diff;
    }

    void diff_command( p_cmd_test_t cmd, path const & test_out_dir, path const & gen_test_out_dir ) {
      timer_t t( "diff_command" );
      path good_dir = path(boda_test_dir.exp) / "good_tr";
      ensure_is_dir( good_dir.string(), 1 );
      // note: test_out_dir should equivalent to gen_test_out_dir (but not ==). we check that:
      // path const test_out_dir( cmd->command->boda_output_dir.exp ); // test_out_dir is cached from this
      assert_st( equivalent( test_out_dir, gen_test_out_dir ) );
      if( !exists( test_out_dir ) ) { // test must create its output dir
	rt_err( strprintf( "test '%s' did not create its expected output directory '%s'.", 
			   cmd->test_name.c_str(), test_out_dir.c_str() ) );
      }

      path const test_good_dir = good_dir / cmd->test_name;
      bool update_tr = 0;
      bool tgd_exists = 0;
      if( !exists( test_good_dir ) ) {
	printf("NEW_TEST: no existing good results archive for test %s, will generate\n",cmd->test_name.c_str());
	update_tr = 1;
      } else { 
	assert_st( is_directory( test_good_dir ) );
	tgd_exists = 1; // in case we need to update, we'll need to remove first
	// compare good and test directories
	bool output_missing = 0;
        bool output_new_or_diff = 0;
	map_str_ziu32_t tags;
	tag_dir_files( tags, test_good_dir, 0 );
	tag_dir_files( tags, test_out_dir, 1 );
	tags.erase(".keep"); // ingore .keep file's presense or absence anywhere
	for( map_str_ziu32_t::const_iterator i = tags.begin(); i != tags.end(); ++i ) {
	  uint32_t const & tv = i->second.v;
	  if( tv == 1 ) { printf( "DIFF: file '%s' only in known-good output dir.\n", str(i->first).c_str()); 
	    output_missing = 1; continue; 
	  }
	  if( tv == 2 ) { printf( "DIFF: file '%s' only in under-test output dir.\n", str(i->first).c_str()); 
	    output_new_or_diff = 1; continue; 
	  }
	  assert_st( tv == 3 ); // file in both known-good an under-test output dirs
	  if( diff_file( test_good_dir, test_out_dir, i->first ) ) { 
	    printf( "DIFF: file '%s' differs between known-good(-) and under-test(+):\n", str(i->first).c_str()); 
	    output_new_or_diff = 1; continue; 
	  }
	}
	if( output_missing || output_new_or_diff ) {
	  if( update_failing ) { 
	    printf("UPDATE-FAILING: test %s failed, will update.\n",cmd->test_name.c_str());
	    update_tr = 1; 
	  } else {
            if( output_new_or_diff || (output_missing && (!cmd->ignore_missing_outputs) ) ) {
              printf("FAIL: test %s failed.\n",cmd->test_name.c_str());
              ++num_fail;
            } else {
              assert_st( output_missing && cmd->ignore_missing_outputs );
              printf("WARNING: test %s missing some outputs, but ignore_missing_outputs=1: no failure.\n",cmd->test_name.c_str());
            }
	  }
	}
      }	  
      if( update_tr ) {
	printf("UPDATING good results dir for test %s.\n",cmd->test_name.c_str());
	if( tgd_exists ) {
	  uint32_t const num_rem = remove_all( test_good_dir );
	  assert_st( num_rem );
	}
	assert_st( !exists( test_good_dir ) );
	run_system_cmd( strprintf("cp -a %s %s", test_out_dir.c_str(),test_good_dir.c_str()), 0 );
	// since some VCSs :( git ): don't track empty dirs, make the dir non-empty with a file named .keep
	path const keep_fn = test_good_dir / ".keep";
	assert_st( !exists( keep_fn ) );
	ofs_open( keep_fn.string() );
      }
    }

    virtual void main( nesi_init_arg_t * nia ) {
      num_fail = 0;
      uint64_t num_skipped = 0;
      set_string missing_needed_features;
      set_string seen_test_names;
      set_string failed_modes;
      seen_test_names.insert( "good" ); // reserved sub-dir to hold known good results
      regex filt_regex( filt );
      // note: cmd tests should not fail nesi init (otherwise they should be nesi init tests).
      pugi::xml_document doc;
      // HACK/FIXME: if this special filename is used (re-)generate the contents (in the current dir)
      if( xml_fn.exp == "gen_test_compute_tests.xml" ) { p_ostream out = ofs_open( xml_fn.exp ); gen_test_compute_tests( out ); }
      if( xml_fn.exp == "gen_ops_prof_tests.xml" ) { p_ostream out = ofs_open( xml_fn.exp ); gen_ops_prof_tests( out, run_slow ); }
      pugi::xml_node xn = xml_file_get_root( doc, xml_fn.exp );
      tix = 0;
      for( pugi::xml_node xn_i: xn.children() ) { 
	lexp_name_val_map_t nvm( parse_lexp_list_xml( xn_i, xml_fn.exp ), nia );
	nvm.insert_leaf( "boda_output_dir", "%(test_name)", 1 ); // unused is okay --> inc_use_cnt = 1
	p_cmd_test_t cmd_test;

	bool test_init_failed = 0;
	try { nesi_init_and_check_unused_from_nia( &nvm, &tinfo_p_cmd_test_t, &cmd_test ); }
	catch( rt_exception const & rte ) { 
	  test_init_failed = 1; 
	  p_lexp_t mode = nvm.l->get_kid_by_name("command")->get_kid_by_name("mode");
	  assert_st( mode->leaf_val.exists() );
	  string const & mode_str = mode->leaf_val.str();
	  failed_modes.insert( mode_str );
	}
	if( test_init_failed ) { continue; }
	uint32_t num_spec = bool(cmd_test->command) + bool(cmd_test->cli_str);
	if( num_spec != 1 ) {
	  rt_err( strprintf( "internal test_cmds error: %s of command and cli_str specified for test %s. specify exactly one.",
			     str(num_spec).c_str(), cmd_test->test_name.c_str() ) );
	} 

        bool missing_needed_feature = 0;
        if( cmd_test->needs ) {
          vect_string needs_parts = split( *cmd_test->needs, ',' );
          for( vect_string::const_iterator i = needs_parts.begin(); i != needs_parts.end(); ++i ) {
            if( !is_feature_enabled( i->c_str() ) ) { missing_needed_feature = 1; missing_needed_features.insert( *i ); }
          }
        }
        if( missing_needed_feature ) { ++num_skipped; continue; }

	if( !cmd_test->command ) {
	  assert_st( cmd_test->cli_str );
	  try { 
	    vect_string cli_str_argv = boost::program_options::split_unix( *cmd_test->cli_str );
	    lexp_name_val_map_t nvm_cli( get_lexp_from_argv( cli_str_argv, std::cout ), &nvm );
	    nesi_init_and_check_unused_from_nia( &nvm_cli, &tinfo_p_has_main_t, &cmd_test->command ); 
	  }
	  catch( rt_exception const & rte ) { 
	    test_init_failed = 1; 
	    failed_modes.insert( "(mode-exists-but-cli_str-proc-failed-for:"+*cmd_test->cli_str+")" );
	  }
	}
	if( test_init_failed ) { continue; }
	assert_st( cmd_test->command );
                
	cur_test = cmd_test; // needed by test_print()
	bool const seen_test_name = !seen_test_names.insert( cmd_test->test_name ).second;
	if( seen_test_name ) { rt_err( "duplicate or reserved (e.g. 'good') test name:" + cmd_test->test_name ); }
	if( regex_search( cmd_test->test_name, filt_regex ) && 
	    ( run_slow || (!cmd_test->slow) ) ) {
	  if( verbose ) { std::cout << (*cmd_test) << std::endl; }
	  timer_t t("test_cmds_cmd");
	  // note: no test may be named 'good'
	  path gen_test_out_dir = path(boda_output_dir.exp) / cmd_test->test_name;
	  maybe_remove_dir( gen_test_out_dir );
	  // FIXME: it is non-trival to construct the proper nia here ...
	  // command will be reset() in run_command, so cache this here.
	  path const test_out_dir( cmd_test->command->boda_output_dir.exp ); 
	  bool const do_diff = run_command( cmd_test, nia);
	  if( do_diff ) { 
	    if( no_diff ) { printf("NO_DIFF: not checking results of test %s.\n",cmd_test->test_name.c_str()); }
	    else { diff_command( cmd_test, test_out_dir, gen_test_out_dir ); }
	  }
	}
	++tix;
      }
      if( num_fail ) { printf( "test_cmds num_fail=%s\n", str(num_fail).c_str() ); }
      if( num_skipped ) {
        assert_st( !missing_needed_features.empty() );
        printf( "WARNING: skipped some tests due to missing features: num_skipped=%s missing_needed_features=%s\n", str(num_skipped).c_str(), str(missing_needed_features).c_str() );
      }
      if( !failed_modes.empty() ) {
	printf( "WARNING: test_cmds: some modes had test commands that failed to initialize. perhaps these modes aren't enabled?\n");
	printf( "  FAILING MODES:" );
	for( set_string::const_iterator i = failed_modes.begin(); i != failed_modes.end(); ++i ) {
	  printstr( " " + *i );
	}
	printf( "\n" );
      }
    }
  };

#include"gen/test_nesi.cc.nesi_gen.cc"
}

