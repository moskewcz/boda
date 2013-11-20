#include"boda_tu_base.H"
#include"has_main.H"
#include"pyif.H"
#include"str_util.H"
#include"xml_util.H"
#include"lexp.H"

namespace boda 
{
  using pugi::xml_node;
  using pugi::xml_document;

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

    virtual void main( void ) {
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
    string name; //NESI(help="name of test",req=1)
    string input; //NESI(help="input",req=1)

  };

  string tp_if_rel( string const & fn ) {
    assert_st( !fn.empty() );
    if( fn[0] == '/' ) { return fn; }
    return py_boda_test_dir() + "/" + fn;
  }

  struct test_modes_t : public virtual nesi, public has_main_t // NESI(help="test of modes in various configurations", bases=["has_main_t"], type_id="test_modes" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string xml_fn; //NESI(default="modes_tests.xml",help="xml file containing list of tests. relative paths will be prefixed with the boda test dir.")
    string filt; //NESI(default=".*",help="regexp over test name of what tests to run (default runs all tests)")
    uint32_t verbose; //NESI(default=0,help="if true, print each test lexp before running it")
    virtual void main( void ) {
      string const full_xml_fn = tp_if_rel(xml_fn);
      xml_document doc;
      xml_node xn = xml_file_get_root( doc, full_xml_fn );
      for( xml_node xn_i: xn.children() ) { // child elements become list values
	if( 1 ) { // FIXME: should be filt.search( xn_i.name() )
	  p_lexp_t test_lexp = parse_lexp_list_xml( xn_i );
	  if( verbose ) { printf( "*test_lexp=%s\n", str(*test_lexp).c_str() ); }
	  create_and_run_has_main_t( test_lexp );
	}
      }
    }
  };



#include"gen/test_nesi.cc.nesi_gen.cc"
}

