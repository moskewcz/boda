#include"boda_tu_base.H"
#include"nesi.H"
#include"test_base.H"
#include"has_main.H"
#include"timers.H"
#include<boost/regex.hpp>
#include"xml_util.H"
#include"lexp.H"

namespace boda 
{
  using boost::regex;
  using boost::regex_search;

  extern tinfo_t tinfo_p_has_main_t;

  struct test_all_t : public virtual nesi, public has_main_t // NESI(help="run all tests in (by default) %(boda_test_dir)/test_all.xml", bases=["has_main_t"], type_id="test_all" )
  {
    filename_t xml_fn; //NESI(default="%(boda_test_dir)/test_all.xml",help="xml file containing list of testing modes.")
    string filt; //NESI(default=".*",help="regexp over modes of what commands to run (default runs all)")
    uint32_t verbose; //NESI(default=0,help="if true, print each test mode name before running it")

    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    virtual void main( nesi_init_arg_t * nia ) {
      regex filt_regex( filt );
      pugi::xml_document doc;
      pugi::xml_node xn = xml_file_get_root( doc, xml_fn.exp );
      for( pugi::xml_node xn_i: xn.children() ) { 
	lexp_name_val_map_t nvm( parse_lexp_list_xml( xn_i ), nia );
	p_has_main_t test_mode;
	nesi_init_and_check_unused_from_nia( &nvm, &tinfo_p_has_main_t, &test_mode ); 
	if( regex_search( test_mode->mode, filt_regex ) ) {
	  timer_t t("test_all_mode");
	  if( verbose ) { 
	    printf( "--- running %s ---\n", str(test_mode->mode).c_str() );
	  }
	  test_mode->base_setup();
	  test_mode->main( &nvm );
	}	
      }
    }
  };
#include"gen/test_all.cc.nesi_gen.cc"
}
