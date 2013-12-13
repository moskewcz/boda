#include"boda_tu_base.H"
#include"nesi.H"
#include"test_base.H"
#include"has_main.H"
#include<boost/regex.hpp>

namespace boda 
{
  using boost::regex;
  using boost::regex_search;

  extern tinfo_t tinfo_vect_p_has_main_t;

  struct test_all_t : public virtual nesi, public has_main_t // NESI(help="run all tests in (by default) test/test_all.xml", bases=["has_main_t"], type_id="test_all", hide=1 )
  {
    filename_t xml_fn; //NESI(default="%(boda_test_dir)/test_all.xml",help="xml file containing list of testing modes.")
    vect_p_has_main_t test_cmds; //NESI(help="test commands to run. contents of xml_fn will be appended.")
    string filt; //NESI(default=".*",help="regexp over modes of what commands to run (default runs all)")

    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    virtual void main( nesi_init_arg_t * nia ) {
      regex filt_regex( filt );
      nesi_init_and_check_unused_from_xml_fn( nia, &tinfo_vect_p_has_main_t, &test_cmds, xml_fn.exp );
      for( vect_p_has_main_t::const_iterator i = test_cmds.begin(); i != test_cmds.end(); ++i ) {
	if( regex_search( (*i)->mode, filt_regex ) ) {
	  (*i)->base_setup();
	  (*i)->main( nia );
	}	
      }
    }
  };
#include"gen/test_all.cc.nesi_gen.cc"
}
