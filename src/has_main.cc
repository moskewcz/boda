#include"boda_tu_base.H"
#include"has_main.H"
#include"lexp.H"

namespace boda
{
#include"nesi_decls.H"

  extern tinfo_t tinfo_has_main_t;
  extern cinfo_t cinfo_has_main_t;

  void create_and_run_has_main_t( p_lexp_t lexp ) {
    p_has_main_t has_main;
    void * pv = nesi_struct_make_p( &tinfo_has_main_t, &has_main, lexp.get() );
    nesi_struct_init( &tinfo_has_main_t, pv, lexp.get() );
    // check for unused fields in l
    vect_string path;
    lexp_check_unused( lexp.get(), path );
    //printf( "*has_main=%s\n", str(*has_main).c_str() );
    //nesi_dump_xml( std::cout, *has_main, "root" );
    has_main->main();
  }
#include"gen/has_main.H.nesi_gen.cc"
}
