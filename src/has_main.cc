#include"boda_tu_base.H"
#include"has_main.H"
#include"lexp.H"
#include"nesi.H"
namespace boda
{
  extern tinfo_t tinfo_p_has_main_t;

  void create_and_run_has_main_t( p_lexp_t lexp ) {
    p_has_main_t has_main;
    nesi_init_and_check_unused_from_lexp( &tinfo_p_has_main_t, &has_main, lexp );
    has_main->main();
  }
#include"gen/has_main.H.nesi_gen.cc"
}
