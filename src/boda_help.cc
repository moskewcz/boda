// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"nesi.H"
#include"str_util.H"
#include"lexp.H"
#include"has_main.H"

// TODO: move parts of boda.cc here ...

namespace boda
{
#include"nesi_decls.H"

  extern tinfo_t tinfo_has_main_t;
  extern cinfo_t cinfo_has_main_t;

  void nesi_struct_derived_modes( cinfo_t const * const ci, bool const show_all ) {
    if( (!show_all) && ci->hide ) { return; } // skip if class is hidden. note: will ignore any derived classes as well.
    if( ci->tid_str ) { printf( "%s ", ci->tid_str ); }
    for( cinfo_t const * const * dci = ci->derived; *dci; ++dci ) { nesi_struct_derived_modes( *dci, show_all ); }
  }

  struct compsup_t : virtual public nesi, public has_main_t // NESI(help="completion support mode",
			  // bases=["has_main_t"], type_id="compsup")
  {
    uint32_t show_all; //NESI(default=0,help="if true, show hidden modes")
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    virtual void main( nesi_init_arg_t * nia ) { nesi_struct_derived_modes( &cinfo_has_main_t, show_all ); printf("\n"); }
  };
#include"gen/boda_help.cc.nesi_gen.cc"
}
