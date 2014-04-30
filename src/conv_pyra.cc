// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"lexp.H"

namespace boda 
{
  using namespace boost;

  struct conv_pyra_t : virtual public nesi, public has_main_t // NESI(help="conv_ana / blf_pack integration test",bases=["has_main_t"], type_id="conv_pyra")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    filename_t pipe_fn; //NESI(default="%(boda_test_dir)/conv_pyra_pipe.xml",help="input pipe XML filename")
    filename_t ptt_fn; //NESI(default="%(boda_test_dir)/conv_pyra_net.prototxt",help="input net prototxt template filename")
    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="output filename.")

    virtual void main( nesi_init_arg_t * nia ) { 
      p_string ptt_str = read_whole_fn( ptt_fn );
      string out_pt_str;
      str_format_from_nvm_str( out_pt_str, *ptt_str, 
			       strprintf( "(xsize=%s,ysize=%s)", str(100).c_str(), str(105).c_str() ) );
      (*ofs_open( out_fn )) << out_pt_str;
    }
  };

#include"gen/conv_pyra.cc.nesi_gen.cc"

}
