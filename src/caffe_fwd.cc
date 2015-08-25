// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"has_conv_fwd.H"
#include"timers.H"
#include"conv_util.H"

namespace boda 
{

  struct caffe_fwd_t : virtual public nesi, public has_conv_fwd_t // NESI(help="compute conv pipe forward using caffe",
			   // bases=["has_conv_fwd_t"], type_id="caffe" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    p_conv_pipe_t cp;
    uint32_t num_imgs;

    virtual void init( p_conv_pipe_t const & cp_, uint32_t const & num_imgs_ );
    virtual void run_fwd( p_map_str_p_nda_float_t const & fwd );
  };

  void caffe_fwd_t::init( p_conv_pipe_t const & cp_, uint32_t const & num_imgs_ ) {
    num_imgs = num_imgs_;
    assert_st( num_imgs );
    cp = cp_;
    assert_st( cp );
    assert_st( cp->finalized );
  }

  void caffe_fwd_t::run_fwd( p_map_str_p_nda_float_t const & fwd ) {
    timer_t t("caffe_fwd_t::run_fwd");
    assert_st( cp->finalized );

  }  
#include"gen/caffe_fwd.cc.nesi_gen.cc"
}
