// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"lexp.H"
#include"conv_util.H"
#include"blf_pack.H"
#include"img_io.H"
#include"disp_util.H"

#include "caffeif.H"
#include <glog/logging.h>
#include <google/protobuf/text_format.h>

namespace boda 
{
  using namespace boost;


  p_conv_pipe_t make_p_conv_pipe_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );
  p_img_pyra_pack_t make_p_img_pyra_pack_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );
  
  struct conv_pyra_t : virtual public nesi, public has_main_t // NESI(help="conv_ana / blf_pack integration test",
		       // bases=["has_main_t"], type_id="conv_pyra")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    p_run_cnet_t run_cnet; //NESI(default="(ptt_fn=%(boda_test_dir)/conv_pyra_imagenet_deploy.prototxt)",help="cnet running options")

    filename_t pipe_fn; //NESI(default="%(boda_test_dir)/conv_pyra_pipe.xml",help="input pipe XML filename")

    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="output filename.")
    
    filename_t img_in_fn; //NESI(default="%(boda_test_dir)/pascal/000001.jpg",help="input image filename")
    filename_t img_out_fn; // NESI(default="%(boda_output_dir)/out_%%s.png", help="format for filenames of"
                           //   " output image bin files. %%s will replaced with the bin index.")

    string out_layer_name;//NESI(default="conv5",help="output layer name of which to output top blob of")

    uint32_t write_output; //NESI(default=0,help="if true, write output images/bins (slow)")
    uint32_t disp_output; //NESI(default=0,help="if true, display output images/bins")

    p_img_pyra_pack_t ipp; //NESI(default="()",help="pyramid packing options")

    virtual void main( nesi_init_arg_t * nia ) { 
      timer_t t("conv_prya_top");

      p_conv_pipe_t conv_pipe = make_p_conv_pipe_t_init_and_check_unused_from_lexp( parse_lexp_xml_file( pipe_fn.exp ), 0 );
      conv_pipe->calc_support_info();

      p_img_t img_in( new img_t );
      img_in->load_fn( img_in_fn.exp );
      //p_img_pyra_pack_t ipp = make_p_img_pyra_pack_t_init_and_check_unused_from_lexp( parse_lexp("(mode=img_pyra_pack,img_in_fn=fixme.remove)"), nia );

      ipp->in_sz.d[0] = img_in->w; ipp->in_sz.d[1] = img_in->h;
      ipp->do_place_imgs( conv_pipe->conv_sis.back(), img_in );

      run_cnet->in_sz = ipp->bin_sz;
      run_cnet->in_num_imgs = ipp->bin_imgs.size();
      run_cnet->out_layer_name = out_layer_name; // FIXME: too error prone? automate / check / inherit?
      run_cnet->setup_cnet();

      {
	timer_t t("conv_prya_copy_bins_in");
	// copy images to batch
	for( uint32_t bix = 0; bix != run_cnet->in_num_imgs; ++bix ) {
	  subtract_mean_and_copy_img_to_batch( run_cnet->in_batch, bix, ipp->bin_imgs[bix] );
	}
      }
      p_nda_float_t out_batch = run_cnet->run_one_blob_in_one_blob_out();
      dims_t const & obd = out_batch->dims;
      assert( obd.sz() == 4 );
      assert( obd.dims(0) == run_cnet->in_batch->dims.dims(0) );

      uint32_t sqrt_out_chan = uint32_t( ceil( sqrt( double( obd.dims(1) ) ) ) );
      assert( sqrt_out_chan );
      assert( (sqrt_out_chan*sqrt_out_chan) >= obd.dims(1) );

      if( write_output || disp_output ) {
	timer_t t("conv_pyra_write_output");
	float const out_min = nda_reduce( *out_batch, min_functor<float>(), 0.0f ); // note clamp to 0
	//assert_st( out_min == 0.0f ); // shouldn't be any negative values
	float const out_max = nda_reduce( *out_batch, max_functor<float>(), 0.0f ); // note clamp to 0
	float const out_rng = out_max - out_min;
	vect_p_img_t out_imgs;
	for( uint32_t bix = 0; bix != obd.dims(0); ++bix ) {
	  p_img_t out_img( new img_t );
	  out_img->set_sz_and_alloc_pels( obd.dims(3)*sqrt_out_chan, obd.dims(2)*sqrt_out_chan ); // w, h
	  for( uint32_t y = 0; y < out_img->h; ++y ) {
	    for( uint32_t x = 0; x < out_img->w; ++x ) {
	      uint32_t const bx = x / sqrt_out_chan;
	      uint32_t const by = y / sqrt_out_chan;
	      uint32_t const bc = (y%sqrt_out_chan)*sqrt_out_chan + (x%sqrt_out_chan);
	      uint32_t gv;
	      if( bc < obd.dims(1) ) {
		float const norm_val = ((out_batch->at4(bix,bc,by,bx)-out_min) / out_rng );
		//gv = grey_to_pel( uint8_t( std::min( 255.0, 255.0 * norm_val ) ) );
		gv = grey_to_pel( uint8_t( std::min( 255.0, 255.0 * (log(.01) - log(std::max(.01f,norm_val))) / (-log(.01)) )));
	      } else { gv = grey_to_pel( 0 ); }
	      out_img->set_pel( x, y, gv );
	    }
	  }
	  if( write_output ) {
	    filename_t ofn = filename_t_printf( img_out_fn, str(bix).c_str() );
	    out_img->save_fn_png( ofn.exp, 1 );
	  }
	  if( disp_output ) { out_imgs.push_back( out_img ); }
	}
	if( disp_output ) { 
	  disp_win_t disp_win;
	  disp_win.disp_skel( out_imgs, 0 ); 
	}
      }
    }
  };  

#include"gen/conv_pyra.cc.nesi_gen.cc"
}
