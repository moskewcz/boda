// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"lexp.H"
#include"nesi.H"
#include"conv_util.H"
#include"blf_pack.H"
#include"img_io.H"
#include"disp_util.H"
#include"cap_util.H"

#include "caffeif.H"
#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include"asio_util.H"

namespace boda 
{
  struct conv_pyra_t : virtual public nesi, public has_main_t // NESI(help="conv_ana / blf_pack integration test",
		       // bases=["has_main_t"], type_id="conv_pyra" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    p_run_cnet_t run_cnet; //NESI(default="(ptt_fn=%(boda_test_dir)/conv_pyra_imagenet_deploy.prototxt)",help="cnet running options")
    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="output filename.")
    
    //filename_t img_in_fn; //xNESI(default="%(boda_test_dir)/pascal/000001.jpg",help="input image filename")
    filename_t img_out_fn; // NESI(default="%(boda_output_dir)/out_%%s.png", help="format for filenames of"
                           //   " output image bin files. %%s will replaced with the bin index.")
    string out_layer_name;//NESI(default="conv5",help="output layer name of which to output top blob of")
    uint32_t write_output; //NESI(default=0,help="if true, write output images/bins (slow)")
    uint32_t disp_output; //NESI(default=1,help="if true, display output images/bins")
    p_img_pyra_pack_t ipp; //NESI(default="()",help="pyramid packing options")
    p_capture_t capture; //NESI(default="()",help="capture from camera options")
    
    p_img_t feat_img; 
    p_asio_fd_t cap_afd;
    disp_win_t disp_win;

    void on_cap_read( error_code const & ec ) { 
      if( ec == errc::operation_canceled ) { return; }
      assert_st( !ec );
      capture->on_readable( 1 );
      ipp->scale_and_pack_img_into_bins( capture->cap_img );
      for( uint32_t bix = 0; bix != ipp->bin_imgs.size(); ++bix ) {
	subtract_mean_and_copy_img_to_batch( run_cnet->in_batch, 0, ipp->bin_imgs[bix] );
	p_nda_float_t out_batch = run_cnet->run_one_blob_in_one_blob_out();
	if( write_output || disp_output ) {
	  timer_t t("conv_pyra_write_output");
	  copy_batch_to_img( out_batch, 0, feat_img );
	  disp_win.update_disp_imgs();
	}
      }
      setup_capture_on_read( *cap_afd, &conv_pyra_t::on_cap_read, this );
    }

    void on_quit( error_code const & ec ) { cap_afd->cancel(); }
   
    virtual void main( nesi_init_arg_t * nia ) { 
      timer_t t("conv_prya_top");
      //p_img_t img_in( new img_t );
      //img_in->load_fn( img_in_fn.exp );
      //u32_pt_t const img_in_sz( img_in->w, img_in->h );
      u32_pt_t const img_in_sz( capture->cap_res );
      ipp->in_sz = img_in_sz;
      run_cnet->in_sz = ipp->bin_sz;
      run_cnet->in_num_imgs = 1;
      run_cnet->out_layer_name = out_layer_name; // FIXME: too error prone? automate / check / inherit?
      run_cnet->setup_cnet();

      p_conv_pipe_t conv_pipe = run_cnet->get_pipe();
      ipp->do_place_imgs( conv_pipe->conv_sis.back() );

      feat_img.reset( new img_t );
      u32_pt_t const feat_img_sz = run_cnet->get_one_blob_img_out_sz();
      feat_img->set_sz_and_alloc_pels( feat_img_sz.d[0], feat_img_sz.d[1] ); // w, h
      capture->cap_start();
      disp_win.disp_setup( vect_p_img_t{feat_img,capture->cap_img} );

      io_service_t & io = get_io( &disp_win );
      cap_afd.reset( new asio_fd_t( io, ::dup(capture->get_fd() ) ) );
      setup_capture_on_read( *cap_afd, &conv_pyra_t::on_cap_read, this );
      register_quit_handler( disp_win, &conv_pyra_t::on_quit, this );
      io.run();
    }
  };  
#include"gen/conv_pyra.cc.nesi_gen.cc"
}
