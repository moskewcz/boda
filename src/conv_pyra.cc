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
#include"anno_util.H"

namespace boda 
{

  struct conv_pyra_t : virtual public nesi, public has_main_t // NESI(help="conv_ana / blf_pack integration test",
		       // bases=["has_main_t"], type_id="conv_pyra" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    p_cnet_predict_t cnet_predict; //NESI(default="(ptt_fn=%(models_dir)/nin_imagenet_nopad/deploy.prototxt)",help="cnet running options")
    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="output filename.")
    
    //filename_t img_in_fn; //xNESI(default="%(boda_test_dir)/pascal/000001.jpg",help="input image filename")
    filename_t img_out_fn; // NESI(default="%(boda_output_dir)/out_%%s.png", help="format for filenames of"
                           //   " output image bin files. %%s will replaced with the bin index.")
    string out_layer_name;//NESI(default="conv5",help="output layer name of which to output top blob of")
    //uint32_t write_output; //xNESI(default=0,help="if true, write output images/bins (slow)")
    uint32_t disp_feats; //NESI(default=1,help="if true, display output feature images/bins")
    p_img_pyra_pack_t ipp; //NESI(default="()",help="pyramid packing options")
    p_capture_t capture; //NESI(default="()",help="capture from camera options")

    uint32_t zero_trash; //NESI(default=1,help="1=zero out trash features (not computed from valid per-scale data)")
    
    p_img_t in_img;
    p_img_t feat_img; 
    p_asio_fd_t cap_afd;
    disp_win_t disp_win;

    void on_cap_read( error_code const & ec ) { 
      assert_st( !ec );
      capture->on_readable( 1 );
      p_img_t ds_img = resample_to_size( capture->cap_img, ipp->in_sz );
      in_img->share_pels_from( ds_img );
      ipp->scale_and_pack_img_into_bins( in_img );
      for( uint32_t bix = 0; bix != ipp->bin_imgs.size(); ++bix ) {
	subtract_mean_and_copy_img_to_batch( cnet_predict->in_batch, 0, ipp->bin_imgs[bix] );
	p_nda_float_t out_batch = cnet_predict->run_one_blob_in_one_blob_out();
	if( disp_feats && (bix == 0) ) {
	  timer_t t("conv_pyra_write_output");
	  if( zero_trash ) {
	    for( vect_scale_info_t::const_iterator i = cnet_predict->scale_infos.begin(); i != cnet_predict->scale_infos.end(); ++i ) {
	      if( i->bix != 0 ) { continue; } // wrong plane
	      copy_batch_to_img( out_batch, i->bix, feat_img, i32_to_u32(i->feat_box) );
	    }
	  }
	  else { copy_batch_to_img( out_batch, 0, feat_img, u32_box_t{} ); }
	}
      }
      disp_win.update_disp_imgs();
      setup_capture_on_read( *cap_afd, &conv_pyra_t::on_cap_read, this );
    }

    void on_lb( error_code const & ec ) { 
      lb_event_t const & lbe = get_lb_event(&disp_win);
      //printf( "lbe.img_ix=%s lbe.xy=%s\n", str(lbe.img_ix).c_str(), str(lbe.xy).c_str() );
      if( disp_feats && (lbe.img_ix == 0) ) { feat_pyra_anno_for_xy( 0, i32_to_u32( lbe.xy ) ); }
      //else if( lbe.img_ix == 1 ) { img_to_feat_anno( i32_to_u32( lbe.xy ) ); }
      register_lb_handler( disp_win, &conv_pyra_t::on_lb, this );
    }

    // FIXME: dup'd with cap_app.cc
    void anno_feat_img_xy( u32_pt_t const & feat_xy ) {
      u32_box_t feat_pel_box{feat_xy,feat_xy+u32_pt_t{1,1}};
      i32_box_t const feat_img_pel_box = u32_to_i32(feat_pel_box.scale(cnet_predict->out_s));
      feat_annos->push_back( anno_t{feat_img_pel_box, rgba_to_pel(170,40,40), 0, 
	    str(feat_xy), rgba_to_pel(220,220,255) } );
    } 
    
    void feat_pyra_anno_for_xy( uint32_t const bix, u32_pt_t const & pyra_img_xy ) {
      feat_annos->clear();
      img_annos->clear();
      u32_pt_t const pyra_xy = floor_div_u32( pyra_img_xy, cnet_predict->out_s );
      for( vect_scale_info_t::const_iterator i = cnet_predict->scale_infos.begin(); i != cnet_predict->scale_infos.end(); ++i ) {
	if( i->bix != bix ) { continue; } // wrong plane
	if( i->feat_box.strictly_contains( u32_to_i32(pyra_xy) ) ) {
	  printf( "(*i)=%s\n", str((*i)).c_str() );
	  anno_feat_img_xy( pyra_xy );

	  // FIXME: use correct chan? doesn't matter?
	  i32_box_t const & img_box = cnet_predict->pred_state.at( i->get_psix( 0, pyra_xy ) ).img_box; 
	  img_annos->push_back( anno_t{img_box, rgba_to_pel(170,40,40), 0, str(img_box),rgba_to_pel(220,220,255)});

	}
      }
      setup_annos();
    }   

    virtual void main( nesi_init_arg_t * nia ) { 
      timer_t t("conv_prya_top");
      //p_img_t img_in( new img_t );
      //img_in->load_fn( img_in_fn.exp );
      //u32_pt_t const img_in_sz( img_in->w, img_in->h );
      ipp->in_sz = cnet_predict->in_sz; // 'nominal' scale=1.0 desired image size ...
      in_img.reset( new img_t );
      in_img->set_sz_and_alloc_pels( ipp->in_sz );
      cnet_predict->in_sz = ipp->bin_sz; // but, we will actually run cnet with images of size ipp->bin_sz
      cnet_predict->in_num_imgs = 1;
      cnet_predict->out_layer_name = out_layer_name; // FIXME: too error prone? automate / check / inherit?
      cnet_predict->setup_cnet();

      ipp->do_place_imgs( cnet_predict->conv_pipe->conv_sis.back() );


      vect_p_img_t disp_imgs;
      if( disp_feats ) { 
	feat_img.reset( new img_t );
	u32_pt_t const feat_img_sz = cnet_predict->get_one_blob_img_out_sz();
	feat_img->set_sz_and_alloc_pels( feat_img_sz );
	feat_img->fill_with_pel( grey_to_pel( 0 ) );
	disp_imgs.push_back( feat_img ); 
      }
      disp_imgs.push_back( in_img );
      disp_win.disp_setup( disp_imgs );
      register_lb_handler( disp_win, &conv_pyra_t::on_lb, this );

      cnet_predict->setup_scale_infos( ipp->sizes, ipp->placements, ipp->in_sz );
      cnet_predict->setup_predict();
      img_annos.reset( new vect_anno_t );
      feat_annos.reset( new vect_anno_t );
      setup_annos();

      io_service_t & io = get_io( &disp_win );
      capture->cap_start();
      cap_afd.reset( new asio_fd_t( io, ::dup(capture->get_fd() ) ) );
      setup_capture_on_read( *cap_afd, &conv_pyra_t::on_cap_read, this );
      io.run();
    }

    p_vect_anno_t feat_annos;
    p_vect_anno_t img_annos;
    
    void setup_annos( void ) {
      for( vect_scale_info_t::const_iterator i = cnet_predict->scale_infos.begin(); 
	   i != cnet_predict->scale_infos.end(); ++i ) {
	assert_st( i->feat_box.is_strictly_normalized() );
	if( disp_feats ) {
	  if (i->bix == 0) { // only working on plane 0 for now
	    feat_annos->push_back( anno_t{ i->feat_img_box, rgba_to_pel(170,40,40), 0, 
		  str(i->img_sz), rgba_to_pel(220,220,255) } );
	  } else {
	    printf( "warning: unhanded bix=%s (>1 plane or scale didn't fit if bix=const_max)\n", str(i->bix).c_str() ); 
	  }
	}
      }
      uint32_t diix = 0;
      if( disp_feats ) { disp_win.update_img_annos( diix++, feat_annos ); }
      disp_win.update_img_annos( diix++, img_annos );
    }
  
  };  
#include"gen/conv_pyra.cc.nesi_gen.cc"
}
