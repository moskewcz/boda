// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"has_main.H"
#include"lexp.H"
#include"img_io.H"
#include"results_io.H"
#include"disp_util.H"
#include"cap_util.H"

#include"asio_util.H"
#include"anno_util.H"
#include"rand_util.H"

namespace boda 
{
  struct display_test_t : virtual public nesi, public has_main_t // NESI(help="video display test",
			  // bases=["has_main_t"], type_id="display_test")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    virtual void main( nesi_init_arg_t * nia ) { 
      disp_win_t disp_win;
      disp_win.disp_setup( {{500,500},{200,200}} ); 
      p_vect_anno_t annos( new vect_anno_t );
      annos->push_back( anno_t{{{100,50},{200,250}}, rgba_to_pel(170,40,40), 0, "foo biz baz\nfoo bar\njiggy", rgba_to_pel(220,220,255) } );
      disp_win.update_img_annos( 0, annos );
      disp_win.update_img_annos( 1, annos );
      disp_win.update_disp_imgs();

      io_service_t & io = get_io( &disp_win );
      io.run();
    }
  };

  void mod_adj( uint32_t & val, uint32_t const & max_val, int32_t inc ) { // val must be (strictly) < max_val
    assert_st( val < max_val );
    while( inc < 0 ) { inc += max_val; }
    val += inc;
    while( !(val < max_val) ) { val -= max_val; }
  }

  struct display_pil_t : virtual public nesi, public load_imgs_from_pascal_classes_t, public has_main_t // NESI(
			 // help="display PASCAL VOC list of images in video window",
			 // bases=["load_imgs_from_pascal_classes_t","has_main_t"], type_id="display_pil")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    double fps; //NESI(default=5,help="frames to (try to ) send to display per second (note: independant of display rate)")
    uint32_t rand_winds; //NESI(default=0,help="if set, display 1/2 image size random windows instead of full image")
    uint32_t auto_adv; //NESI(default=1,help="if set, slideshow mode")
    uint32_t do_score; //NESI(default=1,help="if set, run scoring")
    disp_win_t disp_win;
    p_vect_p_img_t disp_imgs;
    p_deadline_timer_t frame_timer;
    time_duration frame_dur;

    display_pil_t( void ) { }
    uint32_t cur_img_ix;
    boost::random::mt19937 gen;

    filename_t prc_txt_fn; //NESI(default="%(boda_output_dir)/prc_",help="output: text prc curve base filename")
    filename_t prc_png_fn; //NESI(default="%(boda_output_dir)/mAP_",help="output: png prc curve base filename")
    p_vect_p_vect_scored_det_t scored_dets;

    void score_img( p_img_info_t const & img_info ) {
      for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
	p_vect_scored_det_t const & sds = scored_dets->at( i - classes->begin() );
	vect_base_scored_det_t & img_sds = sds->get_per_img_sds( img_info->ix );
	img_sds.push_back( base_scored_det_t{u32_box_t{{50,50},{200,200}},10.4} );
	//oct_dfc( cout, dpm_fast_cascade_dir.exp, scored_dets->back(), img_info->full_fn, img_info->ix );
      }
    }

    void on_frame( error_code const & ec ) {
      if( ec ) { return; }
      assert( !ec );
      frame_timer->expires_at( frame_timer->expires_at() + frame_dur );
      frame_timer->async_wait( bind( &display_pil_t::on_frame, this, _1 ) ); 
      assert( cur_img_ix < all_imgs->size() );
      p_img_t const & img = all_imgs->at(cur_img_ix);
      u32_pt_t src_pt, dest_pt;
      u32_pt_t copy_sz{u32_pt_t_const_max};
      if( rand_winds ) {
	copy_sz = ceil_div( img->sz, {2,2} );
	src_pt = random_pt( img->sz - copy_sz, gen );
	dest_pt = random_pt( disp_imgs->at(0)->sz - copy_sz, gen );
      }
      img_copy_to_clip( img.get(), disp_imgs->at(0).get(), dest_pt, src_pt, copy_sz );
      assert_st( cur_img_ix < img_db->img_infos.size() );
      p_img_info_t img_info = img_db->img_infos[cur_img_ix];
      p_vect_anno_t annos( new vect_anno_t );
      for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
	string const & cn = *i;
	vect_gt_det_t const & gt_dets = img_info->gt_dets[cn];	
	vect_gt_match_t * gtms = 0;
	// annotate SDs
	if( do_score ) {
	  p_vect_scored_det_t const & sds = scored_dets->at( i - classes->begin() );
	  gtms = &sds->get_gtms( img_info->ix, gt_dets.size() );
	  
	  vect_base_scored_det_t & img_sds = sds->get_per_img_sds( img_info->ix );
	  for( vect_base_scored_det_t::const_iterator i = img_sds.begin(); i != img_sds.end(); ++i ) {
	    annos->push_back( anno_t{u32_to_i32(*i), rgba_to_pel(40,40,170), 0, cn + "=" + str(i->score), rgba_to_pel(220,220,255) } );
	  }	
	}

	// annotate GTs
	for( uint32_t i = 0; i != gt_dets.size(); ++i ) {
	  bool const is_matched = (!gtms) || gtms->at(i).matched;
	  uint32_t gt_color = is_matched ? rgba_to_pel(40,170,40) : rgba_to_pel(170,40,40);
	  annos->push_back( anno_t{u32_to_i32(gt_dets[i]), gt_color, 0, cn, rgba_to_pel(220,220,255) } );
	  if( gtms && is_matched ) { auto_adv = 0; }
	}

      }

      disp_win.update_img_annos( 0, annos );
      if( auto_adv ) { mod_adj( cur_img_ix, all_imgs->size(), 1 ); }
      disp_win.update_disp_imgs();
    }
    void on_quit( error_code const & ec ) { get_io( &disp_win ).stop(); }

    void on_lb( error_code const & ec ) { 
      register_lb_handler( disp_win, &display_pil_t::on_lb, this ); // re-register handler for next event
      lb_event_t const & lbe = get_lb_event(&disp_win);
      //printf( "lbe.is_key=%s lbe.keycode=%s\n", str(lbe.is_key).c_str(), str(lbe.keycode).c_str() );
      bool unknown_command = 0;
      if( 0 ) { }
      else if( lbe.is_key && (lbe.keycode == 'd') ) { mod_adj( cur_img_ix, all_imgs->size(),  1 ); auto_adv=0; }
      else if( lbe.is_key && (lbe.keycode == 'a') ) { mod_adj( cur_img_ix, all_imgs->size(), -1 ); auto_adv=0; }
      else if( lbe.is_key && (lbe.keycode == 'p') ) { auto_adv ^= 1; }
      else if( lbe.is_key ) { // unknown command handlers
	unknown_command = 1; 
	printf("unknown/unhandled UI key event with keycode = %s\n", str(lbe.keycode).c_str() ); } 
      else { unknown_command = 1; printf("unknown/unhandled UI event\n"); } // unknown command
      if( !unknown_command ) { // if known command, force redisplay now
	frame_timer->cancel();
	frame_timer->expires_from_now( time_duration() );
	frame_timer->async_wait( bind( &display_pil_t::on_frame, this, _1 ) ); 
      }
    }

    virtual void main( nesi_init_arg_t * nia ) {
      load_all_imgs();

      if( do_score ) {
	// setup scored_dets
	scored_dets.reset( new vect_p_vect_scored_det_t );
	for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
	  scored_dets->push_back( p_vect_scored_det_t( new vect_scored_det_t( *i ) ) );
	}
	for (uint32_t ix = 0; ix < img_db->img_infos.size(); ++ix) {
	  p_img_info_t img_info = img_db->img_infos[ix];
	  score_img( img_info );
	}	
	for( vect_p_vect_scored_det_t::iterator i = scored_dets->begin(); i != scored_dets->end(); ++i ) { 
	  (*i)->merge_per_img_sds(); }
	img_db->score_results( scored_dets, prc_txt_fn.exp, prc_png_fn.exp );
      }

      disp_imgs = disp_win.disp_setup( {{640,480}} );
      
      cur_img_ix = 0;
      io_service_t & io = get_io( &disp_win );
      frame_timer.reset( new deadline_timer_t( io ) );
      frame_dur = microseconds( 1000 * 1000 / fps );
      frame_timer->expires_from_now( time_duration() );
      frame_timer->async_wait( bind( &display_pil_t::on_frame, this, _1 ) );
      register_quit_handler( disp_win, &display_pil_t::on_quit, this );
      register_lb_handler( disp_win, &display_pil_t::on_lb, this );

      io.run();


    }

  };

#include"gen/disp_app.cc.nesi_gen.cc"

}
