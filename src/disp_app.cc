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
#include"io_util.H"

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

  // scoring experiments example command lines:
  // ../../lib/boda display_pil --pil-fn="%(pascal_data_dir)/ImageSets/Main/%%s_test.txt" --pascal-classes-fn="%(boda_test_dir)/pascal_classes.txt" --rand-winds=0 --fps=1000 --prc-png-fn="" --do-score=1

  struct display_pil_t : virtual public nesi, public load_pil_t // NESI(
			 // help="display PASCAL VOC list of images in video window",
			 // bases=["load_pil_t"], type_id="display_pil")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    double fps; //NESI(default=5,help="frames to (try to ) send to display per second (note: independant of display rate)")
    uint32_t rand_winds; //NESI(default=0,help="if set, display 1/2 image size random windows instead of full image")
    uint32_t auto_adv; //NESI(default=1,help="if set, slideshow mode")
    uint32_t do_score; //NESI(default=0,help="if set, run scoring. 1: use rp_boxes; 2: use res_fn.")
    double det_show_thresh; //NESI(default=0.0,help="when do_score is set, visualize/show only dets with score >= this value")
    uint32_t quit_after_score; //NESI(default=0,help="if set, quit after scoring.")
    uint32_t show_filt; //NESI(default=0,help="0 == show_all, 1 == show matched, 2 == show unmatched")
    disp_win_t disp_win;
    p_vect_p_img_t disp_imgs;
    p_deadline_timer_t frame_timer;
    time_duration frame_dur;

    display_pil_t( void ) { }
    uint32_t cur_img_ix;
    boost::random::mt19937 gen;

    filename_t prc_txt_fn; //NESI(default="%(boda_output_dir)/prc_",help="output: text prc curve base filename")
    filename_t prc_png_fn; //NESI(default="%(boda_output_dir)/mAP_",help="output: png prc curve base filename")
    p_vect_p_per_class_scored_dets_t scored_dets;

    filename_t res_fn; //NESI(default="%(bench_dir)/results/%%s_test.txt",help="format for filenames of pascal-VOC format DPM detection results files. %%s will be replaced with the class name")

    vect_u32_box_t rp_boxes;
    filename_t rp_boxes_fn; //NESI(default="rps.txt",help="input: region proposal boxes")


    void score_img( p_img_info_t const & img_info ) {
      p_vect_base_scored_det_t img_sds( new vect_base_scored_det_t );
      for( vect_u32_box_t::const_iterator i = rp_boxes.begin(); i != rp_boxes.end(); ++i ) {
	img_sds->push_back( base_scored_det_t{*i,1} );
      }

      for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
	p_per_class_scored_dets_t const & sds = scored_dets->at( i - classes->begin() );
	//p_vect_base_scored_det_t & img_sds = sds->get_per_img_sds( img_info->ix );
	sds->get_per_img_sds( img_info->ix, 0 ) = img_sds;
	//p_img_t const & img = img_info->img;
	//img_sds.push_back( base_scored_det_t{u32_box_t{img->sz.scale_and_round(0.14),img->sz.scale_and_round(1.0 - 0.14)},10.4} );
	//oct_dfc( cout, dpm_fast_cascade_dir.exp, scored_dets->back(), img_info->full_fn, img_info->ix );
      }
    }

    void on_frame( error_code const & ec ) {
      if( ec ) { return; }
      assert( !ec );
      frame_timer->expires_at( frame_timer->expires_at() + frame_dur );
      frame_timer->async_wait( bind( &display_pil_t::on_frame, this, _1 ) ); 
      uint32_t const start_img = cur_img_ix;
      while( 1 ) {
	bool keep_img = 0;
	assert_st( cur_img_ix < img_db->img_infos.size() );
	p_img_info_t img_info = img_db->img_infos[cur_img_ix];
	p_img_t const & img = img_info->img;
	u32_pt_t src_pt, dest_pt;
	u32_pt_t copy_sz{u32_pt_t_const_max};
	if( rand_winds ) {
	  copy_sz = ceil_div( img->sz, {2,2} );
	  src_pt = random_pt( img->sz - copy_sz, gen );
	  dest_pt = random_pt( disp_imgs->at(0)->sz - copy_sz, gen );
	}
	if( !rand_winds ) { disp_imgs->at(0)->fill_with_pel( grey_to_pel( 128 ) ); }
	img_copy_to_clip( img.get(), disp_imgs->at(0).get(), dest_pt, src_pt, copy_sz );
	p_vect_anno_t annos( new vect_anno_t );
	for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
	  string const & cn = *i;
	  vect_gt_det_t const & gt_dets = img_info->gt_dets[cn];
	  vect_gt_match_t * gtms = 0;
	  // annotate SDs
	  if( do_score ) {
	    p_per_class_scored_dets_t const & sds = scored_dets->at( i - classes->begin() );
	    gtms = &sds->get_gtms( img_info->ix, gt_dets.size() );
	    p_vect_base_scored_det_t const & img_sds = sds->get_per_img_sds( img_info->ix, 0 );
	    if( img_sds ) {
              for( vect_base_scored_det_t::const_iterator i = img_sds->begin(); i != img_sds->end(); ++i ) {
                //printf( "cn=%s i->score=%s\n", str(cn).c_str(), str(i->score).c_str() );
                if( i->score > det_show_thresh ) {
                  annos->push_back( anno_t{u32_to_i32(*i), rgba_to_pel(40,40,170), 0, cn + "=" + str(i->score), rgba_to_pel(220,220,255) } );
                }
              }
            }
	  }

	  // annotate GTs
	  keep_img |= (show_filt == 0); // always keep if show_filt==0
	  keep_img |= (!gt_dets.empty()); // keep if any gts
	  for( uint32_t i = 0; i != gt_dets.size(); ++i ) {
	    bool const is_matched = (!gtms) || gtms->at(i).matched;
	    uint32_t gt_color = is_matched ? rgba_to_pel(40,170,40) : rgba_to_pel(170,40,40);
	    annos->push_back( anno_t{u32_to_i32(gt_dets[i]), gt_color, 0, cn, rgba_to_pel(220,220,255) } );
	    if( show_filt != 0 ) {
	      if( !gtms ) { keep_img |= 1; } // always keep if no scoring done
	      else if( is_matched ? (show_filt==1) : (show_filt==2) ) { keep_img |= 1; }
	    }
	  }
	}
	if( !keep_img ) { 
	  mod_adj( cur_img_ix, img_db->img_infos.size(), 1 ); 
	  if( cur_img_ix == start_img ) { printf("no images to show.\n"); break; } 
	}
	else {
	  disp_win.update_img_annos( 0, annos );
	  if( auto_adv ) { mod_adj( cur_img_ix, img_db->img_infos.size(), 1 ); }
	  disp_win.update_disp_imgs();
	  break;
	}
      }
    }
    void on_quit( error_code const & ec ) { get_io( &disp_win ).stop(); }

    void on_lb( error_code const & ec ) { 
      register_lb_handler( disp_win, &display_pil_t::on_lb, this ); // re-register handler for next event
      lb_event_t const & lbe = get_lb_event(&disp_win);
      //printf( "lbe.is_key=%s lbe.keycode=%s\n", str(lbe.is_key).c_str(), str(lbe.keycode).c_str() );
      bool unknown_command = 0;
      if( 0 ) { }
      else if( lbe.is_key && (lbe.keycode == 'd') ) { mod_adj( cur_img_ix, img_db->img_infos.size(),  1 ); auto_adv=0; }
      else if( lbe.is_key && (lbe.keycode == 'a') ) { mod_adj( cur_img_ix, img_db->img_infos.size(), -1 ); auto_adv=0; }
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
      load_img_db( 1 );
      if( img_db->img_infos.empty() ) { rt_err( "no images loaded. check loading options/files?" ); }

      if( do_score ) {
	// setup scored_dets
	scored_dets.reset( new vect_p_per_class_scored_dets_t );
        if( do_score == 1 ) { // use one set of rp_boxes as dets for all classes
          read_text_file( rp_boxes, rp_boxes_fn.exp );
          for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
            scored_dets->push_back( p_per_class_scored_dets_t( new per_class_scored_dets_t( *i ) ) );
          }
          for (uint32_t ix = 0; ix < img_db->img_infos.size(); ++ix) {
            p_img_info_t img_info = img_db->img_infos[ix];
            score_img( img_info );
          }
        } else if( do_score == 2 ) { // use res_fn to get per-class results files
          for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
            scored_dets->push_back( read_results_file( img_db, strprintf( res_fn.exp.c_str(), (*i).c_str() ), *i ) );
          }
        } else {
          rt_err( "unknown do_score value; should be 1 or 2 (or 0 to disable)" );
        }
        
	img_db->score_results( scored_dets, prc_txt_fn.exp, prc_png_fn.exp, "", quit_after_score ); // NOTE: no summary output
	if( quit_after_score ) { return; }
      }

      disp_imgs = disp_win.disp_setup( {img_db->get_max_img_sz()} );
      
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
