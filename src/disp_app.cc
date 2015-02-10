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
    uint32_t rand_winds; //NESI(default=1,help="if set, display 1/2 image size random windows instead of full image")
    uint32_t auto_adv; //NESI(default=1,help="if set, slideshow mode")
    disp_win_t disp_win;
    p_vect_p_img_t disp_imgs;
    p_deadline_timer_t frame_timer;
    time_duration frame_dur;

    uint32_t cur_img_ix;
    boost::random::mt19937 gen;
    display_pil_t( void ) { }
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
      // annotate GTs
      assert_st( cur_img_ix < img_db->img_infos.size() );
      p_img_info_t img_info = img_db->img_infos[cur_img_ix];
      p_vect_anno_t annos( new vect_anno_t );
      for( name_vect_gt_det_map_t::const_iterator i = img_info->gt_dets.begin(); i != img_info->gt_dets.end(); ++i ) {
	vect_gt_det_t const & gt_dets = i->second; // note: may be created here (i.e. may be empty)
	string const & cn = i->first;
	for( vect_gt_det_t::const_iterator i = gt_dets.begin(); i != gt_dets.end(); ++i ) {
	  annos->push_back( anno_t{u32_to_i32(*i), rgba_to_pel(170,40,40), 0, cn, rgba_to_pel(220,220,255) } );
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
