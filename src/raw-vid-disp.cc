// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"

#include"disp_util.H" 
#include"asio_util.H"
#include"raw-vid-io.H"

namespace boda 
{
  // FIXME: dupe'd code with display_pil_t
  struct display_raw_vid_t : virtual public nesi, public raw_vid_io_t // NESI(
                             // help="display frame from raw video file in video window",
                             // bases=["raw_vid_io_t"], type_id="display-raw-vid")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    u32_pt_t disp_sz; //NESI(default="300 300",help="X/Y display size")
    double fps; //NESI(default=5,help="frames to (try to ) send to display per second (note: independant of display rate)")
    uint32_t auto_adv; //NESI(default=1,help="if set, slideshow mode")
    disp_win_t disp_win;
    p_vect_p_img_t disp_imgs;
    p_deadline_timer_t frame_timer;
    time_duration frame_dur;

    p_img_t in_img;
    
    void on_frame( error_code const & ec ) {
      if( ec ) { return; }
      assert( !ec );
      frame_timer->expires_at( frame_timer->expires_at() + frame_dur );
      frame_timer->async_wait( bind( &display_raw_vid_t::on_frame, this, _1 ) ); 

      p_img_t img = read_next_frame();
      if( !img ) { return; }

      p_img_t ds_img = resample_to_size( img, in_img->sz );
      in_img->share_pels_from( ds_img );

      disp_win.update_disp_imgs();
    }
    void on_quit( error_code const & ec ) { get_io( &disp_win ).stop(); }

    void on_lb( error_code const & ec ) { 
      register_lb_handler( disp_win, &display_raw_vid_t::on_lb, this ); // re-register handler for next event
      lb_event_t const & lbe = get_lb_event(&disp_win);
      //printf( "lbe.is_key=%s lbe.keycode=%s\n", str(lbe.is_key).c_str(), str(lbe.keycode).c_str() );
      bool unknown_command = 0;
      if( 0 ) { }
      //else if( lbe.is_key && (lbe.keycode == 'd') ) { mod_adj( cur_img_ix, img_db->img_infos.size(),  1 ); auto_adv=0; }
      //else if( lbe.is_key && (lbe.keycode == 'a') ) { mod_adj( cur_img_ix, img_db->img_infos.size(), -1 ); auto_adv=0; }
      else if( lbe.is_key && (lbe.keycode == 'i') ) {
        auto_adv=0;
        printf( "fn_map_pos=%s tot_num_read=%s\n", str(fn_map_pos).c_str(), str(tot_num_read).c_str() );
      }
      else if( lbe.is_key && (lbe.keycode == 'p') ) { auto_adv ^= 1; }
      else if( lbe.is_key ) { // unknown command handlers
	unknown_command = 1; 
	printf("unknown/unhandled UI key event with keycode = %s\n", str(lbe.keycode).c_str() ); } 
      else { unknown_command = 1; printf("unknown/unhandled UI event\n"); } // unknown command
      if( !unknown_command ) { // if known command, force redisplay now
	frame_timer->cancel();
	frame_timer->expires_from_now( time_duration() );
	frame_timer->async_wait( bind( &display_raw_vid_t::on_frame, this, _1 ) ); 
      }
    }

    virtual void main( nesi_init_arg_t * nia ) {
      raw_vid_init();
      
      in_img.reset( new img_t );
      in_img->set_sz_and_alloc_pels( disp_sz );
      disp_win.disp_setup( in_img );

      io_service_t & io = get_io( &disp_win );
      frame_timer.reset( new deadline_timer_t( io ) );
      frame_dur = microseconds( 1000 * 1000 / fps );
      frame_timer->expires_from_now( time_duration() );
      frame_timer->async_wait( bind( &display_raw_vid_t::on_frame, this, _1 ) );
      register_quit_handler( disp_win, &display_raw_vid_t::on_quit, this );
      register_lb_handler( disp_win, &display_raw_vid_t::on_lb, this );
      io.run();
    }

  };
#include"gen/raw-vid-disp.cc.nesi_gen.cc"
}
