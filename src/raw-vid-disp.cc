// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"

#include"disp_util.H" 
#include"asio_util.H"
#include"data-stream.H"
#include"data-to-img.H"

namespace boda 
{
  // FIXME: dupe'd code with display_pil_t
  struct display_raw_vid_t : virtual public nesi, public has_main_t // NESI(
                             // help="display frame from data stream file in video window",
                             // bases=["has_main_t"], type_id="display-raw-vid")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    u32_pt_t disp_sz; //NESI(default="300 300",help="X/Y display size")
    double fps; //NESI(default=5,help="frames to (try to ) send to display per second (note: independant of display rate)")
    uint32_t auto_adv; //NESI(default=1,help="if set, slideshow mode")
    uint32_t print_timestamps; //NESI(default=0,help="if set, print per-frame timestamps")
    vect_p_data_stream_t stream; //NESI(help="data stream to read images from")
    vect_p_data_to_img_t data_to_img; //NESI(help="data stream to img converters (must specify same # of these as data streams)")
    disp_win_t disp_win;
    p_vect_p_img_t disp_imgs;
    p_deadline_timer_t frame_timer;
    time_duration frame_dur;

    vect_p_img_t in_imgs;
    

    
    void on_frame( error_code const & ec ) {
      if( ec ) { return; }
      assert( !ec );
      frame_timer->expires_at( frame_timer->expires_at() + frame_dur );
      frame_timer->async_wait( bind( &display_raw_vid_t::on_frame, this, _1 ) ); 
      if( !auto_adv ) { return; }
      assert_st( stream.size() == data_to_img.size() );
      assert_st( stream.size() == in_imgs.size() );
      bool had_new_img = 0;
      if( print_timestamps ) { printf( "--- frame ---\n"); }
      for( uint32_t i = 0; i != stream.size(); ++i ) {
        data_block_t db = stream[i]->read_next_block();
        if( print_timestamps ) {
          printf( "stream[%s]: got db.timestamp_ns=%s (db.size=%s)\n",
                  str(i).c_str(), str(db.timestamp_ns).c_str(), str(db.sz).c_str() );
        }
        p_img_t img = data_to_img[i]->data_block_to_img( db );
        if( !img ) { continue; }
        had_new_img = 1;
        p_img_t ds_img = resample_to_size( img, in_imgs[i]->sz );
        in_imgs[i]->share_pels_from( ds_img );
      }
      if( had_new_img ) { disp_win.update_disp_imgs(); }
    }
    void on_quit( error_code const & ec ) { get_io( &disp_win ).stop(); }

    void on_lb( error_code const & ec ) { 
      register_lb_handler( disp_win, &display_raw_vid_t::on_lb, this ); // re-register handler for next event
      lb_event_t const & lbe = get_lb_event(&disp_win);
      //printf( "lbe.is_key=%s lbe.keycode=%s\n", str(lbe.is_key).c_str(), str(lbe.keycode).c_str() );
      bool unknown_command = 0;
      if( 0 ) { }
      if( !lbe.is_key ) {
        if( lbe.img_ix != uint32_t_const_max ) {
          assert_st( lbe.img_ix < stream.size() );
          data_to_img[lbe.img_ix]->set_samp_pt( lbe.xy );
          printf( "set_samp_pt(%s)\n", str( lbe.xy ).c_str() );
        }
      }
      //else if( lbe.is_key && (lbe.keycode == 'd') ) { mod_adj( cur_img_ix, img_db->img_infos.size(),  1 ); auto_adv=0; }
      //else if( lbe.is_key && (lbe.keycode == 'a') ) { mod_adj( cur_img_ix, img_db->img_infos.size(), -1 ); auto_adv=0; }
      else if( lbe.is_key && (lbe.keycode == 'i') ) {
        auto_adv=0;
        for( uint32_t i = 0; i != stream.size(); ++i ) {
          printf( "stream[%s]: %s\n", str(i).c_str(), stream[i]->get_pos_info_str().c_str() );
        }
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
      if( stream.size() != data_to_img.size() ) {
        rt_err( strprintf( "error: must specify same number of data streams and data-to-img converters, but; stream.size()=%s and data_to_img.size()=%s\n",
                           str(stream.size()).c_str(), str(data_to_img.size()).c_str() ) );
      }
      for( uint32_t i = 0; i != stream.size(); ++i ) {
        stream[i]->data_stream_init( nia );
        data_to_img[i]->data_to_img_init( nia );
        in_imgs.push_back( make_shared<img_t>() );
        in_imgs.back()->set_sz_and_alloc_pels( disp_sz );
      }
      
      disp_win.disp_setup( in_imgs );

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
