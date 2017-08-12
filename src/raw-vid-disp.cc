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
    u32_pt_t window_sz; //NESI(default="640:480",help="X/Y window size")
    u32_pt_t disp_sz; //NESI(default="300:300",help="X/Y per-stream-image size")
    double fps; //NESI(default=5,help="frames to (try to ) send to display per second (note: independant of display rate)")
    uint32_t auto_adv; //NESI(default=1,help="if set, slideshow mode")
    uint32_t print_timestamps; //NESI(default=0,help="if set, print per-frame timestamps")
    p_multi_data_stream_t src; //NESI(help="multi data stream to read images from")
    vect_p_data_to_img_t data_to_img; //NESI(help="data stream to img converters (must specify same # of these as data streams)")
    disp_win_t disp_win;
    p_vect_p_img_t disp_imgs;
    p_deadline_timer_t frame_timer;
    time_duration frame_dur;

    uint32_t num_srcs;
    vect_p_img_t in_imgs;
    vect_data_block_t src_dbs;
    
    void on_frame( error_code const & ec ) {
      if( ec ) { return; }
      assert( !ec );
      frame_timer->expires_at( frame_timer->expires_at() + frame_dur );
      frame_timer->async_wait( bind( &display_raw_vid_t::on_frame, this, _1 ) ); 
      if( !auto_adv ) { return; }
      assert_st( num_srcs == data_to_img.size() );
      assert_st( num_srcs == in_imgs.size() );
      bool had_new_img = 0;
      if( print_timestamps ) { printf( "--- frame ---\n"); }
      src->multi_read_next_block( src_dbs );
      for( uint32_t i = 0; i != num_srcs; ++i ) {
        if( print_timestamps ) {
          printf( "src[%s]: got db.timestamp_ns=%s (db.size=%s)\n",
                  str(i).c_str(), str(src_dbs[i].timestamp_ns).c_str(), str(src_dbs[i].sz).c_str() );
        }
        p_img_t img = data_to_img[i]->data_block_to_img( src_dbs[i] );
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
          assert_st( lbe.img_ix < num_srcs );
          data_to_img[lbe.img_ix]->set_samp_pt( lbe.xy );
          printf( "set_samp_pt(%s)\n", str( lbe.xy ).c_str() );
        }
      }
      else if( lbe.is_key && (lbe.keycode == 'c') ) { 
        for( uint32_t i = 0; i != num_srcs; ++i ) {
          data_block_t & db = src_dbs[i];
          assert_st( db.valid() );
          p_ostream out = ofs_open( strprintf( "src_%s-%s.csv", str(i).c_str(), str(db.timestamp_ns).c_str() ) );
          (*out) << data_to_img[i]->data_block_to_str( db );
        }
      }
      //else if( lbe.is_key && (lbe.keycode == 'd') ) { mod_adj( cur_img_ix, img_db->img_infos.size(),  1 ); auto_adv=0; }
      //else if( lbe.is_key && (lbe.keycode == 'a') ) { mod_adj( cur_img_ix, img_db->img_infos.size(), -1 ); auto_adv=0; }
      else if( lbe.is_key && (lbe.keycode == 'i') ) {
        auto_adv=0;
        printf( "src: %s\n", src->get_pos_info_str().c_str() );
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
      num_srcs = src->multi_data_stream_init( nia );
      if( num_srcs != data_to_img.size() ) {
        rt_err( strprintf( "error: must specify same number of data streams and data-to-img converters, but; num_srcs=%s and data_to_img.size()=%s\n",
                           str(num_srcs).c_str(), str(data_to_img.size()).c_str() ) );
      }
      for( uint32_t i = 0; i != num_srcs; ++i ) {
        //stream[i]->data_stream_init( nia );
        data_to_img[i]->data_to_img_init( nia );
        in_imgs.push_back( make_shared<img_t>() );
        in_imgs.back()->set_sz_and_alloc_pels( disp_sz );
      }
      disp_win.window_sz = window_sz;
      disp_win.layout_mode = "vert";
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
