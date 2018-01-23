// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"

#include"disp_util.H" 
#include"asio_util.H"
#include"data-stream.H"
#include"data-to-img.H"
#include"ext/half.hpp" // for printing nda elems, which might be half

namespace boda 
{

  struct nda_sample_dump_t {
    std::ostream & out;
    i32_pt_t pt;
    nda_sample_dump_t( std::ostream & out_, i32_pt_t const & pt_  ) : out(out_), pt(pt_) {}
    template< typename T > void op( nda_t const & nda ) const {
      dims_t const & dims = nda.dims;
      T const * const elems = static_cast<T const *>(nda.rp_elems());
      uint64_t const num_elems = nda.elems_sz();
      assert_st( elems );
      uint32_t const ys = dims.dstride("y");
      uint32_t const xs = dims.dstride("x");
      uint32_t const yb = (pt.d[1]|1) - 1;
      uint32_t const xb = (pt.d[0]|1) - 1;
      printf( "yb=%s xb=%s: \n", str(yb).c_str(), str(xb).c_str() );
      for( uint32_t y = yb; y < yb+2; ++y ) {
        printf("   ");
        for( uint32_t x = xb; x < xb+2; ++x ) {
          uint32_t off = y*ys + x*xs;
          for( uint32_t c = 0; c != xs; ++c ) {
            assert_st( off+c < num_elems );
            printstr( (c?",":" ") + strN(elems[off+c]) );
          }
        }
        printf("\n");
      }
    }
  };
  
  // FIXME: dupe'd code with display_pil_t
  struct display_raw_vid_t : virtual public nesi, public has_main_t // NESI(
                             // help="display frame from data stream file in video window",
                             // bases=["has_main_t"], type_id="display-raw-vid")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    u32_pt_t window_sz; //NESI(default="640:480",help="X/Y window size")
    u32_pt_t disp_sz; //NESI(default="300:300",help="X/Y per-stream-image size")
    p_layout_elem_t disp_layout; //NESI(help="layout for images")
    double fps; //NESI(default=5,help="frames to (try to ) send to display per second (note: independant of display rate)")
    uint32_t auto_adv; //NESI(default=1,help="if set, slideshow mode")
    uint32_t auto_restart; //NESI(default=1,help="if set, seek to block 0 at end of stream")
    uint32_t print_timestamps; //NESI(default=0,help="if set, print per-frame timestamps")
    uint32_t display_downsample_factor; //NESI(default=1,help="for display, downsample all images by this factor prior to compositing. this reduces the output-image-size of the display, so a smaller texture can be used, copies are faster, and so on. 2 or 4 are nice values to use, since the downsampling is fast.")
    p_data_stream_t src; //NESI(help="data stream to read images from")
    p_data_stream_t sink; //NESI(help="data stream to write YUV image stream to")
    
    disp_win_t disp_win;
    p_deadline_timer_t frame_timer;
    time_duration frame_dur;

    vect_p_img_t in_imgs;
    data_block_t db;

    uint32_t enable_samp_pt; //NESI(default=0,help="if set, enable sampling with some click (BROKEN!).")
    uint32_t samp_pt_sbix;
    i32_pt_t samp_pt;

    u32_pt_t get_db_img_sz( data_block_t const & db ) {
      if( db.as_img ) {
        return ceil_div( db.as_img->sz, u32_pt_t( display_downsample_factor, display_downsample_factor ) );
      }
      else { return disp_sz; } 
    }

    
    void on_frame( error_code const & ec ) {
      if( ec ) { return; }
      assert( !ec );
      frame_timer->expires_at( frame_timer->expires_at() + frame_dur );
      frame_timer->async_wait( bind( &display_raw_vid_t::on_frame, this, _1 ) ); 
      if( !auto_adv ) { return; }
      read_next_block();
    }

    // FIXME: this is sort-mostly-broken. the sample point is really only valid for the as_img img of the block, not
    // neccessarily for the actual nda. for example, for lidar data rendered a point cloud, the image will be some size
    // set by the rendering. but the nda is, say, a dense 64 by 1000 matrix. so ... either we need to sample the image
    // (always), or try to be clever and sample the nda *only if* the dims match the image dims. but, for now, we'll
    // just disable point sampling, since we don't currently need it and it clutters up the output when we turn it on by
    // accident anyway. oh, and on that topic, we should have a way to cancel sampling after selecting a point.
    void proc_samp_pt( data_block_t const & db ) {
      if( samp_pt_sbix == uint32_t_const_max ) { return; } // sampling not enabled
      if( (!db.subblocks.get()) || !( samp_pt_sbix < db.num_subblocks() ) ) {
        printf( "warning: samp_pt_sbix not in number of subblocks (or no subblocks): samp_pt_sbix=%s but db.num_subblocks()=%s. disabling sampling.\n", str(samp_pt_sbix).c_str(), str(db.num_subblocks()).c_str() );
        samp_pt_sbix = uint32_t_const_max;
        return;
      }
      p_nda_t const & nda = db.subblocks->at( samp_pt_sbix ).nda;
      if( !(nda->dims.get_dim_by_name("x") && nda->dims.get_dim_by_name("y") ) ) {
        printf( "warning: for samp_pt_sbix=%s: nda doesn't have x and y dims. disabling sampling.\n", str(samp_pt_sbix).c_str() );
        samp_pt_sbix = uint32_t_const_max;
        return; 
      }

      nda_dispatch( *nda, nda_sample_dump_t( std::cout, samp_pt ) );      
    }

    // FIXME: it's not clear if this is legal/valid for us to set/change in_imgs more than once at startup: disp_setup() may not be okay to re-call.
    void ensure_disp_win_setup( data_block_t const & db ) {
      assert_st( db.has_subblocks() );
      if( in_imgs.size() == db.num_subblocks() ) { // same number of images, check sizes
        bool size_diff = 0;
        for( uint32_t i = 0; i != db.num_subblocks(); ++i ) {
          if( get_db_img_sz( db.subblocks->at(i) ) != in_imgs[i]->sz ) { size_diff = 1; }
        }
        if( !size_diff ) { return; } // all right sizes
      } 
      in_imgs.clear();
      for( uint32_t i = 0; i != db.num_subblocks(); ++i ) {
        u32_pt_t const img_disp_sz = get_db_img_sz( db.subblocks->at(i) );
        in_imgs.push_back( make_shared<img_t>() );
        in_imgs.back()->set_sz_and_alloc_pels( img_disp_sz );
      }
      disp_win.window_sz = window_sz;
      disp_win.disp_layout = disp_layout;
      disp_win.disp_setup( in_imgs );
    }


    void set_camera_opt( void ) {
      data_stream_opt_t cam_opt;
      cam_opt.name = "camera-pos-rot";
      p_nda_float_t cam_pos_rot = make_shared<nda_float_t>( dims_t{ {2,3}, {"pr","d"}, "float" } );
      for( uint32_t i = 0; i != 3; ++i ) {
        cam_pos_rot->at2(0,i) = disp_win.cam_pos[i];
        cam_pos_rot->at2(1,i) = disp_win.cam_rot[i];
      }
      cam_opt.val = cam_pos_rot;
      src->set_opt( cam_opt );
    }

    
    void read_next_block( void ) {
      set_camera_opt();
      while( 1 ) {
        db = src->proc_block(data_block_t());
        if( !db.need_more_in ) { break; }
      }
      if( !db.valid() ) {
        if( auto_restart ) {
          if( !src->seek_to_block(0) ) { printf( "auto-restart: seek to db.frame_ix=0 failed. disabling.\n" ); auto_restart=0; }
          else { printf( "auto-restart: success\n" ); }
        }
        return;
      }
      if( !db.has_subblocks() ) { rt_err( strprintf( "expected subblocks, but num_subblocks=%s\n", str(db.num_subblocks()).c_str() ) ); }
      ensure_disp_win_setup( db );
      if( enable_samp_pt ) { proc_samp_pt( db ); }
      bool had_new_img = 0;
      for( uint32_t i = 0; i != in_imgs.size(); ++i ) {
        data_block_t const & sdb = db.subblocks->at(i);
        p_img_t const & img = sdb.as_img;
        if( !img ) { continue; }
        had_new_img = 1;
        assert_st( get_db_img_sz( sdb ) == in_imgs[i]->sz ); // should be guarenteed by ensure_disp_win_setup
        in_imgs[i] = resample_to_size( img, in_imgs[i]->sz ); // may or may not actually downsample, should never up-samp (currently)
        disp_win.update_disp_img( i, in_imgs[i] );
        // in_imgs[i]->share_pels_from( img ); // img was ds_img here ...
      }
      if( had_new_img ) {
        if( print_timestamps ) { printf( "--- frame: %s ---\n", str(db).c_str() ); }
        disp_win.update_disp_imgs();
        if( sink ) {
          p_img_t out_frame = disp_win.get_borrowed_output_frame();
          data_block_t out_frame_db = db;
          out_frame_db.subblocks.reset();
          out_frame_db.as_img = out_frame;
          sink->proc_block( out_frame_db );
        }
      }
    }
    void on_quit( error_code const & ec ) { get_io( &disp_win ).stop(); }

    void on_lb( error_code const & ec ) {
      register_lb_handler( disp_win, &display_raw_vid_t::on_lb, this ); // re-register handler for next event
      lb_event_t const & lbe = get_lb_event(&disp_win);
      if( !lbe.valid ) { return; }
      //printf( "lbe.is_key=%s lbe.keycode=%s\n", str(lbe.is_key).c_str(), str(lbe.keycode).c_str() );
      bool unknown_command = 0;
      if( 0 ) { }
      if( !lbe.is_key ) {
        if( enable_samp_pt ) {
          if( lbe.img_ix != uint32_t_const_max ) {
            assert_st( lbe.img_ix < in_imgs.size() );
            samp_pt_sbix = lbe.img_ix;
            samp_pt = lbe.xy;
            printf( "set sampling: samp_pt_sbix=%s samp_pt=%s\n", str(samp_pt_sbix).c_str(), str(samp_pt).c_str() );
          }
        }
      }
      else if( lbe.is_key && (lbe.keycode == 'c') ) {
        assert_st( db.has_subblocks() );
        for( uint32_t i = 0; i != db.num_subblocks(); ++i ) {
          data_block_t & sdb = db.subblocks->at(i);
          assert_st( sdb.valid() );
          p_ostream out = ofs_open( strprintf( "src_%s-%s.csv", str(i).c_str(), str(sdb.timestamp_ns).c_str() ) );
          // (*out) << block_to_str( sdb ); // FIXME: used to use old csv-ish format. for now, removed, but we'll dump nda with str()
          // FIXME: should we implement a nicer to-str() for nda_t? don't we have one somewhere already? hmm. would be nive to have reference/standard python-side code for conversion/import of boda nda's into numpy or the like ..
          (*out) << str( sdb.nda ) << "\n";
        }
      }
      else if( lbe.is_key && (lbe.keycode == 'd') ) { read_next_block(); auto_adv=0; }
      else if( lbe.is_key && (lbe.keycode == 'a') ) {
        if( db.frame_ix > 1 ) {
          if( !src->seek_to_block(db.frame_ix - 1) ) {
            printf( "seek to db.frame_ix-1=%s failed.\n", str(db.frame_ix - 1).c_str() );
          } else {  read_next_block(); }
        }
        auto_adv=0;
      }
      else if( lbe.is_key && (lbe.keycode == 'r') ) { if( !src->seek_to_block(0) ) { printf( "seek to db.frame_ix=0 failed.\n" ); } }
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
      samp_pt_sbix = uint32_t_const_max; // invalid/sentinel value to suppress samp_pt prinouts

      src->data_stream_init( nia );
      if( sink ) { sink->data_stream_init( nia ); }
      
      io_service_t & io = get_io( &disp_win );
      frame_timer.reset( new deadline_timer_t( io ) );
      frame_dur = microseconds( 1000 * 1000 / fps );
      frame_timer->expires_from_now( time_duration() );
      frame_timer->async_wait( bind( &display_raw_vid_t::on_frame, this, _1 ) );
      
      register_quit_handler( disp_win, &display_raw_vid_t::on_quit, this );
      register_lb_handler( disp_win, &display_raw_vid_t::on_lb, this );
      io.run();

      if( sink ) { sink->proc_block( data_block_t() ); } // send end-of-stream to sink
      
    }

  };
#include"gen/raw-vid-disp.cc.nesi_gen.cc"
}
