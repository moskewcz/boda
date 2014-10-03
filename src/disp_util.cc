// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"disp_util.H"
#include<SDL.h>
#include"timers.H"
#include"str_util.H"
#include"img_io.H"
#include<poll.h>
#include"mutex.H"
#include<boost/asio.hpp>
#include<boost/bind.hpp>
#include<boost/date_time/posix_time/posix_time.hpp>


namespace boda 
{
  using namespace boost;

#define DECL_MAKE_P_SDL_OBJ( tn ) p_SDL_##tn make_p_SDL( SDL_##tn * const rp ) { return p_SDL_##tn( rp, SDL_Destroy##tn ); }

  DECL_MAKE_P_SDL_OBJ( Window );
  DECL_MAKE_P_SDL_OBJ( Renderer );
  DECL_MAKE_P_SDL_OBJ( Texture );

#undef DECL_MAKE_P_SDL_OBJ

  struct YV12_buf_t {
    p_uint8_t d;
    uint32_t w;
    uint32_t h;

    // calculated fields
    uint32_t sz;
    uint8_t * Y;
    uint8_t * V;
    uint8_t * U;

    YV12_buf_t( void ) : w(0), h(0), sz(0), Y(0), V(0), U(0) { }
    void set_sz_and_alloc( uint32_t const w_, uint32_t const h_ ) {
      w = w_; assert_st( !(w&1) );
      h = h_; assert_st( !(h&1) );
      sz = w * ( h + (h/2) );
      d = ma_p_uint8_t( sz, 4096 );
      for( uint32_t i = 0; i < sz; ++i ) { d.get()[i] = 128; } // init to grey
      Y = d.get();
      V = Y + w*h; // w*h == size of Y
      U = V + (w/2)*(h/2); // (w/2)*(h/2) == size of V (and U, which is unneeded)
    }

    void YVUat( uint8_t * & out_Y, uint8_t * & out_V, uint8_t * & out_U, uint32_t const & x, uint32_t const & y ) const { 
      assert_st( x < w ); assert_st( y < h );
      out_Y = Y + y*w + x; out_V = V + (y>>1)*(w>>1) + (x>>1); out_U = U + (y>>1)*(w>>1) + (x>>1); 
    }
  };

  void img_to_YV12( YV12_buf_t const & YV12_buf, p_img_t const & img, uint32_t const out_x, uint32_t const out_y )
  {
    uint32_t const w = img->w; assert( !(w&1) );
    uint32_t const h = img->h; assert( !(h&1) );
    uint8_t *out_Y, *out_V, *out_U;
    for( uint32_t y = 0; y < h; ++y ) {
      YV12_buf.YVUat( out_Y, out_V, out_U, out_x, out_y+y );
      uint32_t const * rgb = img->get_row_pels_data( y );
      for( uint32_t x = 0; x < w; ++x, ++rgb ) {
	rgba2y( *rgb, *(out_Y++) );
	if (x % 2 == 0 && y % 2 == 0) { rgba2uv( *rgb, *(out_U++), *(out_V++) ); }
      }
      
    }
  }

  typedef boost::system::error_code error_code;
  typedef boost::asio::posix::stream_descriptor asio_fd_t;
  typedef shared_ptr< asio_fd_t > p_asio_fd_t; 

  struct asio_t {
    asio_t( void ) : frame_timer(io), pipe_read_afd(io), pipe_iter(0), pipe_timer(io), pipe_write_afd(io) { }
    boost::asio::io_service io;
    boost::asio::deadline_timer frame_timer;
    posix_time::time_duration frame_dur;
    asio_fd_t pipe_read_afd;
    uint8_t pipe_read_data;
    uint32_t pipe_iter;
    boost::asio::deadline_timer pipe_timer;
    asio_fd_t pipe_write_afd;
    uint8_t pipe_write_data;
  };
  
  boost::asio::io_service & get_io( disp_win_t * const dw ) { return dw->asio->io; }

  void on_frame( disp_win_t * const dw, error_code const & ec ) {
    if( ec ) { return; } // handle?
    //printf( "dw->asio->frame_timer->expires_at()=%s\n", str(dw->asio->frame_timer.expires_at()).c_str() );
    dw->drain_sdl_events_and_redisplay();
    if( !dw->done ) { 
      dw->asio->frame_timer.expires_at( dw->asio->frame_timer.expires_at() + dw->asio->frame_dur );
      dw->asio->frame_timer.async_wait( bind( on_frame, dw, _1 ) ); 
    }
    else { 
      SDL_Quit();
      dw->asio->io.stop();  // should be optional?
    }
  }

  void disp_win_t::on_pipe_data( error_code const & ec ) {
    if( ec ) { return; }
    printf( "uint32_t(asio->pipe_data)=%s\n", str(uint32_t(asio->pipe_read_data)).c_str() );
    async_read( asio->pipe_read_afd, asio::buffer( &asio->pipe_read_data, 1 ), bind( &disp_win_t::on_pipe_data, this, _1 ) );
  }

  void pipe_stuffer( disp_win_t * const dw, error_code const & ec ) {
    assert_st( !ec ); // timer error?
    if( dw->asio->pipe_iter ) {
      --dw->asio->pipe_iter;
      dw->asio->pipe_timer.expires_at( dw->asio->pipe_timer.expires_at() + posix_time::seconds( 1 ) );
      dw->asio->pipe_timer.async_wait( bind( pipe_stuffer, dw, _1 ) ); 
      // note: write throws on error (including blocking), which is okay since no error (including blocking) is expected / handleable
      write( dw->asio->pipe_write_afd, asio::buffer( &dw->asio->pipe_write_data, 1 ) ); 
    }
  }

  // FIXME: the size of imgs and the w/h of the img_t's inside imgs
  // may not change after setup, but this is not checked.

  void disp_win_t::disp_setup( p_img_t const & img ) {
    p_vect_p_img_t req_imgs( new vect_p_img_t );
    req_imgs->push_back( img );
    disp_setup( req_imgs );
  }

  void disp_win_t::disp_setup( vect_p_img_t const & imgs_ ) {
    p_vect_p_img_t req_imgs( new vect_p_img_t(imgs_) );
    disp_setup( req_imgs );
  }

  p_vect_p_img_t disp_win_t::disp_setup( vect_u32_pt_t const & disp_img_szs ) {
    p_vect_p_img_t req_imgs( new vect_p_img_t );
    for( vect_u32_pt_t::const_iterator i = disp_img_szs.begin(); i != disp_img_szs.end(); ++i ) {
      p_img_t img( new img_t );
      img->set_sz_and_alloc_pels( i->d[0], i->d[1] );
      img->fill_with_pel( grey_to_pel( 128 ) );
      req_imgs->push_back( img );
    }
    disp_setup( req_imgs );
    return req_imgs;
  }

  void disp_win_t::disp_setup( p_vect_p_img_t const & imgs_ ) {
    imgs = imgs_;
    assert_st( !imgs->empty() );
    
    if( SDL_Init( SDL_INIT_VIDEO ) < 0 ) { rt_err( strprintf( "Couldn't initialize SDL: %s\n", SDL_GetError() ) ); }

    window_w = 640;
    window_h = 480;
    assert( !window );
    window = make_p_SDL( SDL_CreateWindow( "boda display", 
							SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
							window_w, window_h,
							SDL_WINDOW_RESIZABLE) );
    if( !window ) { rt_err( strprintf( "Couldn't set create window: %s\n", SDL_GetError() ) ); }
    assert( !renderer );
    renderer = make_p_SDL( SDL_CreateRenderer( window.get(), -1, 0) ) ;
    if (!renderer) { rt_err( strprintf( "Couldn't set create renderer: %s\n", SDL_GetError() ) ); }

#if 0
    SDL_RendererInfo  rinfo;
    SDL_GetRendererInfo( renderer.get(), &rinfo );
    printf( "rinfo.name=%s\n", str(rinfo.name).c_str() );
#endif

    //uint32_t const pixel_format = SDL_PIXELFORMAT_ABGR8888;
    uint32_t const pixel_format = SDL_PIXELFORMAT_YV12;
    YV12_buf.reset( new YV12_buf_t );
    {
      uint32_t img_w = 0;
      uint32_t img_h = 0;
      for( vect_p_img_t::const_iterator i = imgs->begin(); i != imgs->end(); ++i ) {
	img_w += (*i)->w;
	max_eq( img_h, (*i)->h );
      }
      // make w/h even for simplicity of YUV UV (2x downsampled) planes
      if( img_w & 1 ) { ++img_w; }
      if( img_h & 1 ) { ++img_h; }
      YV12_buf->set_sz_and_alloc( img_w, img_h );
    }

    assert( !tex );
    tex = make_p_SDL( SDL_CreateTexture( renderer.get(), pixel_format, SDL_TEXTUREACCESS_STREAMING, 
						       YV12_buf->w, YV12_buf->h ) );

    if( !tex ) { rt_err( strprintf( "Couldn't set create texture: %s\n", SDL_GetError()) ); }


    displayrect.reset( new SDL_Rect );
    displayrect->x = 0;
    displayrect->y = 0;
    displayrect->w = window_w;
    displayrect->h = window_h;

    paused = 0;
    done = 0;
    frame_cnt = 0;
    int const fps = 60;

    asio.reset( new asio_t );
    asio->frame_dur = posix_time::microseconds( 1000 * 1000 / fps );
    asio->frame_timer.expires_from_now( posix_time::time_duration() );
    asio->frame_timer.async_wait( bind( on_frame, this, _1 ) );

#if 0
    int pipe_fds[2];
    int const pipe_ret = pipe( pipe_fds );
    assert_st( pipe_ret == 0 );

    asio->pipe_read_afd.assign( pipe_fds[0] );
    async_read( asio->pipe_read_afd, asio::buffer( &asio->pipe_read_data, 1 ), bind( &disp_win_t::on_pipe_data, this, _1 ) );

    asio->pipe_write_afd.assign( pipe_fds[1] ); // note: ownership transfer
    asio->pipe_write_afd.non_blocking( 1 );

    asio->pipe_write_data = 123;
    asio->pipe_iter = 1000;
    asio->pipe_timer.expires_from_now( posix_time::time_duration() );
    asio->pipe_timer.async_wait( bind( pipe_stuffer, this, _1 ) );
#endif

  }

  // call when changes to imgs should be reflected/copied onto the display texture
  void disp_win_t::update_disp_imgs( void ) {
    if (!paused) {
      uint32_t out_x = 0;
      for( uint32_t i = 0; i != imgs->size(); ++i ) { 
	img_to_YV12( *YV12_buf, imgs->at(i), out_x, YV12_buf->h - imgs->at(i)->h );
	out_x += imgs->at(i)->w;
      }
      SDL_UpdateTexture( tex.get(), NULL, YV12_buf->d.get(), YV12_buf->w );
    }
  }

  void disp_win_t::drain_sdl_events_and_redisplay( void ) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      switch (event.type) {
      case SDL_WINDOWEVENT:
	if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
	  SDL_RenderSetViewport(renderer.get(), NULL);
	  displayrect->w = window_w = event.window.data1;
	  displayrect->h = window_h = event.window.data2;
	}
	break;
      case SDL_MOUSEBUTTONDOWN:
	displayrect->x = event.button.x - window_w / 2;
	displayrect->y = event.button.y - window_h / 2;
	break;
      case SDL_MOUSEMOTION:
	if (event.motion.state) {
	  displayrect->x = event.motion.x - window_w / 2;
	  displayrect->y = event.motion.y - window_h / 2;
	}
	break;
      case SDL_KEYDOWN:
	if( event.key.keysym.sym == SDLK_s ) {
	  for( uint32_t i = 0; i != imgs->size(); ++i ) {
	    imgs->at(i)->save_fn_png( strprintf( "ss_%s.png", str(i).c_str() ) );
	  }
	  paused = 1;
	  break;
	}
	if (event.key.keysym.sym == SDLK_SPACE) {
	  paused = !paused;
	  break;
	}
	if (event.key.keysym.sym == SDLK_r) {
	  imgs->at(0)->fill_with_pel( grey_to_pel( frame_cnt % 256 ) );
	  break;
	}
	if (event.key.keysym.sym != SDLK_ESCAPE) {
	  break;
	}
      case SDL_QUIT:
	done = SDL_TRUE;
	break;
      }
    }
    SDL_RenderClear( renderer.get() );
    SDL_RenderCopy( renderer.get(), tex.get(), NULL, displayrect.get() );
    ++frame_cnt;
    SDL_RenderPresent( renderer.get() );
  }

}
