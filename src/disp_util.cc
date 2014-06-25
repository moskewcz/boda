// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"disp_util.H"
#include<SDL.h>
#include"timers.H"
#include"str_util.H"
#include"img_io.H"
#include<poll.h>
#if 0
struct timespec deadline;
clock_gettime(CLOCK_MONOTONIC, &deadline);

// Add the time you want to sleep
deadline.tv_nsec += 1000;

// Normalize the time to account for the second boundary
if(deadline.tv_nsec >= 1000000000) {
    deadline.tv_nsec -= 1000000000;
    deadline.tv_sec++;
}
clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, NULL);
#endif

namespace boda 
{
  using namespace boost;

  typedef vector< pollfd > vect_pollfd;

  void delay_secs( uint32_t const secs ) {
    timespec delay;
    delay.tv_sec = secs;
    delay.tv_nsec = 0;
    int const ret = clock_nanosleep( CLOCK_MONOTONIC, 0, &delay, 0 );
    if( ret ) { assert_st( ret == EINTR ); } // note: EINTR is allowed but not handled. could handle using remain arg.
  }

#define DECL_MAKE_P_SDL_OBJ( tn ) p_SDL_##tn make_p_SDL( SDL_##tn * const rp ) { return p_SDL_##tn( rp, SDL_Destroy##tn ); }

  DECL_MAKE_P_SDL_OBJ( Window );
  DECL_MAKE_P_SDL_OBJ( Renderer );
  DECL_MAKE_P_SDL_OBJ( Texture );

#undef DECL_MAKE_P_SDL_OBJ

  void * pipe_stuffer( void * rpv_pfd ) {
    int const pfd = (int)(intptr_t)rpv_pfd;
    uint8_t const c = 123;
    for( uint32_t i = 0; i < 10; ++i ) {
      ssize_t const w_ret = write( pfd, &c, 1 );
      assert_st( w_ret == 1 );
      delay_secs( 1 );
    }
    return 0;
  }

  void disp_win_t::disp_skel( vect_p_img_t const & imgs, poll_req_t * const poll_req ) {
    assert_st( !imgs.empty() );
    
    if( SDL_Init( SDL_INIT_VIDEO ) < 0 ) { rt_err( strprintf( "Couldn't initialize SDL: %s\n", SDL_GetError() ) ); }

    uint32_t window_w = 640;
    uint32_t window_h = 480;
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

    uint32_t img_w = imgs.front()->w;
    uint32_t img_h = imgs.front()->h;
    for( vect_p_img_t::const_iterator i = imgs.begin(); i != imgs.end(); ++i ) {
      max_eq( img_w, (*i)->w );
      max_eq( img_h, (*i)->h );
    }
    assert( !tex );
    tex = make_p_SDL( SDL_CreateTexture( renderer.get(), pixel_format, SDL_TEXTUREACCESS_STREAMING, 
						       img_w, img_h ) );
    p_uint8_t yuv_buf = ma_p_uint8_t( img_w * img_h * 3, 4096 );
    for( uint32_t i = 0; i < img_w * img_h * 3; ++i ) { yuv_buf.get()[i] = 128; }

    if( !tex ) { rt_err( strprintf( "Couldn't set create texture: %s\n", SDL_GetError()) ); }

    int fix = 0;
    bool const nodelay = 0;
    int fps = 60;

    timespec fpsdelay{0,0};
    if( !nodelay ) { fpsdelay.tv_nsec = 1000 * 1000 * 1000 / fps; }

    SDL_Rect displayrect;
    displayrect.x = 0;
    displayrect.y = 0;
    displayrect.w = window_w;
    displayrect.h = window_h;

    SDL_Event event;
    bool paused = 0;
    bool done = 0;

    int pipe_fds[2];
    int const pipe_ret = pipe( pipe_fds );
    assert_st( pipe_ret == 0 );

    pthread_t pipe_stuffer_thread;
    int const pthread_ret = pthread_create( &pipe_stuffer_thread, 0, &pipe_stuffer, (void *)(intptr_t)pipe_fds[1] );
    assert_st( pthread_ret == 0 );

    vect_pollfd pollfds;
    pollfds.push_back( pollfd{ pipe_fds[0], POLLIN } );
    if( poll_req ) { pollfds.push_back( poll_req->get_pollfd() ); }
    while (!done) {
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_WINDOWEVENT:
                if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
		  SDL_RenderSetViewport(renderer.get(), NULL);
		  displayrect.w = window_w = event.window.data1;
		  displayrect.h = window_h = event.window.data2;
                }
                break;
            case SDL_MOUSEBUTTONDOWN:
                displayrect.x = event.button.x - window_w / 2;
                displayrect.y = event.button.y - window_h / 2;
                break;
            case SDL_MOUSEMOTION:
                if (event.motion.state) {
                    displayrect.x = event.motion.x - window_w / 2;
                    displayrect.y = event.motion.y - window_h / 2;
                }
                break;
            case SDL_KEYDOWN:
                if (event.key.keysym.sym == SDLK_SPACE) {
                    paused = !paused;
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

	int const ppoll_ret = ppoll( &pollfds[0], pollfds.size(), &fpsdelay, 0 );
	if( ppoll_ret < 0 ) {
	  assert_st( ppoll_ret == EINTR ); // FIXME: should handle
	} else if( ppoll_ret > 0 ) {
	  {
	    short const re = pollfds[0].revents;
	    assert_st( !( re & POLLERR ) );
	    assert_st( !( re & POLLHUP ) );
	    assert_st( !( re & POLLNVAL ) );
	    if( re & POLLIN ) {
	      uint8_t c = 0;
	      int const read_ret = read( pollfds[0].fd, &c, 1 );
	      assert_st( read_ret == 1 );
	      printf( "c=%s\n", str(uint32_t(c)).c_str() );
	      assert_st( c == 123 );
	    }
	  }
	  if( poll_req ) { assert( 1 < pollfds.size() ); poll_req->check_pollfd( pollfds[1] ); }
	}

        if (!paused) {
	  fix = (fix + 1) % imgs.size();
	  uint8_t * __restrict__ yuv_dest = yuv_buf.get();
	  for( uint32_t y = 0; y < imgs[fix]->h; ++y ) { 
	    for( uint32_t x = 0; x < imgs[fix]->w; ++x ) { 
	      yuv_dest[x] = get_chan( 1, imgs[fix]->get_pel( x, y ) );
	    }
	    yuv_dest += img_w;
	  }
	  SDL_UpdateTexture( tex.get(), NULL, yuv_buf.get(), img_w );
        }
        SDL_RenderClear( renderer.get() );
        SDL_RenderCopy( renderer.get(), tex.get(), NULL, &displayrect);
        SDL_RenderPresent( renderer.get() );
    }

    SDL_Quit();
  }
}
