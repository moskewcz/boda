// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"disp_util.H"
#include<SDL.h>
#include"timers.H"
#include"str_util.H"
#include"img_io.H"

#define DECL_MAKE_P_SDL_OBJ( tn ) \
  typedef shared_ptr< SDL_##tn > p_SDL_##tn; \
  p_SDL_##tn make_p_SDL( SDL_##tn * const rp ) { return p_SDL_##tn( rp, SDL_Destroy##tn ); }


namespace boda 
{
  using namespace boost;

  DECL_MAKE_P_SDL_OBJ( Window );
  DECL_MAKE_P_SDL_OBJ( Renderer );
  DECL_MAKE_P_SDL_OBJ( Texture );

  void disp_skel( vect_p_img_t const & imgs ) {
    assert_st( !imgs.empty() );
    
    if( SDL_Init( SDL_INIT_VIDEO ) < 0 ) { rt_err( strprintf( "Couldn't initialize SDL: %s\n", SDL_GetError() ) ); }

    uint32_t window_w = 640;
    uint32_t window_h = 480;
    p_SDL_Window window = make_p_SDL( SDL_CreateWindow( "boda display", 
							SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
							window_w, window_h,
							SDL_WINDOW_RESIZABLE) );
    if( !window ) { rt_err( strprintf( "Couldn't set create window: %s\n", SDL_GetError() ) ); }
    p_SDL_Renderer renderer = make_p_SDL( SDL_CreateRenderer( window.get(), -1, 0) ) ;
    if (!renderer) { rt_err( strprintf( "Couldn't set create renderer: %s\n", SDL_GetError() ) ); }

#if 0
    SDL_RendererInfo  rinfo;
    SDL_GetRendererInfo( renderer.get(), &rinfo );
    printf( "rinfo.name=%s\n", str(rinfo.name).c_str() );
#endif

    //uint32_t const pixel_format = SDL_PIXELFORMAT_ABGR8888;
    uint32_t const pixel_format = SDL_PIXELFORMAT_YV12;

    uint32_t const img_w = imgs.front()->w;
    uint32_t const img_h = imgs.front()->h;
    for( vect_p_img_t::const_iterator i = imgs.begin(); i != imgs.end(); ++i ) {
      assert_st( img_w == (*i)->w );
      assert_st( img_h == (*i)->h );
    }
    p_SDL_Texture tex = make_p_SDL( SDL_CreateTexture( renderer.get(), pixel_format, SDL_TEXTUREACCESS_STREAMING, 
						       img_w, img_w ) );
    p_uint8_t yuv_buf = ma_p_uint8_t( img_w * img_h * 3, 4096 );
    for( uint32_t i = 0; i < img_w * img_h * 3; ++i ) { yuv_buf.get()[i] = 128; }

    if( !tex ) { rt_err( strprintf( "Couldn't set create texture: %s\n", SDL_GetError()) ); }

    int fix = 0;
    bool const nodelay = 0;
    int fps = 60;
    int fpsdelay;


    if (nodelay) {
        fpsdelay = 0;
    } else {
        fpsdelay = 1000 / fps;
    }

    SDL_Rect displayrect;
    displayrect.x = 0;
    displayrect.y = 0;
    displayrect.w = window_w;
    displayrect.h = window_h;

    SDL_Event event;
    bool paused = 0;
    bool done = 0;
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
        SDL_Delay(fpsdelay);

        if (!paused) {
	  fix = (fix + 1) % imgs.size();
	  uint8_t * __restrict__ yuv_dest = yuv_buf.get();
	  for( uint32_t y = 0; y < img_h; ++y ) { 
	    for( uint32_t x = 0; x < img_w; ++x ) { 
	      *yuv_dest++ = get_chan( 1, imgs[fix]->get_pel( x, y ) );
	    }
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
