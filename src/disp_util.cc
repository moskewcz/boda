// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"disp_util.H"
#include<SDL.h>
#include<SDL_ttf.h>
#include"timers.H"
#include"str_util.H"
#include"img_io.H"
#include<poll.h>
#include"mutex.H"
#include"asio_util.H"
#include"anno_util.H"

namespace boda 
{
  
  SDL_Rect box_to_sdl( i32_box_t const & b ) {
    i32_pt_t const bsz = b.sz();
    i32_pt_t const & bnc = b.p[0];
    return SDL_Rect{bnc.d[0],bnc.d[1],bsz.d[0],bsz.d[1]};
  }
  i32_box_t box_from_sdl( SDL_Rect const & b ) { return i32_box_t{{b.x,b.y},{b.x+b.w,b.y+b.h}}; }

  SDL_Color color_to_sdl( uint32_t const & c ) { return SDL_Color{ get_chan(0,c),get_chan(1,c),get_chan(2,c),get_chan(3,c)}; }
  void sdl_set_color_from_pel( p_SDL_Renderer const & r, uint32_t const & c ) {
    SDL_SetRenderDrawColor( r.get(), get_chan(0,c),get_chan(1,c),get_chan(2,c),get_chan(3,c) ); }

  void rt_err_sdl( char const * const msg ); // like rt_err, but prints sdl error
  void rt_err_sdl( char const * const msg ) { rt_err( strprintf( "%s (SDL error: %s)", msg, SDL_GetError() ) ); }

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

  struct asio_t {
    asio_t( void ) : frame_timer(io), quit_event(io) { }
    io_service_t io;
    deadline_timer_t frame_timer;
    time_duration frame_dur;
    deadline_timer_t quit_event;
  };
  
  // for now, our quit model is to stop all io on quit unless someone
  // has requested the quit event, in which case we assume they will
  // handle things.
  io_service_t & get_io( disp_win_t * const dw ) { return dw->asio->io; }
  deadline_timer_t & get_quit_event( disp_win_t * const dw ) { dw->stop_io_on_quit = 0; return dw->asio->quit_event; }

  void on_frame( disp_win_t * const dw, error_code const & ec ) {
    if( ec ) { return; } // handle?
    //printf( "dw->asio->frame_timer->expires_at()=%s\n", str(dw->asio->frame_timer.expires_at()).c_str() );
    dw->drain_sdl_events_and_redisplay();
    if( !dw->done ) { 
      dw->asio->frame_timer.expires_at( dw->asio->frame_timer.expires_at() + dw->asio->frame_dur );
      dw->asio->frame_timer.async_wait( bind( on_frame, dw, _1 ) ); 
    }
    else { 
      dw->asio->quit_event.cancel();
      SDL_Quit();
      if( dw->stop_io_on_quit ) { dw->asio->io.stop(); }
    }
  }

  disp_win_t::disp_win_t( void ) : stop_io_on_quit(1), asio( new asio_t ) { }

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
    img_annos.resize( imgs->size() );
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

    asio->frame_dur = microseconds( 1000 * 1000 / fps );
    asio->frame_timer.expires_from_now( time_duration() );
    asio->frame_timer.async_wait( bind( on_frame, this, _1 ) );
    asio->quit_event.expires_from_now( pos_infin );

    // font setup
    if( TTF_Init() < 0 ) { rt_err_sdl( "Couldn't initialize TTF" ); }

    string const font_fn = py_boda_dir() +"/fonts/FreeMono.ttf"; // FIXME: use boost filesystem?
    uint32_t const ptsize = 16;
    font.reset( TTF_OpenFont(font_fn.c_str(), ptsize), TTF_CloseFont );
    if( !font ) { rt_err_sdl( strprintf( "Couldn't load %s pt font from %s", str(ptsize).c_str(), font_fn.c_str() ).c_str() ); }

    int const renderstyle = TTF_STYLE_NORMAL;
    int const outline = 0;
    int const hinting = TTF_HINTING_MONO;
    int const kerning = 1;

    TTF_SetFontStyle( font.get(), renderstyle );
    TTF_SetFontOutline( font.get(), outline );
    TTF_SetFontKerning( font.get(), kerning );
    TTF_SetFontHinting( font.get(), hinting );

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

  void disp_win_t::update_img_annos( uint32_t const & img_ix, p_vect_anno_t const & annos ) { img_annos.at(img_ix) = annos; }

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

    i32_pt_t const disp_sz{displayrect->w,displayrect->h}; // display window is of size displayrect w,h
    i32_pt_t const disp_off{displayrect->x,displayrect->y}; // display window x,y is the offset where the neg_corner of the texure will be drawn. 
    i32_pt_t const tex_sz = { YV12_buf->w, YV12_buf->h }; // the texture is always it is always drawn resized to the window size (regardless of offset)
    uint32_t out_x = 0;
    for( uint32_t i = 0; i != imgs->size(); ++i ) { 
      p_img_t const & img = imgs->at(i);
      // calculate what region in the display window this image occupies
      // note: result may be clipped offscreen if it is outside of the visible area of {{0,0},disp_sz}
      i32_pt_t const img_nc = { out_x, YV12_buf->h - img->h };
      i32_pt_t const disp_img_nc = (img_nc*disp_sz/tex_sz) + disp_off;
      i32_pt_t const img_sz = { img->w, img->h };
      i32_pt_t const disp_img_sz = img_sz*disp_sz/tex_sz;
      //i32_box_t const disp_img_box = {disp_img_nc,disp_img_nc+disp_img_sz};
      //printf( "disp_img_box=%s\n", str(disp_img_box).c_str() );
      out_x += imgs->at(i)->w;
      // draw annotations
      p_vect_anno_t const & annos = img_annos.at(i);
      if( !annos ) { continue; }
      for( vect_anno_t::const_iterator i = annos->begin(); i != annos->end(); ++i ) {
	// render box
	sdl_set_color_from_pel( renderer, i->box_color );
	SDL_Rect anno_box = box_to_sdl( (i->box*disp_img_sz/img_sz) + disp_img_nc );
	if( i->fill ) { SDL_RenderFillRect(renderer.get(), &anno_box ); } 
	else { SDL_RenderDrawRect(renderer.get(), &anno_box ); }
	// render string

	p_SDL_Texture str_tex;
	// note: we might use anno_box.w instead of disp_img_sz.d[0]
	// as the auto-wrapping size here to keep the text inside
	// anno_box in X. but, that's not really what we want: if we
	// ever auto-wrap it's probably bad, and if we ever can't fix
	// a word on a line it's bad. so we might just prefer to use a
	// large/infinite value for the wrapping (and let text
	// overflow the anno_box in X as needed). but, the wrapLength
	// must be non-zero to enable wrapping at all, and then it
	// determines the width of the returned surface. so ... we
	// pick a hopefully okay-ish value ...
        p_SDL_Surface text( TTF_RenderText_Blended_Wrapped( font.get(), i->str.c_str(), color_to_sdl(i->str_color), disp_img_sz.d[0] ), 
			    SDL_FreeSurface );
	if( !text ) { printf("text render failed\n"); }
        else { 
	  //assert_st( text->w == anno_box.w );
	  SDL_Rect text_box = anno_box; // for - corner
	  text_box.h = text->h; // may be +- height of anno_box
	  text_box.w = text->w; // may be +- width of anno_box
	  str_tex = make_p_SDL( SDL_CreateTextureFromSurface( renderer.get(), text.get() ) ); 
	  SDL_RenderCopy( renderer.get(), str_tex.get(), NULL, &text_box );
	  
	}

      }
    }      
    ++frame_cnt;
    SDL_RenderPresent( renderer.get() );
  }

}
