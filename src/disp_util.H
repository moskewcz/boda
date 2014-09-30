#ifndef _DISP_UTIL_H_
#define _DISP_UTIL_H_
#include"boda_base.H"

#define PRE_DECL_SDL_OBJ( tn ) \
  struct SDL_##tn; \
  typedef boost::shared_ptr< SDL_##tn > p_SDL_##tn;

PRE_DECL_SDL_OBJ( Window );
PRE_DECL_SDL_OBJ( Renderer );
PRE_DECL_SDL_OBJ( Texture );
PRE_DECL_SDL_OBJ( Rect );

#undef PRE_DECL_SDL_OBJ
namespace boost { namespace system { struct error_code; } }
namespace boda {

  struct img_t;
  typedef shared_ptr< img_t > p_img_t; 
  typedef vector< p_img_t > vect_p_img_t;

  struct YV12_buf_t; typedef shared_ptr< YV12_buf_t > p_YV12_buf_t; 
  typedef vector< img_t > vect_img_t; 
  typedef shared_ptr< img_t > p_img_t; 
  typedef vector< p_img_t > vect_p_img_t;
  typedef shared_ptr< vect_p_img_t > p_vect_p_img_t; 

  struct asio_t; typedef shared_ptr< asio_t > p_asio_t; 

  struct disp_win_t {
    p_SDL_Window window;
    p_SDL_Renderer renderer;
    p_SDL_Texture tex;

    p_SDL_Rect displayrect;
    bool paused;
    bool done;

    uint32_t window_w;
    uint32_t window_h;

    p_YV12_buf_t YV12_buf;
    p_vect_p_img_t imgs;

    uint64_t frame_cnt;

    p_asio_t asio;

    read_req_t * read_req;
    void disp_skel( vect_p_img_t & imgs_, read_req_t * const read_req_ );
    void drain_sdl_events_and_redisplay( void );
    void on_pipe_data( boost::system::error_code const & ec );
  };

}

#endif /* _DISP_UTIL_H_ */
