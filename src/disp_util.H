#ifndef _DISP_UTIL_H_
#define _DISP_UTIL_H_
#include"boda_base.H"

#define PRE_DECL_SDL_OBJ( tn ) \
  struct SDL_##tn; \
  typedef boost::shared_ptr< SDL_##tn > p_SDL_##tn;

PRE_DECL_SDL_OBJ( Window );
PRE_DECL_SDL_OBJ( Renderer );
PRE_DECL_SDL_OBJ( Texture );

#undef PRE_DECL_SDL_OBJ

namespace boda {

  struct img_t;
  typedef shared_ptr< img_t > p_img_t; 
  typedef vector< p_img_t > vect_p_img_t;

  struct disp_win_t {
    p_SDL_Window window;
    p_SDL_Renderer renderer;
    p_SDL_Texture tex;

    void disp_skel( vect_p_img_t const & imgs );
  };

}

#endif /* _DISP_UTIL_H_ */
