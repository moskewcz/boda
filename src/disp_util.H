#ifndef _DISP_UTIL_H_
#define _DISP_UTIL_H_
#include"boda_base.H"
#include"geom_prim.H"

#define PRE_DECL_SDL_OBJ( tn ) \
  struct SDL_##tn; \
  typedef std::shared_ptr< SDL_##tn > p_SDL_##tn;

PRE_DECL_SDL_OBJ( Window );
PRE_DECL_SDL_OBJ( Renderer );
PRE_DECL_SDL_OBJ( Texture );
PRE_DECL_SDL_OBJ( Rect );

#undef PRE_DECL_SDL_OBJ

struct SDL_Surface;
typedef std::shared_ptr< SDL_Surface > p_SDL_Surface;
struct _TTF_Font;
typedef std::shared_ptr< _TTF_Font > p_TTF_Font;

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

  struct anno_t;
  typedef vector< anno_t > vect_anno_t; 
  typedef shared_ptr< vect_anno_t > p_vect_anno_t; 
  typedef vector< p_vect_anno_t > vect_p_vect_anno_t; 

  struct layout_elem_t;
  typedef vector< layout_elem_t > vect_layout_elem_t; 
  typedef shared_ptr< layout_elem_t > p_layout_elem_t; 
  typedef vector< p_layout_elem_t > vect_p_layout_elem_t;

  struct layout_elem_t : virtual public nesi // NESI(help="op base descriptor") 
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string mode; //NESI(req=1,help="packing mode: vert, horiz, grid, or leaf:IX")
    vect_p_layout_elem_t kids; //NESI(help="child nodes")

    // recursively set sz/nc for layout tree, return in map leaf_ncs
    void calc_layout( map_str_u32_pt_t const & leaf_szs, map_str_u32_pt_t & leaf_ncs );
    void set_size( map_str_u32_pt_t const & leaf_szs );
    void set_leaf_ncs( u32_pt_t path_sum_nc, map_str_u32_pt_t & leaf_ncs );

    // calculated values
    u32_pt_t sz;
    u32_pt_t rel_nc; // relative to parent
  };

  struct disp_win_t {
    p_SDL_Window window;
    p_SDL_Renderer renderer;
    p_SDL_Texture tex;

    p_SDL_Rect displayrect;
    uint32_t cam_mode;
    float pan_orig_cam_x;
    float pan_orig_cam_y;
    float pan_orig_cam_z;
    float pan_orig_cam_rx;
    float pan_orig_cam_ry;
    float pan_orig_cam_rz;
    float cam_pos[3];
    float cam_rot[3];
    void reset_cam( void );
    int32_t zoom;
    i32_pt_t pan_pin;
    i32_pt_t pan_orig_dr;
    bool paused;
    bool done;
    bool stop_io_on_quit;

    u32_pt_t window_sz;

    p_YV12_buf_t YV12_buf;
    p_vect_p_img_t imgs;
    vect_u32_pt_t imgs_buf_nc;
    p_layout_elem_t disp_layout;
    vect_p_vect_anno_t img_annos;

    uint64_t frame_cnt;
    
    p_asio_t asio;

    zi_bool font_setup_done;
    p_TTF_Font font;

    disp_win_t( void );

    void disp_setup( p_vect_p_img_t const & imgs_ ); // reference to list of existing images (for now: don't modify it later)
    void disp_setup( vect_p_img_t const & imgs_ ); // list of existing images 
    void disp_setup( p_img_t const & img ); // one existing image 
    p_vect_p_img_t disp_setup( vect_u32_pt_t const & disp_img_szs ); // create/usr/return a list of images of the given sizes
    void update_dr_for_window_and_zoom( u32_pt_t const & new_win_sz );
    void update_disp_img( uint32_t const pix, p_img_t const & new_img );
    void update_disp_imgs( void );
    p_img_t get_borrowed_output_frame( void );
    void update_img_annos( uint32_t const & img_ix, p_vect_anno_t const & annos );
    void drain_sdl_events_and_redisplay( void );
    void on_lb( int32_t const x, int32_t const y );
  };
}

#endif /* _DISP_UTIL_H_ */
