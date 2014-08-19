// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _CAP_UTIL_H_
#define _CAP_UTIL_H_
#include"boda_base.H"

namespace boda {

  // FIXME: promote mmap_buffer to boda_base and/or use elsewhere?
  struct mmap_buffer;
  typedef shared_ptr< mmap_buffer > p_mmap_buffer;
  typedef vector< p_mmap_buffer > vect_p_mmap_buffer;

  struct capture_t : public poll_req_t, virtual public nesi, public has_main_t // NESI(help="video capture skeleton",
		      // bases=["has_main_t"], type_id="capture")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t pascal_classes_fn; //NESI(default="%(boda_test_dir)/pascal/head_10/pascal_classes.txt",help="file with list of classes to process")
    p_img_db_t img_db; //NESI(default="()", help="image database")
    filename_t pil_fn; //NESI(default="%(boda_test_dir)/pascal/head_10/%%s.txt",help="format for filenames of image list files. %%s will be replaced with the class name")
    filename_t cap_dev; //NESI(default="/dev/video0",help="capture device filename")
    u32_pt_t cap_res; //NESI(default="640 480", help="capture resolution. good choices might be '640 480' or '320 240'. 
    // you can use 'v4l2-ctl --list-formats-ext' to list valid resolutions. (note: v4l2-ctl is in the vl4-utils package in ubuntu).")
    p_vect_p_img_t disp_imgs;
    img_proc_t * img_proc; 
    capture_t( void ) : img_proc(0) {}

    virtual void main( nesi_init_arg_t * nia );
    void cap_loop( img_proc_t * const img_proc_ );

    // poll_req_t iface:
    virtual pollfd get_pollfd( void );
    virtual void check_pollfd( pollfd const & pfd );

    // V4L2 data
    int cap_fd;
    vect_p_mmap_buffer buffers;

    // V4L2 funcs
    void process_image( p_vect_p_img_t const & out, const void *p, int size);
    int read_frame( p_vect_p_img_t const & out );

    void init_mmap(void);
    void stop_capturing(void);
    void start_capturing(void);
    void init_device(void);
    void open_device(void);    
  };
  typedef shared_ptr< capture_t > p_capture_t; 

}

#endif /* _CAP_UTIL_H_ */
