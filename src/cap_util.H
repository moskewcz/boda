// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _CAP_UTIL_H_
#define _CAP_UTIL_H_
#include"boda_base.H"

namespace boda {

  struct capture_t : public read_req_t, virtual public nesi, public has_main_t // NESI(help="video capture skeleton",
		      // bases=["has_main_t"], type_id="capture")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
#if 0 // FIXME: NESI doesn't understand #if 0'd blocks, so we use the xNESI hack below ...
    filename_t pascal_classes_fn; //xNESI(default="%(boda_test_dir)/pascal/head_10/pascal_classes.txt",help="file with list of classes to process")
    p_img_db_t img_db; //xNESI(default="()", help="image database")
    filename_t pil_fn; //xNESI(default="%(boda_test_dir)/pascal/head_10/%%s.txt",help="format for filenames of image list files. %%s will be replaced with the class name")
#endif
    uint32_t echo_capture; //NESI(default="1", help="if non-zero, display captured frames" )

    filename_t cap_dev; //NESI(default="/dev/video0",help="capture device filename")
    u32_pt_t cap_res; //NESI(default="640 480", help="capture resolution. good choices might be '640 480' or '320 240'. 
    // you can use 'v4l2-ctl --list-formats-ext' to list valid resolutions. (note: v4l2-ctl is in the vl4-utils package in ubuntu).")
    p_img_t cap_img;
    vect_p_img_t disp_imgs;
    img_proc_t * img_proc; 
    capture_t( void ) : img_proc(0) {}

    virtual void main( nesi_init_arg_t * nia );
    void cap_loop( img_proc_t * const img_proc_ );

    // read_req_t iface:
    virtual int get_fd( void );
    virtual void on_readable( void );

    // V4L2 data
    int cap_fd;
    vect_p_uint8_t buffers;

    // V4L2 funcs
    void process_image( p_img_t const & img, const void *p, int size);
    void read_frame( p_img_t const & out_img );

    void init_mmap(void);
    void stop_capturing(void);
    void start_capturing(void);
    void init_device(void);
    void open_device(void);    
  };
  typedef shared_ptr< capture_t > p_capture_t; 

}

#endif /* _CAP_UTIL_H_ */
