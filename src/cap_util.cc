// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"lexp.H"
#include"img_io.H"
#include"results_io.H"
#include"disp_util.H"
#include"cap_util.H"

#include"caffeif.H"

// v4l2 capture headers
#include <fcntl.h>              /* low-level i/o */
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

namespace boda 
{
  using namespace boost;

  static int xioctl( int fh, unsigned long int request, void *arg ) {
    int ret;
    while( 1 ) {
      ret = ioctl(fh, request, arg);      
      if( (ret == -1) && (errno == EINTR) ) { continue; }
      return ret;
    }
  }

  struct mmap_buffer { void   *start; size_t  length;  };
  typedef shared_ptr< mmap_buffer > p_mmap_buffer;
  typedef vector< p_mmap_buffer > vect_p_mmap_buffer;

  struct mmap_buffer_deleter { 
    void operator()( mmap_buffer * const b ) const { if( munmap( b->start, b->length) == -1 ) { rt_err("munmap"); } } 
  };

  p_mmap_buffer make_p_mmap_buffer( int const fd, size_t const length, off_t const offset ) {
    mmap_buffer ret;

    ret.length = length;
    ret.start =
      mmap(NULL /* start anywhere */,
	   length,
	   PROT_READ | PROT_WRITE /* required */,
	   MAP_SHARED /* recommended */,
	   fd, offset);
    if( MAP_FAILED == ret.start ) { rt_err("mmap"); }
    return p_mmap_buffer( new mmap_buffer( ret ), mmap_buffer_deleter() ); 
  }

  p_run_cnet_t make_p_run_cnet_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );

  void capture_t::main( nesi_init_arg_t * nia ) { cap_loop( 0 ); }

  void capture_t::cap_loop( img_proc_t * const img_proc_ ) { 
    img_proc = img_proc_; 
    cap_img.reset( new img_t );
    cap_img->set_sz_and_alloc_pels( cap_res.d[0], cap_res.d[1] ); // w, h
    if( echo_capture ) { disp_imgs.push_back( cap_img ); }
#if 0
    p_vect_string classes = readlines_fn( pascal_classes_fn );
    for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
      bool const is_first_class = (i == (*classes).begin());
      read_pascal_image_list_file( img_db, filename_t_printf( pil_fn, (*i).c_str() ), 
				   true && is_first_class, !is_first_class );
    }
    img_db_get_all_loaded_imgs( disp_imgs, img_db ); // FIXME: disp_imgs isn't a p_ vect anymore ...
#endif

    cap_fd = -1;
    open_device();
    init_device();
    start_capturing();

    disp_win_t disp_win;
    disp_win.disp_skel( disp_imgs, this ); 

    stop_capturing();
    buffers.clear();
    if( cap_fd != -1 ) { if( -1 == close(cap_fd) ) { rt_err("close"); } }

  }

  // poll_req_t iface:
  pollfd capture_t::get_pollfd( void ) { assert_st( cap_fd != -1 ); return pollfd{ cap_fd, POLLIN }; }
  void capture_t::check_pollfd( pollfd const & pfd ) { read_frame( cap_img ); }


  // V4L2 code
  void capture_t::process_image( p_img_t const & img, const void *p, int size)
  {
    uint8_t const * src = (uint8_t const * )p;
    assert_st( !(cap_res.d[0] & 1) );
    for( uint32_t i = 0; i != cap_res.d[1]; ++i ) {
      for( uint32_t j = 0; j != cap_res.d[0]; j += 2 ) {
	img->set_pel( j+0, i, yuva2rgba(src[0],src[1],src[3]) );
	img->set_pel( j+1, i, yuva2rgba(src[2],src[1],src[3]) );
	src += 4;
      }
    }
    if( img_proc ) { img_proc->on_img( img ); }
  }

  void must_q_buf( int const & cap_fd, v4l2_buffer & buf ) { 
    if( -1 == xioctl( cap_fd, VIDIOC_QBUF, &buf ) ) { rt_err("VIDIOC_QBUF"); } }

  // read any availible frames, but only process the freshest/newest one (discarding any others)
  void capture_t::read_frame( p_img_t const & out_img )
  {
    v4l2_buffer buf = {};
    v4l2_buffer last_buf = {};

    // these two fields should never change/be changed by xioctl(), but this is not checked ...
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; 
    buf.memory = V4L2_MEMORY_MMAP;

    bool last_buf_valid = 0;
    while( 1 ) {
      buf.index = uint32_t_const_max; // set output to an invalid value that the xioctl() should overwrite
      if (-1 == xioctl(cap_fd, VIDIOC_DQBUF, &buf)) {
	switch (errno) {
	case EAGAIN: // no frames left/availible, we're done no matter what
	  if( last_buf_valid ) { // if we got a any frames, process the last one (only)
	    process_image( out_img, buffers[last_buf.index]->start, buf.bytesused);
	    must_q_buf( cap_fd, last_buf );
	  }
	  return;
	case EIO:
	  /* Could ignore EIO, see spec. */
	  /* fall through */
	default:
	  rt_err("VIDIOC_DQBUF");
	}
      }
      // note: we assume the bufs will be DQ'd in squence order. we could handle the general case ...
      // printf( "buf.sequence=%s\n", str(buf.sequence).c_str() );
      // buf is now a valid buf
      assert(buf.index < buffers.size());
      if( last_buf_valid ) { must_q_buf( cap_fd, last_buf ); }
      last_buf_valid = 1;
      last_buf = buf;
    }
  }

  void capture_t::stop_capturing(void) {
    enum v4l2_buf_type type;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if( -1 == xioctl(cap_fd, VIDIOC_STREAMOFF, &type) ) { rt_err("VIDIOC_STREAMOFF"); }
  }

  void capture_t::start_capturing(void) {
    unsigned int i;
    enum v4l2_buf_type type;
    for (i = 0; i < buffers.size(); ++i) {
      v4l2_buffer buf = {};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;
      if( -1 == xioctl(cap_fd, VIDIOC_QBUF, &buf) ) { rt_err("VIDIOC_QBUF"); }
    }
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if( -1 == xioctl(cap_fd, VIDIOC_STREAMON, &type) ) { rt_err("VIDIOC_STREAMON"); }
  }
    
  void capture_t::init_mmap(void) {
    v4l2_requestbuffers req = {};
    req.count = 2;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (-1 == xioctl(cap_fd, VIDIOC_REQBUFS, &req)) {
      if (EINVAL == errno) { rt_err( strprintf( "%s does not support memory mapping\n", cap_dev.in.c_str() ) ); }
      else { rt_err("VIDIOC_REQBUFS"); }
    }
    if (req.count < 2) { rt_err( strprintf( "Insufficient buffer memory on %s\n", cap_dev.in.c_str() ) ); }
    assert_st( buffers.empty() );

    for( uint32_t bix = 0; bix < req.count; ++bix) {
      v4l2_buffer buf = {};
      buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory      = V4L2_MEMORY_MMAP;
      buf.index       = bix;
      if( -1 == xioctl(cap_fd, VIDIOC_QUERYBUF, &buf) ) { rt_err("VIDIOC_QUERYBUF"); }
	
      buffers.push_back( make_p_mmap_buffer( cap_fd, buf.length, buf.m.offset ) );
    }
  }

  void capture_t::init_device(void) {
    struct v4l2_capability cap;
    if (-1 == xioctl(cap_fd, VIDIOC_QUERYCAP, &cap)) {
      if( EINVAL == errno ) { rt_err( strprintf( "%s is not a V4L2 device\n", cap_dev.in.c_str()) ); }
      else { rt_err("VIDIOC_QUERYCAP"); }
    }
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) { 
      rt_err( strprintf( "%s is not a video capture device\n", cap_dev.in.c_str() ) );
    }
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
      rt_err( strprintf( "%s does not support streaming i/o\n", cap_dev.in.c_str() ) );
    }

    /* Select video input, video standard and tune here. */
    struct v4l2_crop crop;
    v4l2_cropcap cropcap = {};
    cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (0 == xioctl(cap_fd, VIDIOC_CROPCAP, &cropcap)) {
      crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      crop.c = cropcap.defrect; /* reset to default */
      if (-1 == xioctl(cap_fd, VIDIOC_S_CROP, &crop)) {
	switch (errno) {
	case EINVAL:
	  /* Cropping not supported. */
	  break;
	default:
	  /* Errors ignored. */
	  break;
	}
      }
    } else {
      /* Errors ignored. */
    }

    v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if( -1 == xioctl( cap_fd, VIDIOC_G_FMT, &fmt ) ) { rt_err("VIDIOC_G_FMT"); } // get current settings
    // alter the settings we need
    fmt.fmt.pix.width       = cap_res.d[0];
    fmt.fmt.pix.height      = cap_res.d[1];
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    if( -1 == xioctl( cap_fd, VIDIOC_S_FMT, &fmt ) ) { rt_err("VIDIOC_S_FMT"); } // try to set our changes to the format

    // check that the current/set format is okay. note that the driver can change (any?) of the fields from what we passed to VIDIOC_S_FMT ...
    if( fmt.fmt.pix.field != V4L2_FIELD_NONE ) {
      printf( "note: fmt.fmt.pix.field was != V4L2_FIELD_NONE. proceeding anyway. fmt.fmt.pix.field=%s fmt.fmt.pix.colorspace=%s\n", 
	      str(fmt.fmt.pix.field).c_str(), str(fmt.fmt.pix.colorspace).c_str() );
    }
    if( fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV ) {
      rt_err( strprintf( "requested pixelformat=V4L2_PIX_FMT_YUYV, but got fmt_res=%s", str(fmt.fmt.pix.pixelformat).c_str() ) );
    }
    u32_pt_t const fmt_res( fmt.fmt.pix.width, fmt.fmt.pix.height );
    if( fmt_res != cap_res ) { rt_err( strprintf( "requested cap_res=%s, but got fmt_res=%s", str(cap_res).c_str(), str(fmt_res).c_str() ) ); }
    if( fmt.fmt.pix.bytesperline != (cap_res.d[0]*2) ) {  // note: *2 is hard-coded for YUVU format req'd above
      rt_err( strprintf( "TODO: fmt.fmt.pix.bytesperline=%s != (cap_res.d[0]*2)=%s", 
			 str(fmt.fmt.pix.bytesperline).c_str(), str(cap_res.d[0]*2).c_str() ) );
    }
    assert_st( fmt.fmt.pix.sizeimage == (fmt.fmt.pix.bytesperline*cap_res.d[1]) ); // could handle, but not possible/okay?

    init_mmap();
  }

  void capture_t::open_device(void) {
    struct stat st;
    if (-1 == stat(cap_dev.exp.c_str(), &st)) { rt_err( strprintf( "stat failed for '%s': %d, %s\n",
								   cap_dev.in.c_str(), errno, strerror(errno) ) ); }
    if (!S_ISCHR(st.st_mode)) { rt_err( strprintf( "%s is not a device\n", cap_dev.in.c_str() ) ); }
    cap_fd = open(cap_dev.exp.c_str(), O_RDWR /* required */ | O_NONBLOCK, 0);
    if (-1 == cap_fd) { rt_err( strprintf("Cannot open '%s': %d, %s\n", cap_dev.in.c_str(), errno, strerror(errno) ) ); }
  }
  
#include"gen/cap_util.H.nesi_gen.cc"
//#include"gen/cap_util.cc.nesi_gen.cc"
  
}
