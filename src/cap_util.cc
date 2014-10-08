// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"lexp.H"
#include"img_io.H"
#include"results_io.H"
#include"cap_util.H"

#include"caffeif.H"

// v4l2 capture headers
#include <fcntl.h>              /* low-level i/o */
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/epoll.h>
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

  p_run_cnet_t make_p_run_cnet_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );

  void capture_t::cap_start( void  ) { 
    if( !cap_img ) {
      cap_img.reset( new img_t );
      cap_img->set_sz_and_alloc_pels( cap_res.d[0], cap_res.d[1] ); // w, h
    }
    cap_fd = -1;
    open_device();
    init_device();
    start_capturing();
  }

  void capture_t::cap_stop( void ) {
    if( cap_fd != -1 ) { 
      stop_capturing();
      buffers.clear();
      if( -1 == close(cap_fd) ) { rt_err("close"); } 
      cap_fd = -1;
    }
  }

  // read_req_t iface:
  int capture_t::get_fd( void ) { assert_st( cap_fd != -1 ); return cap_fd; }
  bool capture_t::on_readable( bool const want_frame ) { return read_frame( cap_img, want_frame ); }


  // V4L2 code
  void capture_t::process_image( p_img_t const & img, const void *p, int size )
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
  }

  void must_q_buf( int const & cap_fd, v4l2_buffer & buf ) { 
    if( -1 == xioctl( cap_fd, VIDIOC_QBUF, &buf ) ) { rt_err("VIDIOC_QBUF"); } }

#if 0
  void dq_all_events( int const fd ) {
    v4l2_event ev;
    while( 1 )
    {
      if (-1 == xioctl( fd, VIDIOC_DQEVENT, &ev )) {
	switch (errno) {
	case EINVAL: // no events
	//case EAGAIN: // also no events?
	  return;
	case EIO:
	  // maybe ignore?
	default:
	  rt_err_errno("VIDIOC_DQEVENT");
	}
      }
      printf( "ev.type=%s\n", str(ev.type).c_str() );
    }
  }
#endif

  // read any availible frames, but only process the freshest/newest one (discarding any others)
  bool capture_t::read_frame( p_img_t const & out_img, bool const want_frame )
  {
    timer_t t("read_frame");
    //dq_all_events( cap_fd );
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
	    if( want_frame ) { process_image( out_img, buffers[last_buf.index].get(), buf.bytesused); }
	    must_q_buf( cap_fd, last_buf );
	  }
	  return last_buf_valid && want_frame;
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
      // note: seems like a V4L2/epoll() kernel issues can cause at
      // least late (and then dropped) frames here (with some V4L2
      // drivers?). in particular, the epoll() doesn't return until
      // *all* buffers are full, so we drop frames here (we have
      // nothing to do with them as they are stale), *and* we might be
      // dropping more frames at the camera/driver level ...
      if( last_buf_valid ) { ++read_but_dropped_frames; must_q_buf( cap_fd, last_buf ); }
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
	
      buffers.push_back( make_mmap_shared_p_uint8_t( cap_fd, buf.length, buf.m.offset ) );
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

  void capture_t::main( nesi_init_arg_t * nia ) { 
    // note: works, but subject to V4L2/epoll() kernel bug(s).
    cap_start(); 
    int epfd = -1;
    neg_one_fail( epfd = epoll_create( 1 ), "epoll_create" );
    int const all_ev = EPOLLIN|EPOLLPRI|EPOLLOUT|EPOLLERR|EPOLLHUP|EPOLLET;
    epoll_event ev{ all_ev }; // EPOLLIN|EPOLLET };
    ev.data.fd = cap_fd;
    neg_one_fail( epoll_ctl( epfd, EPOLL_CTL_ADD, cap_fd, &ev ), "epoll_ctl" );
    //printf( "EPOLLIN=%s EPOLLPRI=%s EPOLLOUT=%s EPOLLERR=%s EPOLLHUP=%s EPOLLET=%s\n", str(EPOLLIN).c_str(), str(EPOLLPRI).c_str(), str(EPOLLOUT).c_str(), str(EPOLLERR).c_str(), str(EPOLLHUP).c_str(), str(EPOLLET).c_str() );
    for( uint32_t i = 0; i != 10; ) {
      int num_ev;
      ev = {};
      num_ev = epoll_wait( epfd, &ev, 1, 2000 );
      if( num_ev == -1 ) { if( errno == EINTR ) { continue; } else { rt_err_errno("epoll_wait"); } }
      if( num_ev == 0 ) { rt_err("epoll timeout"); }
      assert_st( num_ev == 1 );
      assert_st( ev.data.fd == cap_fd );
      //printf( "ev.events=%s\n", str(ev.events).c_str() );
      if( read_frame( cap_img, 1 ) ) { ++i; printstr("."); } // got a frame
      else { printstr("-"); }
    }
    printstr("\n");
    // note: the following should be zero. but, due to bug, =9 on my 3.2 kernel ...
    printf( "read_but_dropped_frames=%s\n", str(read_but_dropped_frames).c_str() ); 
    cap_stop(); 
  }
#include"gen/cap_util.H.nesi_gen.cc"
//#include"gen/cap_util.cc.nesi_gen.cc"
  
}
