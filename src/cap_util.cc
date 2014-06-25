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

  struct cap_skel_t : public poll_req_t, virtual public nesi, public has_main_t // NESI(help="video capture skeleton",
		      // bases=["has_main_t"], type_id="cap_skel")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t pascal_classes_fn; //NESI(default="%(boda_test_dir)/pascal/head_10/pascal_classes.txt",help="file with list of classes to process")
    p_img_db_t img_db; //NESI(default="()", help="image database")
    filename_t pil_fn; //NESI(default="%(boda_test_dir)/pascal/head_10/%%s.txt",help="format for filenames of image list files. %%s will be replaced with the class name")
    uint32_t cap_cam; //NESI(default=5, help="capture N frames from /dev/video0")
    filename_t cap_dev; //NESI(default="/dev/video0",help="capture device filename")

    p_vect_p_img_t disp_imgs;
    virtual void main( nesi_init_arg_t * nia ) { 

      disp_imgs.reset( new vect_p_img_t );

      p_img_t img( new img_t );
      img->set_sz_and_alloc_pels( 640, 480 ); // w, h
      disp_imgs->push_back( img );

#if 0
      p_vect_string classes = readlines_fn( pascal_classes_fn );
      for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
	bool const is_first_class = (i == (*classes).begin());
	read_pascal_image_list_file( img_db, filename_t_printf( pil_fn, (*i).c_str() ), 
				     true && is_first_class, !is_first_class );
      }
      img_db_get_all_loaded_imgs( disp_imgs, img_db );
#endif

      if( cap_cam ) { 
	fd = -1;
	force_format = 1;
	open_device();
	init_device();
	start_capturing();
	frames_left = cap_cam;
      }

      //if( cap_cam ) { mainloop( disp_imgs, cap_cam ); }
      disp_win_t disp_win;
      disp_win.disp_skel( *disp_imgs, cap_cam ? this : 0 ); 

      if( cap_cam ) {
	stop_capturing();
	buffers.clear();
	if( fd != -1 ) { if( -1 == close(fd) ) { rt_err("close"); } }
      }
    }

    virtual pollfd get_pollfd( void ) { assert_st( fd != -1 ); return pollfd{ fd, POLLIN }; }
    virtual void check_pollfd( pollfd const & pfd ) { if( frames_left-- ) { read_frame( disp_imgs ); } }

    // V4L2 cap example vars/code

    int                     fd;
    vect_p_mmap_buffer      buffers;
    int                     force_format;
    uint32_t frames_left;

    void process_image( p_vect_p_img_t const & out, const void *p, int size)
    {
      fflush(stderr);
      fprintf(stderr, ".");
      fflush(stdout);

      //p_img_t img( new img_t );
      //img->set_sz_and_alloc_pels( 640, 480 ); // w, h
      //out->push_back( img );
      assert_st( out->size() == 1 );
      p_img_t img = out->at(0);
      uint8_t const * src = (uint8_t const * )p;
      for( uint32_t i = 0; i != 480; ++i ) {
	for( uint32_t j = 0; j != 640; ++j ) {
	  img->set_pel( j, i, grey_to_pel(*src) );
	  src += 2;
	}
      }
    }

    int read_frame( p_vect_p_img_t const & out )
    {
      v4l2_buffer buf = {};

      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;

      if (-1 == xioctl(fd, VIDIOC_DQBUF, &buf)) {
	switch (errno) {
	case EAGAIN:
	  return 0;

	case EIO:
	  /* Could ignore EIO, see spec. */

	  /* fall through */

	default:
	  rt_err("VIDIOC_DQBUF");
	}
      }

      assert(buf.index < buffers.size());

      process_image( out, buffers[buf.index]->start, buf.bytesused);

      if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
	rt_err("VIDIOC_QBUF");

      return 1;
    }

    void mainloop( p_vect_p_img_t const & out, uint32_t const frame_count )
    {
      unsigned int count;

      count = frame_count;

      while (count-- > 0) {
	for (;;) {
	  fd_set fds;
	  struct timeval tv;
	  int r;

	  FD_ZERO(&fds);
	  FD_SET(fd, &fds);

	  /* Timeout. */
	  tv.tv_sec = 2;
	  tv.tv_usec = 0;

	  r = select(fd + 1, &fds, NULL, NULL, &tv);

	  if (-1 == r) {
	    if (EINTR == errno)
	      continue;
	    rt_err("select");
	  }

	  if (0 == r) {
	    fprintf(stderr, "select timeout\n");
	    exit(EXIT_FAILURE);
	  }

	  if (read_frame( out ))
	    break;
	  /* EAGAIN - continue select loop. */
	}
      }
    }

    void stop_capturing(void) {
      enum v4l2_buf_type type;
      type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      if( -1 == xioctl(fd, VIDIOC_STREAMOFF, &type) ) { rt_err("VIDIOC_STREAMOFF"); }
    }

    void start_capturing(void) {
      unsigned int i;
      enum v4l2_buf_type type;
      for (i = 0; i < buffers.size(); ++i) {
	v4l2_buffer buf = {};
	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;
	buf.index = i;
	if( -1 == xioctl(fd, VIDIOC_QBUF, &buf) ) { rt_err("VIDIOC_QBUF"); }
      }
      type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      if( -1 == xioctl(fd, VIDIOC_STREAMON, &type) ) { rt_err("VIDIOC_STREAMON"); }
    }
    
    void uninit_device(void) { buffers.clear(); }

    void init_mmap(void) {
      v4l2_requestbuffers req = {};
      req.count = 4;
      req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      req.memory = V4L2_MEMORY_MMAP;
      if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
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
	if( -1 == xioctl(fd, VIDIOC_QUERYBUF, &buf) ) { rt_err("VIDIOC_QUERYBUF"); }
	
	buffers.push_back( make_p_mmap_buffer( fd, buf.length, buf.m.offset ) );
      }
    }

    void init_device(void) {
      struct v4l2_capability cap;
      if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap)) {
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
      if (0 == xioctl(fd, VIDIOC_CROPCAP, &cropcap)) {
	crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	crop.c = cropcap.defrect; /* reset to default */
	if (-1 == xioctl(fd, VIDIOC_S_CROP, &crop)) {
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
      if (force_format) {
	fmt.fmt.pix.width       = 640;
	fmt.fmt.pix.height      = 480;
	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
	fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;

	if( -1 == xioctl(fd, VIDIOC_S_FMT, &fmt) ) { rt_err("VIDIOC_S_FMT"); }
	/* Note VIDIOC_S_FMT may change width and height. */
      } else {
	/* Preserve original settings as set by v4l2-ctl for example */
	if (-1 == xioctl(fd, VIDIOC_G_FMT, &fmt)) { rt_err("VIDIOC_G_FMT"); }
      }	
      init_mmap();
    }

    void open_device(void) {
      struct stat st;
      if (-1 == stat(cap_dev.exp.c_str(), &st)) { rt_err( strprintf( "stat failed for '%s': %d, %s\n",
								      cap_dev.in.c_str(), errno, strerror(errno) ) ); }
      if (!S_ISCHR(st.st_mode)) { rt_err( strprintf( "%s is not a device\n", cap_dev.in.c_str() ) ); }
      fd = open(cap_dev.exp.c_str(), O_RDWR /* required */ | O_NONBLOCK, 0);
      if (-1 == fd) { rt_err( strprintf("Cannot open '%s': %d, %s\n", cap_dev.in.c_str(), errno, strerror(errno) ) ); }
    }


    int v4l2_capture_main( uint32_t const num_frames, p_vect_p_img_t const & out ) {
      return 0;
    }
  };
#include"gen/cap_util.cc.nesi_gen.cc"
  
}
