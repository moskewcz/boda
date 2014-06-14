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

  int v4l2_capture_main( uint32_t const num_frames, p_vect_p_img_t const & out );

  struct cap_skel_t : virtual public nesi, public has_main_t // NESI(help="video capture skeleton",
		      // bases=["has_main_t"], type_id="cap_skel")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t pascal_classes_fn; //NESI(default="%(boda_test_dir)/pascal/head_10/pascal_classes.txt",help="file with list of classes to process")
    p_img_db_t img_db; //NESI(default="()", help="image database")
    filename_t pil_fn; //NESI(default="%(boda_test_dir)/pascal/head_10/%%s.txt",help="format for filenames of image list files. %%s will be replaced with the class name")
    uint32_t cap_cam; //NESI(default=5, help="capture N frames from /dev/video0")

    virtual void main( nesi_init_arg_t * nia ) { 

      p_vect_p_img_t disp_imgs( new vect_p_img_t );

      p_vect_string classes = readlines_fn( pascal_classes_fn );
      for( vect_string::const_iterator i = (*classes).begin(); i != (*classes).end(); ++i ) {
	bool const is_first_class = (i == (*classes).begin());
	read_pascal_image_list_file( img_db, filename_t_printf( pil_fn, (*i).c_str() ), true && is_first_class , !is_first_class );
      }
      img_db_get_all_loaded_imgs( disp_imgs, img_db );

      if( cap_cam ) { v4l2_capture_main( cap_cam, disp_imgs ); }
      
      disp_win_t disp_win;
      disp_win.disp_skel( *disp_imgs ); 

      

    }
  };


#define CLEAR(x) memset(&(x), 0, sizeof(x))

  struct buffer {
    void   *start;
    size_t  length;
  };

  static char            const *dev_name;
  static int              fd = -1;
  struct buffer          *buffers;
  static unsigned int     n_buffers;
  static int		out_buf;
  static int              force_format;

  static void errno_exit(const char *s)
  {
    fprintf(stderr, "%s error %d, %s\n", s, errno, strerror(errno));
    exit(EXIT_FAILURE);
  }

  static int xioctl(int fh, unsigned long int request, void *arg)
  {
    int r;

    do {
      r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);

    return r;
  }

  static void process_image( p_vect_p_img_t const & out, const void *p, int size)
  {
    if (out_buf)
      fwrite(p, size, 1, stdout);

    fflush(stderr);
    fprintf(stderr, ".");
    fflush(stdout);

    p_img_t img( new img_t );
    img->set_sz_and_alloc_pels( 640, 480 ); // w, h
    uint8_t const * src = (uint8_t const * )p;
    for( uint32_t i = 0; i != 480; ++i ) {
      for( uint32_t j = 0; j != 640; ++j ) {
	img->set_pel( j, i, grey_to_pel(*src) );
	src += 2;
      }
    }
    out->push_back( img );
  }

  static int read_frame( p_vect_p_img_t const & out )
  {
    struct v4l2_buffer buf;

    CLEAR(buf);

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
	errno_exit("VIDIOC_DQBUF");
      }
    }

    assert(buf.index < n_buffers);

    process_image( out, buffers[buf.index].start, buf.bytesused);

    if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
      errno_exit("VIDIOC_QBUF");

    return 1;
  }

  static void mainloop( p_vect_p_img_t const & out, uint32_t const frame_count )
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
	  errno_exit("select");
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

  static void stop_capturing(void)
  {
    enum v4l2_buf_type type;

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(fd, VIDIOC_STREAMOFF, &type))
      errno_exit("VIDIOC_STREAMOFF");
  }

  static void start_capturing(void)
  {
    unsigned int i;
    enum v4l2_buf_type type;

    for (i = 0; i < n_buffers; ++i) {
      struct v4l2_buffer buf;

      CLEAR(buf);
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;

      if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
	errno_exit("VIDIOC_QBUF");
    }
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(fd, VIDIOC_STREAMON, &type))
      errno_exit("VIDIOC_STREAMON");

  }

  static void uninit_device(void)
  {
    unsigned int i;

    for (i = 0; i < n_buffers; ++i)
      if (-1 == munmap(buffers[i].start, buffers[i].length))
	errno_exit("munmap");

    free(buffers);
  }

  static void init_mmap(void)
  {
    struct v4l2_requestbuffers req;

    CLEAR(req);

    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
      if (EINVAL == errno) {
	fprintf(stderr, "%s does not support "
		"memory mapping\n", dev_name);
	exit(EXIT_FAILURE);
      } else {
	errno_exit("VIDIOC_REQBUFS");
      }
    }

    if (req.count < 2) {
      fprintf(stderr, "Insufficient buffer memory on %s\n",
	      dev_name);
      exit(EXIT_FAILURE);
    }

    buffers = (buffer *)calloc(req.count, sizeof(*buffers));

    if (!buffers) {
      fprintf(stderr, "Out of memory\n");
      exit(EXIT_FAILURE);
    }

    for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
      struct v4l2_buffer buf;

      CLEAR(buf);

      buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory      = V4L2_MEMORY_MMAP;
      buf.index       = n_buffers;

      if (-1 == xioctl(fd, VIDIOC_QUERYBUF, &buf))
	errno_exit("VIDIOC_QUERYBUF");

      buffers[n_buffers].length = buf.length;
      buffers[n_buffers].start =
	mmap(NULL /* start anywhere */,
	     buf.length,
	     PROT_READ | PROT_WRITE /* required */,
	     MAP_SHARED /* recommended */,
	     fd, buf.m.offset);

      if (MAP_FAILED == buffers[n_buffers].start)
	errno_exit("mmap");
    }
  }

  static void init_device(void)
  {
    struct v4l2_capability cap;
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    struct v4l2_format fmt;

    if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap)) {
      if (EINVAL == errno) {
	fprintf(stderr, "%s is no V4L2 device\n",
		dev_name);
	exit(EXIT_FAILURE);
      } else {
	errno_exit("VIDIOC_QUERYCAP");
      }
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
      fprintf(stderr, "%s is no video capture device\n",
	      dev_name);
      exit(EXIT_FAILURE);
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
      fprintf(stderr, "%s does not support streaming i/o\n",
	      dev_name);
      exit(EXIT_FAILURE);
    }

    /* Select video input, video standard and tune here. */


    CLEAR(cropcap);

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


    CLEAR(fmt);

    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (force_format) {
      fmt.fmt.pix.width       = 640;
      fmt.fmt.pix.height      = 480;
      fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
      fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;

      if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt))
	errno_exit("VIDIOC_S_FMT");

      /* Note VIDIOC_S_FMT may change width and height. */
    } else {
      /* Preserve original settings as set by v4l2-ctl for example */
      if (-1 == xioctl(fd, VIDIOC_G_FMT, &fmt))
	errno_exit("VIDIOC_G_FMT");
    }

	
    init_mmap();
  }

  static void close_device(void)
  {
    if (-1 == close(fd))
      errno_exit("close");

    fd = -1;
  }

  static void open_device(void)
  {
    struct stat st;

    if (-1 == stat(dev_name, &st)) {
      fprintf(stderr, "Cannot identify '%s': %d, %s\n",
	      dev_name, errno, strerror(errno));
      exit(EXIT_FAILURE);
    }

    if (!S_ISCHR(st.st_mode)) {
      fprintf(stderr, "%s is no device\n", dev_name);
      exit(EXIT_FAILURE);
    }

    fd = open(dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

    if (-1 == fd) {
      fprintf(stderr, "Cannot open '%s': %d, %s\n",
	      dev_name, errno, strerror(errno));
      exit(EXIT_FAILURE);
    }
  }


  int v4l2_capture_main( uint32_t const num_frames, p_vect_p_img_t const & out ) {
    dev_name = "/dev/video0";
    force_format = 1;
    out_buf = 0;

    open_device();
    init_device();
    start_capturing();
    mainloop( out, num_frames );
    stop_capturing();
    uninit_device();
    close_device();
    fprintf(stderr, "\n");
    return 0;
  }

#include"gen/cap_util.cc.nesi_gen.cc"
  
}
