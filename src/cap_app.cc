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
#include"pyif.H" // for py_boda_dir()

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>



namespace boda 
{

  // prints errno
  void neg_one_fail( int const & ret, char const * const func_name ) {
    if( ret == -1 ) { rt_err( strprintf( "%s() failed with errno=%s (%s)", func_name, str(errno).c_str(),
					 strerror(errno) ) ); }
  }

  void fork_and_exec_self( vect_string const & args ) {
    vect_rp_char argp;
    for( vect_string::const_iterator i = args.begin(); i != args.end(); ++i ) { argp.push_back( (char *)i->c_str() ); }
    argp.push_back( 0 );
    string const self_exe = py_boda_dir() + "/lib/boda"; // note: uses readlink on /proc/self/exe internally
    pid_t const ret = fork();
    if( ret == 0 ) {
      execve( self_exe.c_str(), &argp[0], environ );
      rt_err( strprintf( "execve of '%s' failed. envp=environ, args=%s", self_exe.c_str(), str(args).c_str() ) );
    }
    // ret == child pid, not used
  }

#include"gen/build_info.cc"

  string get_boda_shm_filename( void ) { return strprintf( "/boda-rev-%s-pid-%s-top.shm", build_rev, 
							   str(getpid()).c_str() ); }
  int get_boda_shm_fd( bool const create ) {
    int oflags = O_RDWR;
    if( create ) { oflags |= O_CREAT | O_TRUNC; }
    string const fn = get_boda_shm_filename();
    int const fd = shm_open( fn.c_str(), oflags, S_IRUSR | S_IWUSR );
    if( fd == -1 ) { rt_err( strprintf( "shm_open() failed with errno=%s", str(errno).c_str() ) ); }
    // for now, we'll just pass the open fd to our child process, so
    // we don't need the file/name/link anymore, and by unlinking it
    // here we can try to minimize the chance / amount of OS-level shm
    // leakage.
    neg_one_fail( shm_unlink( fn.c_str() ), "shm_unlink" );
    // by default, the fd returned from shm_open() has FD_CLOEXEC
    // set. it seems okay to remove it so that it will stay open
    // across execve.
    int fd_flags = 0;
    neg_one_fail( fd_flags = fcntl( fd, F_GETFD ), "fcntl" );
    fd_flags &= ~FD_CLOEXEC;
    neg_one_fail( fcntl( fd, F_SETFD, fd_flags ), "fcntl" );
    // resize the shm segment for later mapping via mmap()
    neg_one_fail( ftruncate( fd, 1024*1024*4 ), "ftruncate" );
    return fd;
  }

  void delay_secs( uint32_t const secs );

  struct cs_disp_t : virtual public nesi, public has_main_t // NESI(help="client-server video display test",
			  // bases=["has_main_t"], type_id="cs_disp")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    vect_p_img_t disp_imgs;
    virtual void main( nesi_init_arg_t * nia ) { 
      int const boda_shm_fd = get_boda_shm_fd(1);
      p_uint8_t pels = make_mmap_shared_p_uint8_t( boda_shm_fd, 1024*1024*4, 0 ); // FIXME: check img->row_align wrt map page sz?
      p_img_t img( new img_t );
      //img->set_sz_and_alloc_pels( 100, 100 );
      img->set_sz( 100, 100 );
      img->pels = pels; // make_mmap_shared_p_uint8_t( -1, img->sz_raw_bytes(), 0 ); // FIXME: check img->row_align wrt map page sz?
      img->fill_with_pel( grey_to_pel( 128 ) );
      disp_imgs.push_back( img );

      // fork/exec
      fork_and_exec_self( {"boda","display_test",strprintf("--boda_shm_fd=%s",str(boda_shm_fd).c_str())} );

      for( uint32_t j = 0; j != 10000; ++j ) {
	for( uint32_t i = 0; i != 10; ++i ) {
	  img->fill_with_pel( grey_to_pel( 50 + i*15 ) );
	}
      }

    }
  };


  struct display_test_t : virtual public nesi, public has_main_t // NESI(help="video display test",
			  // bases=["has_main_t"], type_id="display_test")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    int32_t boda_shm_fd; //NESI(default=-1,help="if set, an open fd created by shm_open() in the parent process.")
    vect_p_img_t disp_imgs;
    virtual void main( nesi_init_arg_t * nia ) { 
      p_uint8_t pels = make_mmap_shared_p_uint8_t( boda_shm_fd, 1024*1024*4, 0 ); // FIXME: check img->row_align wrt map page sz?
      p_img_t img( new img_t );
      //img->set_sz_and_alloc_pels( 100, 100 );
      img->set_sz( 100, 100 );
      img->pels = pels; // make_mmap_shared_p_uint8_t( -1, img->sz_raw_bytes(), 0 ); // FIXME: check img->row_align wrt map page sz?
      img->fill_with_pel( grey_to_pel( 128 ) );
      disp_imgs.push_back( img );
      //fork();

      disp_win_t disp_win;
      disp_win.disp_skel( disp_imgs, 0 ); 
    }
  };

  struct capture_classify_t : virtual public nesi, public has_main_t // NESI(help="cnet classifaction from video capture",
			      // bases=["has_main_t"], type_id="capture_classify")
			    , public img_proc_t
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    p_capture_t capture; //NESI(default="()",help="capture from camera options")    
    p_cnet_predict_t cnet_predict; //NESI(default="()",help="cnet running options")    
    virtual void main( nesi_init_arg_t * nia ) { cnet_predict->setup_predict(); capture->cap_loop( this ); }
    virtual void on_img( p_img_t const & img ) { cnet_predict->do_predict( img ); }
  };

  struct capture_feats_t : virtual public nesi, public has_main_t // NESI(help="cnet classifaction from video capture",
			   // bases=["has_main_t"], type_id="capture_feats")
			 , public img_proc_t
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    p_capture_t capture; //NESI(default="()",help="capture from camera options")    
    p_run_cnet_t run_cnet; //NESI(default="(ptt_fn=%(boda_test_dir)/conv_pyra_imagenet_deploy.prototxt,out_layer_name=conv5)",help="cnet running options")
    p_img_t feat_img;
    virtual void main( nesi_init_arg_t * nia ) { 
      run_cnet->in_sz = capture->cap_res;
      run_cnet->setup_cnet(); 
      feat_img.reset( new img_t );
      u32_pt_t const feat_img_sz = run_cnet->get_one_blob_img_out_sz();
      feat_img->set_sz_and_alloc_pels( feat_img_sz.d[0], feat_img_sz.d[1] ); // w, h
      capture->disp_imgs.push_back( feat_img );
      capture->cap_loop( this ); 
    }
    virtual void on_img( p_img_t const & img ) { 
      subtract_mean_and_copy_img_to_batch( run_cnet->in_batch, 0, img );
      p_nda_float_t out_batch = run_cnet->run_one_blob_in_one_blob_out();
      copy_batch_to_img( out_batch, 0, feat_img );
    }
  };


#include"gen/cap_app.cc.nesi_gen.cc"

}
