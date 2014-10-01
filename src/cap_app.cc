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

#include<boost/asio.hpp>
#include<boost/bind.hpp>
#include<boost/date_time/posix_time/posix_time.hpp>

namespace boda 
{
  // prints errno
  void neg_one_fail( int const & ret, char const * const func_name ) {
    if( ret == -1 ) { rt_err( strprintf( "%s() failed with errno=%s (%s)", func_name, str(errno).c_str(),
					 strerror(errno) ) ); }
  }

  struct shm_header_t {
    pthread_once_t init_once;
    pthread_mutex_t mut;
    uint64_t tot_sz;
    uint64_t used_sz;
    uint64_t avail_sz( void ) const { assert( used_sz <= tot_sz ); return tot_sz - used_sz; }
  };

  uint32_t const pagesize = 4096;

  struct shm_seg_t;
  shm_seg_t * once_shm_seg = 0; // only used once in call to shm_seg_once() from pthread_once()
  void shm_seg_once( void );

  struct shm_seg_t {
    int shm_fd;
    p_uint8_t shm;
    uint64_t shm_len;
    shm_header_t * head;
    // uint8_t * comm_buff;

    // because we're allocating a shared memory segment in a 'root' process and sharing it with
    // fork()/exec()'d child processes, we have the luxury being able to run the below init() with
    // is_master=1 before any child processes have been started. so, we almost don't really need to
    // be using pthread_once() here. however, it seems that there's not really any way to
    // synchronize startup (using only the single shared memory segment) other than to use
    // pthread_once() *and* assert that PTHREAD_ONCE_INIT is acutally 0, which is the value the shm
    // seg will get inited to by ftruncate(). in particular, we can't use
    // sem_init()/pthread_mutex_init() since there would be (in thoery) an init/lock race. in
    // practice it seems hard to believe any memory ordering would be loose enough that somehow we
    // could sem_init() *before a fork/exec* and have the child's sem_lock() see the state before
    // the init() ... maybe there's some guarantees i'm not aware of that would make that okay. but,
    // on the other hand, using pthread_once() to init mutexes and such seems pretty standard/okay.
    void init( int const & shm_fd_, bool const is_master ) {
      shm_fd = shm_fd_;
      assert( PTHREAD_ONCE_INIT == 0 ); // ftruncate() will init the shm segment to 0, so this better hold
      assert( pagesize == sysconf( _SC_PAGESIZE ) ); // probably would be okay to be dynamic or different, but seems questionable.
      shm_len = u32_ceil_align( sizeof(shm_header_t), pagesize ); // almost certainly 1
      if( is_master ) { neg_one_fail( ftruncate( shm_fd, shm_len ), "ftruncate" ); }
      shm = make_mmap_shared_p_uint8_t( shm_fd, shm_len, 0 );
      head = (shm_header_t *)shm.get();
      once_shm_seg = this; // ugly, but we can't pass args to shm_seg_once ...
      pthread_once( &head->init_once, shm_seg_once );
    }
    
    void once( void ) {
      pthread_mutex_init( &head->mut, 0 );
      head->tot_sz = shm_len;
      head->used_sz = head->tot_sz;//sizeof(shm_header_t); // FIXME: is this what we want wrt page granularity?
    }

    template< typename T >void rebase( ptrdiff_t const & delta, T * & p ) { p = (T *)((uint8_t *)p + delta); }
    void rebase_all( ptrdiff_t const & delta ) {
      rebase( delta, head );
      //rebase( delta, comm_buff );
    }
    // based on shm_len, update mapping for shm
    void update_mapping( void ) { 
      uint8_t * orig_smh = shm.get();
      printf( "remap: shm.get()=%s shm_len=%s\n", str((void*)shm.get()).c_str(), str(shm_len).c_str() );
      remap_mmap_shared_p_uint8_t( shm, shm_len );
      if( shm.get() != orig_smh ) { rebase_all( shm.get() - orig_smh ); }
    }
    void double_shm( void ) {
      shm_len *= 2;
      neg_one_fail( ftruncate( shm_fd, shm_len ), "ftruncate" );
      update_mapping();
      head->tot_sz = shm_len;
    }
    void check_and_maybe_update_mapping( void ) { if( shm_len != head->tot_sz ) { shm_len = head->tot_sz; update_mapping(); } }

    uint64_t shm_alloc_off( uint64_t const sz ) {
      pthread_mutex_lock( &head->mut );
      check_and_maybe_update_mapping();
      while( head->avail_sz() < sz ) { double_shm(); }
      uint64_t const ret = head->used_sz;
      head->used_sz += sz;
      pthread_mutex_unlock( &head->mut ); // note: &head->mut may have changed since the lock above!
      return ret;
    }
    p_uint8_t shm_alloc( uint64_t const sz ) {
      uint64_t const off = shm_alloc_off( sz );
      return make_mmap_shared_p_uint8_t( shm_fd, sz, off );
    }


  };
  inline void shm_seg_once( void ) { assert( once_shm_seg ); once_shm_seg->once(); }



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
  int get_boda_shm_fd( void ) {
    bool const create = 1; // we could perhaps support making this optional
    int oflags = O_RDWR;
    if( create ) { oflags |= O_CREAT | O_TRUNC | O_EXCL ; }
    string const fn = get_boda_shm_filename();
    shm_unlink( fn.c_str() ); // we don't want to use any existing shm, so try to remove it if it exists.
    // note that, in theory, we could have just unlinked an in-use shm, and thus broke some other
    // processes. also note that between the unlink above and the shm_open() below, someone could
    // create shm with the name we want, and then we will fail. note that, generally speaking,
    // shm_open() doesn't seem secure/securable (maybe one could use a random shm name here?), but
    // we're mainly trying to just be robust -- including being non-malicious
    // errors/bugs/exceptions.
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
    return fd;
  }

  void delay_secs( uint32_t const secs );

  struct cs_disp_t : virtual public nesi, public has_main_t // NESI(help="client-server video display test",
			  // bases=["has_main_t"], type_id="cs_disp")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    virtual void main( nesi_init_arg_t * nia ) { 
      int const boda_shm_fd = get_boda_shm_fd();
      shm_seg_t shm_seg;
      shm_seg.init( boda_shm_fd, 1 );
      // FIXME: check img->row_align wrt map page sz?
      uint64_t pels_off = shm_seg.shm_alloc_off( 1024*1024*4 ); 
      p_uint8_t pels = make_mmap_shared_p_uint8_t( boda_shm_fd, 1024*1024*4, pels_off ); 
      p_img_t img( new img_t );
      //img->set_sz_and_alloc_pels( 100, 100 );
      img->set_sz( 100, 100 );
      img->pels = pels; // make_mmap_shared_p_uint8_t( -1, img->sz_raw_bytes(), 0 ); // FIXME: check img->row_align wrt map page sz?
      img->fill_with_pel( grey_to_pel( 128 ) );

      // fork/exec
      fork_and_exec_self( {"boda","display_ipc",
	    strprintf("--pels-off=%s",str(pels_off).c_str()),
	    strprintf("--boda-shm-fd=%s",str(boda_shm_fd).c_str())} );

      for( uint32_t j = 0; j != 10000; ++j ) {
	for( uint32_t i = 0; i != 10; ++i ) {
	  img->fill_with_pel( grey_to_pel( 50 + i*15 ) );
	}
      }

    }
  };


  typedef boost::system::error_code error_code;
  typedef boost::asio::posix::stream_descriptor asio_fd_t;
  typedef shared_ptr< asio_fd_t > p_asio_fd_t; 
  boost::asio::io_service & get_io( disp_win_t * const dw );

  struct display_ipc_t : virtual public nesi, public has_main_t // NESI(help="video display over ipc test",
			  // bases=["has_main_t"], type_id="display_ipc")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    int32_t boda_shm_fd; //NESI(help="an open fd created by shm_open() in the parent process.",req=1)
    uint64_t pels_off; //NESI(help="offset to pels",req=1)
    virtual void main( nesi_init_arg_t * nia ) { 
      shm_seg_t shm_seg;
      shm_seg.init( boda_shm_fd, 0 );
      p_uint8_t pels = make_mmap_shared_p_uint8_t( boda_shm_fd, 1024*1024*4, pels_off ); // FIXME: check img->row_align wrt map page sz?
      p_img_t img( new img_t );
      //img->set_sz_and_alloc_pels( 100, 100 );
      img->set_sz( 100, 100 );
      img->pels = pels; // make_mmap_shared_p_uint8_t( -1, img->sz_raw_bytes(), 0 ); // FIXME: check img->row_align wrt map page sz?
      img->fill_with_pel( grey_to_pel( 128 ) );
      //fork();
      disp_win_t disp_win;
      disp_win.disp_setup( img );
      get_io( &disp_win ).run();
    }
  };

  // DEMOABLE items:
  //  -- sunglasses (wearing or on surface)
  //  -- keyboard (zoom in)
  //  -- loafer (better have both in view. socks optional)
  //  -- fruit/pear (might work ...)
  //  -- water bottle
  //  -- window shade
  //  -- teddy bear (more or less ...)
  //  -- coffee mug (great if centered)
  //  -- sandals 
  //  -- napkin (aka handkerchief, better be on flat surface) 
  //  -- paper towel (ehh, maybe ...)
  //  -- hat (aka mortarboard / cowboy hat )

  struct capture_classify_t : virtual public nesi, public has_main_t // NESI(help="cnet classifaction from video capture",
			      // bases=["has_main_t"], type_id="capture_classify")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    p_capture_t capture; //NESI(default="()",help="capture from camera options")    
    p_cnet_predict_t cnet_predict; //NESI(default="()",help="cnet running options")    
    p_asio_fd_t cap_afd;
    void on_cap_read( error_code const & ec ) { 
      assert_st( !ec );
      capture->on_readable();
      cnet_predict->do_predict( capture->cap_img ); 
      async_read( *cap_afd, boost::asio::null_buffers(), bind( &capture_classify_t::on_cap_read, this, _1 ) );
    }
    virtual void main( nesi_init_arg_t * nia ) { 
      cnet_predict->setup_predict(); 
      capture->cap_start();
      disp_win_t disp_win;
      disp_win.disp_setup( capture->cap_img );

      boost::asio::io_service & io = get_io( &disp_win );
      cap_afd.reset( new asio_fd_t( io, ::dup(capture->get_fd() ) ) );
      async_read( *cap_afd, boost::asio::null_buffers(), bind( &capture_classify_t::on_cap_read, this, _1 ) );
      io.run();
    }
  };

  struct capture_feats_t : virtual public nesi, public has_main_t // NESI(help="cnet classifaction from video capture",
			   // bases=["has_main_t"], type_id="capture_feats")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    p_capture_t capture; //NESI(default="()",help="capture from camera options")    
    p_run_cnet_t run_cnet; //NESI(default="(ptt_fn=%(boda_test_dir)/conv_pyra_imagenet_deploy.prototxt,out_layer_name=conv3)",help="cnet running options")
    p_img_t feat_img;
    p_asio_fd_t cap_afd;
    void on_cap_read( error_code const & ec ) { 
      assert_st( !ec );
      capture->on_readable();

      subtract_mean_and_copy_img_to_batch( run_cnet->in_batch, 0, capture->cap_img );
      p_nda_float_t out_batch = run_cnet->run_one_blob_in_one_blob_out();
      copy_batch_to_img( out_batch, 0, feat_img );

      async_read( *cap_afd, boost::asio::null_buffers(), bind( &capture_feats_t::on_cap_read, this, _1 ) );
    }
    virtual void main( nesi_init_arg_t * nia ) { 
      run_cnet->in_sz = capture->cap_res;
      run_cnet->setup_cnet(); 
      feat_img.reset( new img_t );
      u32_pt_t const feat_img_sz = run_cnet->get_one_blob_img_out_sz();
      feat_img->set_sz_and_alloc_pels( feat_img_sz.d[0], feat_img_sz.d[1] ); // w, h

      capture->cap_start();
      disp_win_t disp_win;
      disp_win.disp_setup( vect_p_img_t{feat_img,capture->cap_img} );

      boost::asio::io_service & io = get_io( &disp_win );
      cap_afd.reset( new asio_fd_t( io, ::dup(capture->get_fd() ) ) );
      async_read( *cap_afd, boost::asio::null_buffers(), bind( &capture_feats_t::on_cap_read, this, _1 ) );
      io.run();
    }
  };


#include"gen/cap_app.cc.nesi_gen.cc"

}
