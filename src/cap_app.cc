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

#include<boost/asio.hpp>
#include<boost/bind.hpp>
#include<boost/date_time/posix_time/posix_time.hpp>

namespace boda 
{

#include"gen/build_info.cc"

  string get_boda_shm_filename( void ) { return strprintf( "/boda-rev-%s-pid-%s-top.shm", build_rev, 
							   str(getpid()).c_str() ); }

  typedef boost::system::error_code error_code;
  typedef boost::asio::posix::stream_descriptor asio_fd_t;
  typedef shared_ptr< asio_fd_t > p_asio_fd_t; 
  typedef boost::asio::local::stream_protocol::socket asio_alss_t;
  typedef shared_ptr< asio_alss_t > p_asio_alss_t; 
  boost::asio::io_service & get_io( disp_win_t * const dw );

  template< typename STREAM, typename T > inline void bwrite( STREAM & out, T const & o ) { 
    write( out, boost::asio::buffer( (char *)&o, sizeof(o) ) ); }
  template< typename STREAM, typename T > inline void bread( STREAM & in, T & o ) { 
    read( in, boost::asio::buffer( (char *)&o, sizeof(o) ) ); }
  template< typename STREAM > inline void bwrite( STREAM & out, string const & o ) {
    uint32_t const sz = o.size();
    bwrite( out, sz );
    write( out, boost::asio::buffer( (char *)&o[0], o.size()*sizeof(string::value_type) ) );
  }
  template< typename STREAM > inline void bread( STREAM & in, string & o ) {
    uint32_t sz = 0;
    bread( in, sz );
    o.resize( sz );
    read( in, boost::asio::buffer( (char *)&o[0], o.size()*sizeof(string::value_type) ) );
  }

  template< typename STREAM > p_uint8_t recv_shared_p_uint8_t( STREAM & in ) {
    string fn;
    bread( in, fn ); // note: currently always == get_boda_shm_filename() ...
    uint32_t sz = 0;
    bread( in, sz );
    assert_st( sz );
    int const fd = shm_open( fn.c_str(), O_RDWR, S_IRUSR | S_IWUSR );
    if( fd == -1 ) { rt_err( strprintf( "recv-end shm_open() failed with errno=%s", str(errno).c_str() ) ); }
    neg_one_fail( ftruncate( fd, sz ), "ftruncate" );
    p_uint8_t ret = make_mmap_shared_p_uint8_t( fd, sz, 0 );
    uint8_t const done = 1;
    bwrite( in, done );
    return ret;
  }

  template< typename STREAM > p_uint8_t make_and_share_p_uint8_t( STREAM & out, uint32_t const sz ) {
    string const fn = get_boda_shm_filename();
    shm_unlink( fn.c_str() ); // we don't want to use any existing shm, so try to remove it if it exists.
    // note that, in theory, we could have just unlinked an in-use shm, and thus broke some other
    // processes. also note that between the unlink above and the shm_open() below, someone could
    // create shm with the name we want, and then we will fail. note that, generally speaking,
    // shm_open() doesn't seem secure/securable (maybe one could use a random shm name here?), but
    // we're mainly trying to just be robust -- including being non-malicious
    // errors/bugs/exceptions.
    int const fd = shm_open( fn.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_EXCL, S_IRUSR | S_IWUSR );
    if( fd == -1 ) { rt_err( strprintf( "send-end shm_open() failed with errno=%s", str(errno).c_str() ) ); }
    neg_one_fail( ftruncate( fd, sz ), "ftruncate" );
    p_uint8_t ret = make_mmap_shared_p_uint8_t( fd, sz, 0 );
    bwrite( out, fn ); 
    bwrite( out, sz );
    uint8_t done;
    bread( out, done );
    // we're done with the shm segment name now, so free it. notes: (1) we could have freed it
    // earlier and used SCM_RIGHTS to xfer the fd. (2) we could make a better effort to unlink here
    // in the event of errors in the above xfer.
    neg_one_fail( shm_unlink( fn.c_str() ), "shm_unlink" );
    return ret;
  }
  struct cs_disp_t : virtual public nesi, public has_main_t // NESI(help="client-server video display test",
			  // bases=["has_main_t"], type_id="cs_disp")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    virtual void main( nesi_init_arg_t * nia ) { 
      int sp_fds[2];
      neg_one_fail( socketpair( AF_LOCAL, SOCK_STREAM, 0, sp_fds ), "socketpair" );
      set_fd_cloexec( sp_fds[0], 0 ); // we want the parent fd closed in our child
      // fork/exec
      fork_and_exec_self( {"boda","display_ipc", strprintf("--boda-parent-socket-fd=%s",str(sp_fds[1]).c_str()) } );
      neg_one_fail( close( sp_fds[1] ), "close" ); // in the parent, we close the socket child will use

      boost::asio::io_service io;
      asio_alss_t alss( io ); 
      alss.assign( boost::asio::local::stream_protocol(), sp_fds[0] );

      p_img_t img( new img_t );
      //img->set_sz_and_alloc_pels( 100, 100 );
      img->set_sz( 100, 100 );
      img->pels = make_and_share_p_uint8_t( alss, img->sz_raw_bytes() ); // FIXME: check img->row_align wrt map page sz?
      img->fill_with_pel( grey_to_pel( 128 ) );

      uint8_t a_byte = 123;

      for( uint32_t j = 0; j != 10000; ++j ) {
	for( uint32_t i = 0; i != 10; ++i ) {
	  img->fill_with_pel( grey_to_pel( 50 + i*15 ) );
	  bwrite( alss, string( "foo" ) );
	  //write( alss, boost::asio::buffer( &a_byte, 1 ) ); 	  
	  read( alss, boost::asio::buffer( &a_byte, 1 ) );
	}
      }

    }
  };

  struct display_ipc_t : virtual public nesi, public has_main_t // NESI(help="video display over ipc test",
			  // bases=["has_main_t"], type_id="display_ipc")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    int32_t boda_parent_socket_fd; //NESI(help="an open fd created by socketpair() in the parent process.",req=1)

    disp_win_t disp_win;
    p_asio_alss_t alss;
    uint8_t a_byte;

    void on_parent_data( error_code const & ec ) { 
      assert_st( !ec );
      disp_win.update_disp_imgs();
      string ret;
      bread( *alss, ret );
      printf( "ret=%s\n", str(ret).c_str() );
      write( *alss, boost::asio::buffer( &a_byte, 1 ) ); 	  
      async_read( *alss, boost::asio::null_buffers(), bind( &display_ipc_t::on_parent_data, this, _1 ) );
    }

    virtual void main( nesi_init_arg_t * nia ) { 
      //fork();
      boost::asio::io_service & io( get_io( &disp_win ) );
      alss.reset( new asio_alss_t(io)  );
      alss->assign( boost::asio::local::stream_protocol(), boda_parent_socket_fd );

      p_img_t img( new img_t );
      //img->set_sz_and_alloc_pels( 100, 100 );
      img->set_sz( 100, 100 );
      img->pels = recv_shared_p_uint8_t( *alss ); // FIXME: check img->row_align wrt map page sz?
      img->fill_with_pel( grey_to_pel( 128 ) );

      disp_win.disp_setup( img );

      async_read( *alss, boost::asio::null_buffers(), bind( &display_ipc_t::on_parent_data, this, _1 ) );

      io.run();
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
    disp_win_t disp_win;
    void on_cap_read( error_code const & ec ) { 
      assert_st( !ec );
      capture->on_readable();
      cnet_predict->do_predict( capture->cap_img ); 
      disp_win.update_disp_imgs();
      async_read( *cap_afd, boost::asio::null_buffers(), bind( &capture_classify_t::on_cap_read, this, _1 ) );
    }
    virtual void main( nesi_init_arg_t * nia ) { 
      cnet_predict->setup_predict(); 
      capture->cap_start();
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
    disp_win_t disp_win;
    void on_cap_read( error_code const & ec ) { 
      assert_st( !ec );
      capture->on_readable();

      subtract_mean_and_copy_img_to_batch( run_cnet->in_batch, 0, capture->cap_img );
      p_nda_float_t out_batch = run_cnet->run_one_blob_in_one_blob_out();
      copy_batch_to_img( out_batch, 0, feat_img );
      disp_win.update_disp_imgs();
      async_read( *cap_afd, boost::asio::null_buffers(), bind( &capture_feats_t::on_cap_read, this, _1 ) );
    }
    virtual void main( nesi_init_arg_t * nia ) { 
      run_cnet->in_sz = capture->cap_res;
      run_cnet->setup_cnet(); 
      feat_img.reset( new img_t );
      u32_pt_t const feat_img_sz = run_cnet->get_one_blob_img_out_sz();
      feat_img->set_sz_and_alloc_pels( feat_img_sz.d[0], feat_img_sz.d[1] ); // w, h

      capture->cap_start();
      disp_win.disp_setup( vect_p_img_t{feat_img,capture->cap_img} );

      boost::asio::io_service & io = get_io( &disp_win );
      cap_afd.reset( new asio_fd_t( io, ::dup(capture->get_fd() ) ) );
      async_read( *cap_afd, boost::asio::null_buffers(), bind( &capture_feats_t::on_cap_read, this, _1 ) );
      io.run();
    }
  };


#include"gen/cap_app.cc.nesi_gen.cc"

}
