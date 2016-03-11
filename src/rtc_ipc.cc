// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"rtc_compute.H"
#include"has_main.H"
#include"asio_util.H"
#include"rand_util.H"
#include"timers.H"


#include<fcntl.h>
#include<stdio.h>
#include<boost/iostreams/device/file_descriptor.hpp>
#include<boost/iostreams/stream.hpp>

#include<netdb.h>
#include<sys/types.h>
#include<sys/socket.h>
#include<netinet/tcp.h>

namespace boda 
{
  template< typename STREAM > inline void bwrite( STREAM & out, rtc_func_call_t const & o ) { 
    bwrite( out, o.rtc_func_name );
    bwrite( out, o.in_args );
    bwrite( out, o.inout_args );
    bwrite( out, o.out_args );
    bwrite( out, o.u32_args );
    bwrite( out, o.call_tag );
    bwrite( out, o.tpb.v );
    bwrite( out, o.blks.v );
    bwrite( out, o.call_id );
  }
  template< typename STREAM > inline void bread( STREAM & in, rtc_func_call_t & o ) { 
    bread( in, o.rtc_func_name );
    bread( in, o.in_args );
    bread( in, o.inout_args );
    bread( in, o.out_args );
    bread( in, o.u32_args );
    bread( in, o.call_tag );
    bread( in, o.tpb.v );
    bread( in, o.blks.v );
    bread( in, o.call_id );
  }

  struct ipc_var_info_t {
    p_nda_float_t buf;
    dims_t dims;
    ipc_var_info_t( dims_t const & dims_ ) : dims(dims_) {
      buf.reset( new nda_float_t( dims ) );
    }
  };

  typedef map< string, ipc_var_info_t > map_str_ipc_var_info_t;
  typedef shared_ptr< map_str_ipc_var_info_t > p_map_str_ipc_var_info_t;

  namespace io = boost::iostreams;


  struct fd_stream_t {
    io::stream<io::file_descriptor_source> r;
    io::stream<io::file_descriptor_sink> w;
    fd_stream_t( string const & boda_parent_addr, bool const & is_worker ) { init( boda_parent_addr, is_worker ); }
    void init( string const & boda_parent_addr, bool const & is_worker ) {
      vect_string bpa_parts = split( boda_parent_addr, ':' );
      if( bpa_parts.size() != 3 ) { rt_err( "boda_parent_addr must consist of three ':' seperated fields, in the form method:to_parent:to_worker"
					    " where method is 'fns' of 'fds' (sorry, no ':' allowed in filenames for the fns method)." ); }
      // format is method:to_parent:to_worker
      int read_fd = -1;
      int write_fd = -1;
      
      if( bpa_parts[0] == "fns" ) {
	string pfn = bpa_parts[1];
	string wfn = bpa_parts[2];
	if( is_worker ) {
	  neg_one_fail( read_fd = open( wfn.c_str(), O_RDONLY ), "open" ); 
	  neg_one_fail( write_fd = open( pfn.c_str(), O_WRONLY ), "open" );
	} else {
	  neg_one_fail( write_fd = open( wfn.c_str(), O_WRONLY ), "open" );
	  neg_one_fail( read_fd = open( pfn.c_str(), O_RDONLY ), "open" ); 
	}
      } else if( bpa_parts[0] == "fds" ) {
	read_fd = lc_str_u32( bpa_parts[is_worker?2:1] );
	write_fd = lc_str_u32( bpa_parts[is_worker?1:2] );
      } else { rt_err( "unknown boda_parent_addr type %s, should be either 'fns' (filenames) or 'fds' (open file descriptor integers)" ); }

      r.open( io::file_descriptor_source( read_fd, io::never_close_handle ) );
      w.open( io::file_descriptor_sink( write_fd, io::never_close_handle ) );
    }
    void write( char const * const & d, size_t const & sz ) { w.write( d, sz ); }
    void read( char * const & d, size_t const & sz ) { r.read( d, sz ); }
    bool good( void ) { return r.good() && w.good(); }
    void flush( void ) { w.flush(); }
    typedef void pos_type; // flag class as IOStream-like for metaprogramming/template conditionals in boda_base.H
  };
  typedef shared_ptr< fd_stream_t > p_fd_stream_t; 

  // streaming IO using tcp sockets. notes: our buffering/flushing strategy is MSG_MORE + redundant set of TCP_NODELAY
  // -- it's unclear how portable this is, but see the below SO post for some somewhat-linux-specific. for now, we
  // mostly care about linux and android, so i guess we'll see how android goes.
  // http://stackoverflow.com/questions/2547956/flush-kernels-tcp-buffer-for-msg-more-flagged-packets
  void setopt( int const fd, int const level, int const optname, int const optval = 1 ) {
    int const ret = setsockopt( fd, level, optname, &optval, sizeof(optval) );
    if( ret != 0 ) { rt_err(  strprintf( "setsockopt(level=%s,optname=%s) error: %s", 
					 str(level).c_str(), str(optname).c_str(), strerror( errno ) ) ); }
  }
  struct sock_stream_t {
    int fd; 
    int listen_fd; // listen_fd is only used on server (bind/listen/accept) side of connection. technically we only need
		   // one fd at a time, but it seems less confusing to have two explict ones for the two usages.
    sock_stream_t( void ) : fd(-1), listen_fd(-1) { }
    // ah, TCP programming. to help understand the dance and all it's idioms/gotchas/tweaks, it's nice to reference
    // one-or-more existing blocks of code. and then pray you don't see to go examining kernel source or caring too much
    // about portability across OSs or version. but hey, it's not so bad, and the man pages are generally accurate and
    // all. sigh. anyway, ZMQ is a nice place to look, see:
    // https://github.com/zeromq/libzmq/blob/master/src/tcp_listener.cpp 

    // in general, bind+listen/accept are used on the 'server' side, connect is used on the 'client' side.  in
    // general, we set up the listening socket with bind_and_listen(), then spawn a worker process, then accept() an incoming
    // connection from the worker. since the network stack will buffer a backlog of connection attempts for us, there's
    // no race. for now, we only listen for a single worker per sock_stream_t.
    string bind_and_listen( string const & port ) {
      assert_st( fd == -1 );
      assert_st( listen_fd == -1 );

      addrinfo * rp_bind_addrs = 0;
      addrinfo hints = {0};
      hints.ai_socktype = SOCK_STREAM; // want reliable communication, 
      hints.ai_protocol = IPPROTO_TCP; // ... with  TCP (minimally because we try to set TCP_NODELAY) ...
      // hints.ai_family = AF_INET; // but allow any family (IPv4 or IPv6). or, uncomment to force IPv4
      hints.ai_flags = AI_PASSIVE; // for binding to wildcard addr

      int const aret = getaddrinfo( 0, port.c_str(), &hints, &rp_bind_addrs );
      if( aret != 0 ) { rt_err( strprintf("getaddrinfo with port %s failed: %s", port.c_str(), gai_strerror( aret ) ) ); }
      addrinfo * rpa = rp_bind_addrs; // for now, only try first returned addr
      listen_fd = socket( rpa->ai_family, rpa->ai_socktype, rpa->ai_protocol );
      if( listen_fd == -1 ) { rt_err( string("error creating socket for listen: ") + strerror(errno) ); }
      setopt( listen_fd, SOL_SOCKET, SO_REUSEADDR ); // 'helps' with avoiding can't-bind-port after crashes, etc ... see SO posts

      int const bret = bind( listen_fd, rpa->ai_addr, rpa->ai_addrlen );
      if( bret != 0 ) { rt_err( strprintf( "bind to port %s failed: %s", port.c_str(), strerror( errno ) ) ); }
      int const lret = listen( listen_fd, 1 );
      if( lret != 0 ) { rt_err( strprintf( "listen on port %s failed: %s", port.c_str(), strerror( errno ) ) ); }

      // get and return our (the parent's) addr to pass to the worker so it can connect back
      char hbuf[NI_MAXHOST+1] = {0};
      int const ghnret = gethostname( hbuf, NI_MAXHOST );
      if( ghnret != 0 ) { rt_err( strprintf( "post-good-bind gethostname() failed: %s", strerror( errno ) ) ); }
      if( strlen(hbuf)==NI_MAXHOST ) { rt_err( strprintf( "post-good-bind gethostname() failed: name too long" ) ); }
      string const parent_addr = string(hbuf) + ":" + port;
      freeaddrinfo( rp_bind_addrs );
      return parent_addr;
    }
    void accept_and_stop_listen( void ) {
      int const aret = accept( listen_fd, 0, 0 );
      if( aret == -1 ) { rt_err( strprintf( "accept failed: %s", strerror( errno ) ) ); }
      // we're done listening now, so we can close the listening socket
      int const cret = close( listen_fd );
      if( cret == -1 ) { rt_err( strprintf( "post-accept close of listening socket failed: %s", strerror( errno ) ) ); }
      listen_fd = -1;
      assert( fd == -1 );
      fd = aret; // use the socket returned from accept() for communicaiton
      // set TCP_NODELAY here at socket creation time, so the socket is in a consistent state for all writes wrt this
      // option (as opposed to the having just the write before the first flush have TCP_NODELAY set).
      flush(); 
    }
    void connect_to_parent( string const & parent_addr ) {
      assert_st( listen_fd == -1 );
      vect_string host_and_port = split( parent_addr, ':' );
      assert_st( host_and_port.size() == 2 ); // host:port
      addrinfo * rp_connect_addrs = 0;
      addrinfo hints = {0};
      hints.ai_socktype = SOCK_STREAM; // better be ... right? allow any family or protocol, though.
      int const aret = getaddrinfo( host_and_port[0].c_str(), host_and_port[1].c_str(), &hints, &rp_connect_addrs );
      if( aret != 0 ) { rt_err( strprintf("getaddrinfo(\"%s\") failed: %s", parent_addr.c_str(), gai_strerror( aret ) ) ); }
      addrinfo * rpa = rp_connect_addrs; // for now, only try first returned addr
      assert_st( fd == -1 );
      fd = socket( rpa->ai_family, rpa->ai_socktype, rpa->ai_protocol );
      if( fd == -1 ) { rt_err( string("error creating socket for connect: ") + strerror(errno) ); }
      int const ret = connect( fd, rpa->ai_addr, rpa->ai_addrlen );
      if( ret != 0 ) { rt_err( strprintf( "connect to %s failed: %s", str(parent_addr).c_str(), strerror( errno ) ) ); }
      flush(); // see above comment in accept_and_stop_listen()
      freeaddrinfo( rp_connect_addrs );
    }
    void write( char const * const & d, size_t const & sz ) { 
      size_t sz_written = 0;
      assert_st( sz );
      while( sz_written < sz ) { 
	int const ret = send( fd, d + sz_written, sz - sz_written, MSG_NOSIGNAL | MSG_MORE );
	if( ret == -1 ) { if( errno == EINTR ) { continue; } else { rt_err( string("socket write error: ") + strerror( errno ) ); } }
	assert_st( ret > 0 ); // FIXME: other returns possible? make into rt_err() call?
	sz_written += ret;
      }
    }
    void read( char * const & d, size_t const & sz ) { 
      size_t sz_read = 0;
      assert_st( sz );
      while( sz_read < sz ) { 
	int const ret = recv( fd, d + sz_read, sz - sz_read, 0 );
	if( ret == -1 ) { if( errno == EINTR ) { continue; } else { rt_err( string("socket read error: ") + strerror( errno ) ); } }
	assert_st( ret > 0 ); // FIXME: other returns possible? make into rt_err() call?
	sz_read += ret;
      }
    }
    bool good( void ) { return 1; } // it's all good, all the time. hmm.
    void flush( void ) { setopt( fd, IPPROTO_TCP, TCP_NODELAY ); }
    typedef void pos_type; // flag class as IOStream-like for metaprogramming/template conditionals in boda_base.H
  };
  typedef shared_ptr< sock_stream_t > p_sock_stream_t; 


  struct ipc_compute_t : virtual public nesi, public rtc_compute_t // NESI(help="rtc-over-IPC wrapper/server",
			   // bases=["rtc_compute_t"], type_id="ipc" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    zi_bool init_done;
    string remote_rtc; //NESI(default="(be=ocl)",help="remote rtc configuration")
    p_string fifo_fn; //NESI(help="if set, use a named fifo for communication instead of a socketpair.")
    uint32_t print_dont_fork; //NESI(default=0,help="if set, don't actually fork to create a fifo-based worker, just print the command to do so.")

    p_map_str_ipc_var_info_t vis;

    p_fd_stream_t worker;

    void init( void ) {
      assert_st( !init_done.v );
      vis.reset( new map_str_ipc_var_info_t );

      string bpa;
      if( !fifo_fn ) {
	int const worker_fd = create_boda_worker( {"boda","ipc_compute_worker","--rtc="+remote_rtc} );
        bpa = strprintf("fds:%s:%s", str(worker_fd).c_str(), str(worker_fd).c_str() );
      } else {
	bpa = create_boda_worker_fifo( {"boda","ipc_compute_worker","--rtc="+remote_rtc}, *fifo_fn, print_dont_fork );
      }
      worker.reset( new fd_stream_t( bpa, 0 ) );	

      bwrite( *worker, string("init") );
      worker->flush();

      init_done.v = 1;
    }

    ~ipc_compute_t( void ) {
      if( init_done.v ) {
	bwrite( *worker, string("quit") );
	worker->flush();
      }
    }

    void compile( string const & cucl_src, bool const show_compile_log, bool const enable_lineinfo ) {
      bwrite( *worker, string("compile") ); bwrite( *worker, cucl_src ); bwrite( *worker, show_compile_log ); bwrite( *worker, enable_lineinfo ); 
      worker->flush();
    }
    void copy_to_var( string const & vn, float const * const v ) {
      uint32_t const sz = get_var_sz_floats( vn );
      bwrite( *worker, string("copy_to_var") ); bwrite( *worker, vn ); bwrite_bytes( *worker, (char const *)v, sz*sizeof(float) ); 
      worker->flush();
    }
    void copy_from_var( float * const v, string const & vn ) {
      uint32_t const sz = get_var_sz_floats( vn );
      bwrite( *worker, string("copy_from_var") ); bwrite( *worker, vn ); 
      worker->flush();
      bread_bytes( *worker, (char *)v, sz*sizeof(float) ); 
    }
    void create_var_with_dims_floats( string const & vn, dims_t const & dims ) { 
      must_insert( *vis, vn, ipc_var_info_t{dims} ); 
      bwrite( *worker, string("create_var_with_dims_floats") ); bwrite( *worker, vn ); bwrite( *worker, dims ); 
      worker->flush();
    }
    dims_t get_var_dims_floats( string const & vn ) { return must_find( *vis, vn ).dims; }
    void set_var_to_zero( string const & vn ) { bwrite( *worker, string("set_var_to_zero") ); bwrite( *worker, vn ); worker->flush(); }
    

    // note: post-compilation, MUST be called exactly once on all functions that will later be run()
    void check_runnable( string const name, bool const show_func_attrs ) {
      bwrite( *worker, string("check_runnable") ); bwrite( *worker, name ); bwrite( *worker, show_func_attrs ); worker->flush();
    }

    virtual float get_dur( uint32_t const & b, uint32_t const & e ) { 
      float ret;
      bwrite( *worker, string("get_dur") ); bwrite( *worker, b ); bwrite( *worker, e ); worker->flush(); bread( *worker, ret );
      return ret; 
    } 
    virtual float get_var_compute_dur( string const & vn ) { assert_st(0); } // not-yet-used-iface at higher level
    virtual float get_var_ready_delta( string const & vn1, string const & vn2 ) { assert_st(0); } // not-yet-used-iface at higher level

    void run( rtc_func_call_t & rfc ) { 
      bwrite( *worker, string("run") ); bwrite( *worker, rfc ); worker->flush(); bread( *worker, rfc.call_id );
    } 

    void finish_and_sync( void ) { bwrite( *worker, string("finish_and_sync") ); worker->flush(); }
    void release_per_call_id_data( void ) { bwrite( *worker, string("release_per_call_id_data") ); worker->flush(); }

    void profile_start( void ) { bwrite( *worker, string("profile_start") ); worker->flush(); }
    void profile_stop( void ) { bwrite( *worker, string("profile_stop") ); worker->flush(); }
  };


  struct cs_test_master_t : virtual public nesi, public has_main_t // NESI(help="cs-testing master/server", bases=["has_main_t"], type_id="cs_test_master")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string port; //NESI(default="12791", help="service/port to use for network (TCP) communication")
    p_sock_stream_t worker;
    virtual void main( nesi_init_arg_t * nia ) { 
      worker.reset( new sock_stream_t );
      string const parent_addr = worker->bind_and_listen( port );
      printf( "boda_master: listening on parent_addr=%s\n", str(parent_addr).c_str() );      
      printf( "boda_master: entering accept_and_stop_listen() ... \n" );      
      worker->accept_and_stop_listen();
      printf( "boda_master: connected to worker.\n" );
      vect_string cmds{ "giggle", "quit" };
      for( vect_string::const_iterator i = cmds.begin(); i != cmds.end(); ++i ) {
	string const & cmd = *i;
	bwrite( *worker, cmd );
	printf( "boda_master: sent cmd=%s\n", str(cmd).c_str() );	
      }
    }    
  };

  struct cs_test_worker_t : virtual public nesi, public has_main_t // NESI(help="cs-testing worker/client", bases=["has_main_t"], type_id="cs_test_worker")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string parent_addr; //NESI(help="how to communicate with boda parent process; network address in the form host:port",req=1)
    p_sock_stream_t parent;
    virtual void main( nesi_init_arg_t * nia ) { 
      parent.reset( new sock_stream_t );
      printf( "boda_worker: connecting to parent_addr=%s\n", str(parent_addr).c_str() );      
      parent->connect_to_parent( parent_addr );
      printf( "boda_worker: connected to parent.\n" );
      string cmd;
      while( 1 ) {
	bread( *parent, cmd );
	printf( "boda_worker: got cmd=%s\n", str(cmd).c_str() );
	if( 0 ) {} 
	else if( cmd == "quit" ) { break; }
	else if( cmd == "giggle" ) { printf( "boda_worker: tee hee hee.\n" ); }
      }
    } 
  };

  struct ipc_compute_worker_t : virtual public nesi, public has_main_t // NESI(help="rtc-over-IPC worker/client", bases=["has_main_t"], type_id="ipc_compute_worker")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    p_rtc_compute_t rtc; //NESI(default="(be=ocl)",help="rtc back-end to use")

    string boda_parent_addr; //NESI(help="how to communicate with boda parent process; either open fds (perhaps created by socketpair() in the parent process, or perhaps stdin/stdout), or the names of a pair of named files/fifos to open.",req=1)

    p_map_str_ipc_var_info_t vis;

    p_img_t in_img;
    p_img_t out_img;

    uint8_t proc_done;
    
    p_fd_stream_t parent;

    ipc_compute_worker_t( void ) : proc_done(1) { }

    virtual void main( nesi_init_arg_t * nia ) { 
      global_timer_log_set_disable_finalize( 1 );

      vis.reset( new map_str_ipc_var_info_t );
      parent.reset( new fd_stream_t( boda_parent_addr, 1 ) );

      string cmd;
      while( 1 ) {
	bread( *parent, cmd );
	if( 0 ) {} 
	else if( cmd == "init" ) { rtc->init(); }
	else if( cmd == "quit" ) { break; }
	else if( cmd == "compile" ) {
	  string cucl_src; bool show_compile_log; bool enable_lineinfo;
	  bread( *parent, cucl_src ); bread( *parent, show_compile_log ); bread( *parent, enable_lineinfo );
	  rtc->compile( cucl_src, show_compile_log, enable_lineinfo );
	}
	else if( cmd == "copy_to_var" ) {
	  string vn;
	  bread( *parent, vn );
	  ipc_var_info_t & vi = must_find( *vis, vn );
	  uint32_t const sz = rtc->get_var_sz_floats( vn );
	  assert_st( sz == vi.buf->elems.sz );
	  bread_bytes( *parent, (char *)&vi.buf->elems[0], sz*sizeof(float) ); 
	  rtc->copy_to_var( vn, &vi.buf->elems[0] );
	}
	else if( cmd == "copy_from_var" ) {
	  string vn;
	  bread( *parent, vn );
	  ipc_var_info_t & vi = must_find( *vis, vn );
	  uint32_t const sz = rtc->get_var_sz_floats( vn );
	  assert_st( sz == vi.buf->elems.sz );
	  rtc->copy_from_var( &vi.buf->elems[0], vn );
	  bwrite_bytes( *parent, (char const *)&vi.buf->elems[0], sz*sizeof(float) ); 
	  parent->flush();
	}
	else if( cmd == "create_var_with_dims_floats" ) {
	  string vn; dims_t dims;
	  bread( *parent, vn ); 
	  bread( *parent, dims );
	  must_insert( *vis, vn, ipc_var_info_t{dims} );
	  rtc->create_var_with_dims_floats( vn, dims );
	}
	else if( cmd == "set_var_to_zero" ) {
	  string vn;
	  bread( *parent, vn );
	  rtc->set_var_to_zero( vn );
	}
	else if( cmd == "check_runnable" ) { 
	  string name; bool show_func_attrs;
	  bread( *parent, name ); bread( *parent, show_func_attrs );
	  rtc->check_runnable( name, show_func_attrs );
	}
	else if( cmd == "get_dur" ) { 
	  uint32_t b,e; bread( *parent, b ); bread( *parent, e ); 
	  float const ret = rtc->get_dur( b, e ); 
	  bwrite( *parent, ret ); parent->flush(); 
	}
	else if( cmd == "run" ) { rtc_func_call_t rfc; bread( *parent, rfc ); rtc->run( rfc ); bwrite( *parent, rfc.call_id ); parent->flush(); }
	else if( cmd == "finish_and_sync" ) { rtc->finish_and_sync(); }
	else if( cmd == "profile_start" ) { rtc->profile_start(); }
	else if( cmd == "profile_stop" ) { rtc->profile_stop(); }
	else if( cmd == "release_per_call_id_data" ) { rtc->release_per_call_id_data(); }
	else { rt_err("bad command:"+cmd); }
      }
    }
  };

#include"gen/rtc_ipc.cc.nesi_gen.cc"
}
