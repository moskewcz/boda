// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"rtc_compute.H"
#include"has_main.H"
#include"asio_util.H"
#include"rand_util.H"
#include"timers.H"
#include"lexp.H"

#include<fcntl.h>
#include<stdio.h>
#include<boost/iostreams/device/file_descriptor.hpp>
#include<boost/iostreams/stream.hpp>
#include<boost/program_options/parsers.hpp> // for split_unix()

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
    bwrite( out, o.nda_args );
    bwrite( out, o.has_cucl_arg_info.v );
    bwrite( out, o.cucl_arg_info );
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
    bread( in, o.nda_args );
    bread( in, o.has_cucl_arg_info.v );
    bread( in, o.cucl_arg_info );
    bread( in, o.call_tag );
    bread( in, o.tpb.v );
    bread( in, o.blks.v );
    bread( in, o.call_id );
  }

  template< typename STREAM > inline void bwrite( STREAM & out, op_base_t const & o ) { 
    bwrite( out, o.type );
    bwrite( out, o.dims_vals );
    bwrite( out, o.str_vals );
  }
  template< typename STREAM > inline void bread( STREAM & in, op_base_t & o ) { 
    bread( in, o.type );
    bread( in, o.dims_vals );
    bread( in, o.str_vals );
  }

  template< typename STREAM > inline void bwrite( STREAM & out, rtc_func_info_t const & o ) { 
    bwrite( out, o.func_name );
    bwrite( out, o.op );
  }
  template< typename STREAM > inline void bread( STREAM & in, rtc_func_info_t & o ) { 
    bread( in, o.func_name );
    bread( in, o.op );
  }

  struct ipc_var_info_t {
    p_nda_t buf;
    dims_t dims;
    ipc_var_info_t( dims_t const & dims_ ) : dims(dims_) { buf = make_shared<nda_t>( dims ); }
  };

  typedef map< string, ipc_var_info_t > map_str_ipc_var_info_t;
  typedef shared_ptr< map_str_ipc_var_info_t > p_map_str_ipc_var_info_t;

  namespace io = boost::iostreams;

  // ABC so we can abstract over fd_stream_t and sock_stream_t inside the ipc master and worker note that the
  // template-based send/recv code doesn't know or care about this ABC in particular; it will (try to) use any class
  // with the pos_type marker typedef.
  struct stream_t {
    virtual void wait_for_worker( void ) = 0; // for setup, used only in server
    // used for stream communication by templates in boda_base.H
    virtual void write( char const * const & d, size_t const & sz ) = 0;
    virtual void read( char * const & d, size_t const & sz ) = 0;
    virtual bool good( void ) = 0;
    virtual void flush( void ) = 0;
    typedef void pos_type; // flag class as IOStream-like for metaprogramming/template conditionals in boda_base.H
  };
  struct stream_t; typedef shared_ptr< stream_t > p_stream_t; 

  struct fd_stream_t : public stream_t {
    virtual void wait_for_worker( void ) { 
      if( method == "fns" ) { 
	assert_st( read_fd == -1 );
	neg_one_fail( write_fd = open( wfn.c_str(), O_WRONLY ), "open" );
	neg_one_fail( read_fd = open( pfn.c_str(), O_RDONLY ), "open" ); 
	open_streams();
      }
      else if( method == "fds" ) { } // note: no sync needed for "fds" case
      else { rt_err( "fd_stream_t::wait_for_worker(): internal error: unknown method: " + method ); }
    };
    string method;
    string pfn;
    string wfn;
    int read_fd;
    int write_fd;
    io::stream<io::file_descriptor_source> r;
    io::stream<io::file_descriptor_sink> w;
    fd_stream_t( string const & boda_parent_addr, bool const & is_worker ) { init( boda_parent_addr, is_worker ); }
    void init( string const & boda_parent_addr, bool const & is_worker ) {
      vect_string bpa_parts = split( boda_parent_addr, ':' );
      if( bpa_parts.size() != 3 ) { rt_err( "boda_parent_addr must consist of three ':' seperated fields, in the form method:to_parent:to_worker"
					    " where method is 'fns' of 'fds' (sorry, no ':' allowed in filenames for the fns method)." ); }
      // format is method:to_parent:to_worker
      read_fd = -1;
      write_fd = -1;
      method = bpa_parts[0];
      if( method == "fns" ) {
	pfn = bpa_parts[1];
	wfn = bpa_parts[2];
	if( is_worker ) {
	  neg_one_fail( read_fd = open( wfn.c_str(), O_RDONLY ), "open" ); 
	  neg_one_fail( write_fd = open( pfn.c_str(), O_WRONLY ), "open" );
	  open_streams();
	}
	// see wait_for_worker() for opens in !is_worker case
      } else if( bpa_parts[0] == "fds" ) {
	read_fd = lc_str_u32( bpa_parts[is_worker?2:1] );
	write_fd = lc_str_u32( bpa_parts[is_worker?1:2] );
	open_streams();
      } else { rt_err( "unknown boda_parent_addr type "+bpa_parts[0]+", should be either 'tcp' (tcp connection to host:port), 'fns' (filenames) or 'fds' (open file descriptor integers)" ); }
    }
    void open_streams( void ) {
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
  struct sock_stream_t : public stream_t {
    int fd; 
    int listen_fd; // listen_fd is only used on server (bind/listen/accept) side of connection. technically we only need
		   // one fd at a time, but it seems less confusing to have two explict ones for the two usages.
    sock_stream_t( void ) : fd(-1), listen_fd(-1) { }
    sock_stream_t( string const & boda_parent_addr, bool const & is_worker ) : fd(-1), listen_fd(-1) { 
      vect_string host_and_port = split( boda_parent_addr, ':' );
      assert_st( host_and_port[0] == "tcp" ); // should not be here otherwise
      if( host_and_port.size() != 3 ) { rt_err( "for the tcp method, boda_parent_addr must consist of three ':' seperated fields,"
						" in the form tcp:HOST:PORT (where PORT may be a non-numeric service name)" ); }
      if( !is_worker ) { bind_and_listen( host_and_port[2] ); }
      else { connect_to_parent( boda_parent_addr ); }
    }
    virtual void wait_for_worker( void ) { accept_and_stop_listen(); }

#if 0
      char hbuf[NI_MAXHOST+1] = {0};
      int const ghnret = gethostname( hbuf, NI_MAXHOST );
      if( ghnret != 0 ) { rt_err( strprintf( "post-good-bind gethostname() failed: %s", strerror( errno ) ) ); }
      if( strlen(hbuf)==NI_MAXHOST ) { rt_err( strprintf( "post-good-bind gethostname() failed: name too long" ) ); }
      string const parent_addr = string(hbuf) + ":" + port;
#endif

    // ah, TCP programming. to help understand the dance and all it's idioms/gotchas/tweaks, it's nice to reference
    // one-or-more existing blocks of code. and then pray you don't see to go examining kernel source or caring too much
    // about portability across OSs or version. but hey, it's not so bad, and the man pages are generally accurate and
    // all. sigh. anyway, ZMQ is a nice place to look, see:
    // https://github.com/zeromq/libzmq/blob/master/src/tcp_listener.cpp 

    // in general, bind+listen/accept are used on the 'server' side, connect is used on the 'client' side.  in
    // general, we set up the listening socket with bind_and_listen(), then spawn a worker process, then accept() an incoming
    // connection from the worker. since the network stack will buffer a backlog of connection attempts for us, there's
    // no race. for now, we only listen for a single worker per sock_stream_t.
    void bind_and_listen( string const & port ) {
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
      freeaddrinfo( rp_bind_addrs );
    }
    void accept_and_stop_listen( void ) {
      assert( listen_fd != -1 );
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
      assert_st( host_and_port.size() == 3 ); // tcp:host:port
      addrinfo * rp_connect_addrs = 0;
      addrinfo hints = {0};
      hints.ai_socktype = SOCK_STREAM; // better be ... right? allow any family or protocol, though.
      int const aret = getaddrinfo( host_and_port[1].c_str(), host_and_port[2].c_str(), &hints, &rp_connect_addrs );
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

  p_stream_t make_stream_t( string const & boda_parent_addr, bool const & is_worker ) {
    string method = split( boda_parent_addr, ':' )[0];
    if( method == "tcp" ) { 
      
      return p_stream_t( new sock_stream_t( boda_parent_addr, is_worker ) ); 

    }
    else { return p_stream_t( new fd_stream_t( boda_parent_addr, is_worker ) ); }
  }


  struct ipc_compute_t : virtual public nesi, public rtc_compute_t // NESI(help="rtc-over-IPC wrapper/server",
			   // bases=["rtc_compute_t"], type_id="ipc" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    zi_bool init_done;
    string boda_parent_addr; //NESI(default="", help="address to use for communication. FIXME: document.")
    string remote_rtc; //NESI(default="(be=ocl)",help="remote rtc configuration")
    p_string fifo_fn; //NESI(help="if set, use a named fifo for communication instead of a socketpair.")
    uint32_t print_dont_fork; //NESI(default=0,help="if set, don't actually fork to create a fifo-based worker, just print the command to do so.")
    p_string spawn_str; //NESI(help="command to spawn worker process, passed to os.system(). if not set, boda will use fork() to create a local worker. the worker's arguments will be appended.")
    uint32_t spawn_shell_escape_args; //NESI(default=0,help="if set, escape each worker arg suitably for use as a shell argument .")

    p_map_str_ipc_var_info_t vis;

    p_stream_t worker;

    void init( void ) {
      assert_st( !init_done.v );
      vis.reset( new map_str_ipc_var_info_t );

      vect_string worker_args{"boda","ipc_compute_worker","--rtc="+remote_rtc};
      
      string bpa;
      if( !boda_parent_addr.empty() ) {
	// new-and-approved flow: create stream first, then create worker process, ...
	worker = make_stream_t( boda_parent_addr, 0 );
	worker_args.push_back( "--boda-parent-addr="+boda_parent_addr );
        if( spawn_str ) {
          vect_string args = boost::program_options::split_unix( *spawn_str );
          if( spawn_shell_escape_args ) {
            for( vect_string::iterator i = worker_args.begin(); i != worker_args.end(); ++i ) {
              *i = shell_escape( *i );
            }
          } 
          args.insert( args.end(), worker_args.begin()+1, worker_args.end() ); // omit first arg 'boda'
          printf("final || delimted args to pass to execvpe():");
          for( vect_string::const_iterator i = args.begin(); i != args.end(); ++i ) {
            printf( " |%s|", str(*i).c_str() );
          }
          printf("\n");
          fork_and_exec_cmd( args );
        } else {
          if( print_dont_fork ) { 
            fprintf( stderr, "%s\n", join(worker_args," ").c_str());
          } else { 
            fork_and_exec_self( worker_args ); 
          }
        }
      } else if( !fifo_fn ) {
	// old-and-deprecated flow: create worker process, then create stream, ....
	int const worker_fd = create_boda_worker_socketpair( worker_args  );
        boda_parent_addr = strprintf("fds:%s:%s", str(worker_fd).c_str(), str(worker_fd).c_str() );	
	worker = make_stream_t( boda_parent_addr, 0 );
      } else {
	// old-and-deprecated flow: create worker process, then create stream, ....
	boda_parent_addr = create_boda_worker_fifo( worker_args, *fifo_fn, print_dont_fork );
	worker = make_stream_t( boda_parent_addr, 0 );
      }
      worker->wait_for_worker(); // ... then wait for worker.

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
    void compile( string const & cucl_src, bool const show_compile_log, bool const enable_lineinfo,
		  vect_rtc_func_info_t const & func_infos, bool const show_func_attrs ) {
      bwrite( *worker, string("compile") ); 
      bwrite( *worker, cucl_src ); bwrite( *worker, show_compile_log ); bwrite( *worker, enable_lineinfo ); 
      bwrite( *worker, func_infos ); bwrite( *worker, show_func_attrs ); 
      worker->flush();

      uint32_t ret = 0;
      string err_str;
      bread( *worker, ret ); // 0 --> no error
      if( ret ) { 
        bread( *worker, err_str );
        rt_err( "------BEGIN NESTED ERROR FROM IPC WORKER ------\n" + err_str 
                + "------END NESTED ERROR FROM IPC WORKER ------\n" );
      }
    }
    void copy_nda_to_var( string const & vn, p_nda_t const & nda ) {
      dims_t const & dims = get_var_dims( vn );
      assert_st( dims == nda->dims );
      bwrite( *worker, string("copy_nda_to_var") ); 
      bwrite( *worker, vn );
      bwrite( *worker, dims );
      bwrite_bytes( *worker, (char const *)nda->rp_elems(), dims.bytes_sz() ); 
      worker->flush();
    }
    void copy_var_to_nda( p_nda_t const & nda, string const & vn ) {
      dims_t const & dims = get_var_dims( vn );
      assert_st( dims == nda->dims );
      bwrite( *worker, string("copy_var_to_nda") ); 
      bwrite( *worker, vn );
      bwrite( *worker, dims );
      worker->flush();
      bread_bytes( *worker, (char *)nda->rp_elems(), dims.bytes_sz() ); 
    }
    void create_var_with_dims( string const & vn, dims_t const & dims ) { 
      must_insert( *vis, vn, ipc_var_info_t{dims} ); 
      bwrite( *worker, string("create_var_with_dims") ); bwrite( *worker, vn ); bwrite( *worker, dims ); 
      worker->flush();
    }
    void create_var_with_dims_as_reshaped_view_of_var( string const & vn, dims_t const & dims, string const & src_vn ) {
      must_insert( *vis, vn, ipc_var_info_t{dims} ); 
      bwrite( *worker, string("create_var_with_dims_as_reshaped_view_of_var") ); 
      bwrite( *worker, vn ); bwrite( *worker, dims ); bwrite( *worker, src_vn ); 
      worker->flush();
    }

    void release_var( string const & vn ) {
      must_erase( *vis, vn ); 
      bwrite( *worker, string("release_var") ); bwrite( *worker, vn );
      worker->flush();
    }
    dims_t get_var_dims( string const & vn ) { return must_find( *vis, vn ).dims; }
    void set_var_to_zero( string const & vn ) { bwrite( *worker, string("set_var_to_zero") ); bwrite( *worker, vn ); worker->flush(); }
    
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
    void release_all_funcs( void ) { bwrite( *worker, string("release_all_funcs") ); worker->flush(); }

    void profile_start( void ) { bwrite( *worker, string("profile_start") ); worker->flush(); }
    void profile_stop( void ) { bwrite( *worker, string("profile_stop") ); worker->flush(); }
  };

/* TESTING NOTES: automated tests for some configurations are TODO

# for testing fns mode, you need some FIFOs made with mkfifo:
moskewcz@maaya:~/git_work/boda/run/tr4$ mkfifo boda_fifo_to_parent
moskewcz@maaya:~/git_work/boda/run/tr4$ mkfifo boda_fifo_to_worker
moskewcz@maaya:~/git_work/boda/run/tr4$ ll
total 0
prw-rw-r-- 1 moskewcz moskewcz 0 Mar 11 17:07 boda_fifo_to_parent
prw-rw-r-- 1 moskewcz moskewcz 0 Mar 11 17:07 boda_fifo_to_worker
moskewcz@maaya:~/git_work/boda/run/tr4$ 

# then, you can run a master and worker using them. first start a master:
moskewcz@maaya:~/git_work/boda/run/tr4$ boda cs_test_master --boda-parent-addr=fns:boda_fifo_to_parent:boda_fifo_to_worker
boda_master: listening on parent_addr=fns:boda_fifo_to_parent:boda_fifo_to_worker
boda_master: entering accept_and_stop_listen() ... 
# ... the master should hang here for now ...

# then, in another shell, with the master still running, and in the same directory, start a worker: it should finish right away:

moskewcz@maaya:~/git_work/boda/run/tr4$ boda cs_test_worker --boda-parent-addr=fns:boda_fifo_to_parent:boda_fifo_to_worker
boda_worker: connecting to boda_parent_addr=fns:boda_fifo_to_parent:boda_fifo_to_worker
boda_worker: connected to parent.
boda_worker: got cmd=giggle
boda_worker: tee hee hee.
boda_worker: got cmd=quit
moskewcz@maaya:~/git_work/boda/run/tr4$ 

# ... looking back at the shell where the master was launched, it should now also have finished ...
boda_master: connected to worker.
boda_master: sent cmd=giggle
boda_master: sent cmd=quit
moskewcz@maaya:~/git_work/boda/run/tr4$ 

# 2) this process can be repeated, but with the client using fds mode, to test at least the client side of fds. note
#  that it's unclear if the server side of the fds method makes sense to use or how to test it.
# note: the client prints to stderr (always) to allow the fds mode testing to work (where stdin/stdout are used for IPC)
moskewcz@maaya:~/git_work/boda/run/tr4$ boda cs_test_worker --boda-parent-addr=fds:1:0 < boda_fifo_to_worker > boda_fifo_to_parent
# ... should produce same output as above case ...

# 3) this process can be repeated using "tcp:localhost:12791" as the boda address to test TCP based communication on a single host.
# 4) for (3), the client can be run on another host, with --boda-parent-addr="tcp:master_running_on_host:12791"

 */

  struct cs_test_master_t : virtual public nesi, public has_main_t // NESI(help="cs-testing master/server", bases=["has_main_t"], type_id="cs_test_master")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string boda_parent_addr; //NESI(default="tcp:localhost:12791", help="address to use for communication."
    //               "valid address types are 'tcp', 'fns', and 'fds'. "
    //               "for 'tcp', the address format is tcp:HOSTNAME:PORT (where PORT may be a known service name)")
    p_stream_t worker;
    virtual void main( nesi_init_arg_t * nia ) { 
      worker = make_stream_t( boda_parent_addr, 0 );
      printf( "boda_master: listening on parent_addr=%s\n", str(boda_parent_addr).c_str() );      
      printf( "boda_master: entering accept_and_stop_listen() ... \n" );      
      worker->wait_for_worker();
      printf( "boda_master: connected to worker.\n" );
      vect_string cmds{ "giggle", "quit" };
      for( vect_string::const_iterator i = cmds.begin(); i != cmds.end(); ++i ) {
	string const & cmd = *i;
	bwrite( *worker, cmd );
	printf( "boda_master: sent cmd=%s\n", str(cmd).c_str() );	
      }
    }    
  };

  // note: this mode prints to stderr (always) to allow the fds mode testing to work (where stdin/stdout are used for IPC)
  struct cs_test_worker_t : virtual public nesi, public has_main_t // NESI(help="cs-testing worker/client", bases=["has_main_t"], type_id="cs_test_worker")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string boda_parent_addr; //NESI(help="address of boda parent process in boda format",req=1)
    p_stream_t parent;
    virtual void main( nesi_init_arg_t * nia ) { 
      fprintf( stderr, "boda_worker: connecting to boda_parent_addr=%s\n", str(boda_parent_addr).c_str() );      
      parent = make_stream_t( boda_parent_addr, 1 );
      fprintf( stderr, "boda_worker: connected to parent.\n" );
      string cmd;
      while( 1 ) {
	bread( *parent, cmd );
	fprintf( stderr, "boda_worker: got cmd=%s\n", str(cmd).c_str() );
	if( 0 ) {} 
	else if( cmd == "quit" ) { break; }
	else if( cmd == "giggle" ) { fprintf( stderr, "boda_worker: tee hee hee.\n" ); }
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
    
    p_stream_t parent;

    ipc_compute_worker_t( void ) : proc_done(1) { }

    virtual void main( nesi_init_arg_t * nia ) { 
      global_timer_log_set_disable_finalize( 1 );

      vis.reset( new map_str_ipc_var_info_t );
      parent = make_stream_t( boda_parent_addr, 1 );

      string cmd;
      while( 1 ) {
	bread( *parent, cmd );
	if( 0 ) {} 
	else if( cmd == "init" ) { rtc->init(); }
	else if( cmd == "quit" ) { break; }
	else if( cmd == "compile" ) {
	  string cucl_src; bool show_compile_log; bool enable_lineinfo; vect_rtc_func_info_t func_infos; bool show_func_attrs;
	  bread( *parent, cucl_src ); bread( *parent, show_compile_log ); bread( *parent, enable_lineinfo );
	  bread( *parent, func_infos ); bread( *parent, show_func_attrs );
          uint32_t ret = 0;
          string err_str;
          try {
            rtc->compile( cucl_src, show_compile_log, enable_lineinfo, func_infos, show_func_attrs );
          } catch( rt_exception const & rte ) { ret=1; err_str = rte.what_and_stacktrace(); }
          bwrite( *parent, ret ); // 0 --> no error
          if( ret ) { bwrite( *parent, err_str ); }
	  parent->flush();
	}
	else if( cmd == "copy_nda_to_var" ) {
	  string vn;
	  dims_t dims;
	  bread( *parent, vn );
	  bread( *parent, dims );
	  ipc_var_info_t & vi = must_find( *vis, vn );
          assert_st( dims == vi.buf->dims );
	  bread_bytes( *parent, (char *)vi.buf->rp_elems(), vi.buf->dims.bytes_sz() );
	  rtc->copy_nda_to_var( vn, vi.buf );
	}
	else if( cmd == "copy_var_to_nda" ) {
	  string vn;
	  dims_t dims;
	  bread( *parent, vn );
	  bread( *parent, dims );
	  ipc_var_info_t & vi = must_find( *vis, vn );
          assert_st( dims == vi.buf->dims );
	  rtc->copy_var_to_nda( vi.buf, vn );
	  bwrite_bytes( *parent, (char const *)vi.buf->rp_elems(), vi.buf->dims.bytes_sz() );
	  parent->flush();
	}
	else if( cmd == "create_var_with_dims" ) {
	  string vn; dims_t dims;
	  bread( *parent, vn ); 
	  bread( *parent, dims );
	  must_insert( *vis, vn, ipc_var_info_t{dims} );
	  rtc->create_var_with_dims( vn, dims );
	}
	else if( cmd == "create_var_with_dims_as_reshaped_view_of_var" ) {
	  string vn; dims_t dims; string src_vn;
	  bread( *parent, vn ); 
	  bread( *parent, dims );
	  bread( *parent, src_vn );
	  must_insert( *vis, vn, ipc_var_info_t{dims} );
	  rtc->create_var_with_dims_as_reshaped_view_of_var( vn, dims, src_vn );
	}
	else if( cmd == "release_var" ) {
	  string vn;
	  bread( *parent, vn ); 
	  must_erase( *vis, vn );
	  rtc->release_var( vn );
	}
	else if( cmd == "set_var_to_zero" ) {
	  string vn;
	  bread( *parent, vn );
	  rtc->set_var_to_zero( vn );
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
	else if( cmd == "release_all_funcs" ) { rtc->release_all_funcs(); }
	else { rt_err("bad command:"+cmd); }
      }
    }
  };

#include"gen/rtc_ipc.cc.nesi_gen.cc"
}
