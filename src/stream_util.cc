// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"timers.H"
#include"stream_util.H"

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

  namespace io = boost::iostreams;

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
    bool is_worker;
    int fd; 
    int listen_fd; // listen_fd is only used on server (bind/listen/accept) side of connection. technically we only need
		   // one fd at a time, but it seems less confusing to have two explict ones for the two usages.
    sock_stream_t( void ) : is_worker(0), fd(-1), listen_fd(-1) { }
    sock_stream_t( string const & boda_parent_addr, bool const & is_worker_ ) : is_worker(is_worker_), fd(-1), listen_fd(-1) { 
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
      hints.ai_family = AF_UNSPEC; // but allow any family (IPv4 or IPv6). or, uncomment to force IPv4
      hints.ai_flags = AI_PASSIVE|AI_ADDRCONFIG; // bind to wildcard, use AI_ADDRCONFIG scheme for filtering out IPv4/IPv6 if not in use

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
      hints.ai_flags = AI_ADDRCONFIG; // use AI_ADDRCONFIG scheme for filtering out IPv4/IPv6 if not in use
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
      if( fd == -1 ) { rt_err( "socket_stream_t::write(): no open stream (didn't connect-to-worker/wait-for-worker-to-connect?)"); }
      size_t sz_written = 0;
      assert_st( sz );
      while( sz_written < sz ) { 
	int const ret = send( fd, d + sz_written, sz - sz_written, MSG_NOSIGNAL | MSG_MORE );
	if( ret == -1 ) { if( errno == EINTR ) { continue; } else { fd = -1; rt_err( string("socket-write-error: ") + strerror( errno ) ); } }
        if( ret == 0 ) { fd = -1; rt_err( "socket-write-error, ret = 0" ); }
	assert_st( ret > 0 ); // FIXME: other returns possible? make into rt_err() call?
	sz_written += ret;
      }
    }
    void read( char * const & d, size_t const & sz ) {
      if( fd == -1 ) { rt_err( "socket_stream_t::write(): no open stream (didn't connect-to-worker/wait-for-worker-to-connect?)"); }
      size_t sz_read = 0;
      assert_st( sz );
      while( sz_read < sz ) { 
	int const ret = recv( fd, d + sz_read, sz - sz_read, 0 );
	if( ret == -1 ) { if( errno == EINTR ) { continue; } else { fd = -1; rt_err( string("socket-read-error: ") + strerror( errno ) ); } }
        if( ret == 0 ) { fd = -1; rt_err( "socket-read-error, ret = 0 (eof?)" ); }
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
}
