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
