// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"rtc_compute.H"
#include"has_main.H"
#include"asio_util.H"
#include"rand_util.H"
#include"timers.H"
#include"lexp.H"
#include"stream_util.H"

#include<boost/program_options/parsers.hpp> // for split_unix()

namespace boda 
{
  template< typename STREAM > inline void bwrite( STREAM & out, rtc_arg_t const & o ) { 
    bwrite( out, o.n );
    bwrite( out, o.v );
  }
  template< typename STREAM > inline void bread( STREAM & in, rtc_arg_t & o ) { 
    bread( in, o.n );
    bread( in, o.v );
  }

  template< typename STREAM > inline void bwrite( STREAM & out, rtc_compile_opts_t const & o ) { 
    bwrite( out, o.show_compile_log );
    bwrite( out, o.enable_lineinfo );
    bwrite( out, o.show_func_attrs );
  }
  template< typename STREAM > inline void bread( STREAM & in, rtc_compile_opts_t & o ) { 
    bread( in, o.show_compile_log );
    bread( in, o.enable_lineinfo );
    bread( in, o.show_func_attrs );
  }

  template< typename STREAM > inline void bwrite( STREAM & out, rtc_func_call_t const & o ) { 
    bwrite( out, o.rtc_func_name );
    bwrite( out, o.arg_map );
    bwrite( out, o.tpb.v );
    bwrite( out, o.blks.v );
  }
  template< typename STREAM > inline void bread( STREAM & in, rtc_func_call_t & o ) { 
    bread( in, o.rtc_func_name );
    bread( in, o.arg_map );
    bread( in, o.tpb.v );
    bread( in, o.blks.v );
  }

  template< typename STREAM > inline void bwrite( STREAM & out, op_base_t const & o ) { 
    bwrite( out, o.str_vals );
    bwrite( out, o.nda_vals );
  }
  template< typename STREAM > inline void bread( STREAM & in, op_base_t & o ) { 
    bread( in, o.str_vals );
    bread( in, o.nda_vals );
  }

  template< typename STREAM > inline void bwrite( STREAM & out, rtc_func_info_t const & o ) { 
    bwrite( out, o.func_name );
    bwrite( out, o.func_src );
    bwrite( out, o.arg_names );
    bwrite( out, o.op );
  }
  template< typename STREAM > inline void bread( STREAM & in, rtc_func_info_t & o ) { 
    bread( in, o.func_name );
    bread( in, o.func_src );
    bread( in, o.arg_names );
    bread( in, o.op );
  }

  struct ipc_var_info_t {
    p_nda_t buf;
    dims_t dims;
    ipc_var_info_t( dims_t const & dims_ ) : dims(dims_) { buf = make_shared<nda_t>( dims ); }
  };

  typedef map< string, ipc_var_info_t > map_str_ipc_var_info_t;
  typedef shared_ptr< map_str_ipc_var_info_t > p_map_str_ipc_var_info_t;



  struct ipc_compute_t : virtual public nesi, public rtc_compute_t // NESI(help="rtc-over-IPC wrapper/server",
			   // bases=["rtc_compute_t"], type_id="ipc" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    zi_bool init_done;
    string boda_parent_addr; //NESI(default="", help="address to use for communication. FIXME: document.")
    string remote_rtc; //NESI(default="(be=nvrtc)",help="remote rtc configuration")
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

    virtual string get_plat_tag( void ) {
      bwrite( *worker, string("get_plat_tag") ); 
      worker->flush();
      string ret;
      bread( *worker, ret );
      return ret;
    }
    
    ~ipc_compute_t( void ) {
      if( init_done.v ) {
	bwrite( *worker, string("quit") );
	worker->flush();
      }
    }
    void compile( vect_rtc_func_info_t const & func_infos, rtc_compile_opts_t const & opts ) {
      bwrite( *worker, string("compile") ); 
      bwrite( *worker, func_infos ); bwrite( *worker, opts ); 
      worker->flush();

      uint32_t ret = 0;
      string err_str;
      bread( *worker, ret ); // 0 --> no error
      if( ret ) { 
        bread( *worker, err_str );
        unsup_err( "rtc_ipc: " + err_str );
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
    p_nda_t get_var_raw_native_pointer( string const & vn ) {
      rt_err( "get_var_raw_native_pointer()-over-ipc: not implemented (and not needed/sensible?)");
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
    void release_func( string const & func_name ) {
      bwrite( *worker, string("release_func") ); bwrite( *worker, func_name ); worker->flush(); }
    uint32_t run( rtc_func_call_t const & rfc ) { 
      bwrite( *worker, string("run") ); bwrite( *worker, rfc ); worker->flush(); 
      uint32_t ret = 0;
      string err_str;
      bread( *worker, ret ); // 0 --> no error
      if( ret ) { 
        bread( *worker, err_str );
        unsup_err( "rtc_ipc: " + err_str );
      }
      uint32_t call_id; 
      bread( *worker, call_id ); 
      return call_id; 
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
    p_rtc_compute_t rtc; //NESI(default="(be=nvrtc)",help="rtc back-end to use")

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
	else if( cmd == "quit" ) { break; }
	else if( cmd == "init" ) { rtc->init(); }
	else if( cmd == "get_plat_tag" ) { 
          string const ret = rtc->get_plat_tag();
	  bwrite( *parent, ret ); parent->flush(); 
        }
	else if( cmd == "compile" ) {
	  vect_rtc_func_info_t func_infos; rtc_compile_opts_t opts;
	  bread( *parent, func_infos ); bread( *parent, opts );
          uint32_t ret = 0;
          string err_str;
          try {
            rtc->compile( func_infos, opts );
          } 
          catch( unsup_exception const & rte ) { ret=1; err_str = rte.what(); } // FIXME: stacktrace lost 
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
	else if( cmd == "release_func" ) { string func_name; bread( *parent, func_name ); rtc->release_func( func_name ); }
	else if( cmd == "run" ) { 
          rtc_func_call_t rfc; bread( *parent, rfc ); 
          uint32_t call_id;
          uint32_t ret = 0;
          string err_str;
          try { call_id = rtc->run( rfc ); }
          catch( unsup_exception const & rte ) { ret=1; err_str = rte.what(); } // FIXME: stacktrace lost 
          bwrite( *parent, ret ); 
          if( !ret ) {
            bwrite( *parent, call_id ); 
          } else {
            bwrite( *parent, err_str ); 
          }
          parent->flush(); 
        }
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
