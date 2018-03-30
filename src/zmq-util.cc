// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"timers.H"
#include"img_io.H"
#include"zmq-util.H"
#include<zmq.h>

#include"nesi.H" // for dims_t_set_from_string()

namespace boda {



  p_void make_p_zmq_context( void ) { return p_void( zmq_ctx_new(), zmq_ctx_destroy ); }
  p_void make_p_zmq_socket( p_void const & context, int const & type ) { return p_void( zmq_socket( context.get(), type ), zmq_close ); }

  void zmq_error_raise( string const & tag ) {
    int const zmq_err = zmq_errno();
    rt_err( strprintf( "%s() failed with ret=%s (%s)", tag.c_str(), str(zmq_err).c_str(), zmq_strerror(zmq_err) ) );
  }
  void zmq_check_call_ret( int const & call_ret, string const & tag ) { if( call_ret == -1 ) { zmq_error_raise( tag ); } }

  bool zmq_socket_has_more( p_void const & socket ) {
    int more;
    size_t more_size = sizeof(more);
    int const ret = zmq_getsockopt(socket.get(), ZMQ_RCVMORE, &more, &more_size);
    zmq_check_call_ret(ret, "zmq_socket_has_more");
    assert_st( ret == 0 );
    assert_st( more_size == sizeof(more) );
    return more;
  }

  void zmq_send_data( p_void const & socket, void const * data, uint32_t const & sz, bool const more ) {
    int const ret = zmq_send( socket.get(), data, sz, more ? ZMQ_SNDMORE : 0 );
    if( ret != (int)sz ) { assert( ret == -1 ); zmq_error_raise( "zmq_send_data" ); }
  }
  void zmq_send_str( p_void const & socket, string const & data, bool const more ) {
    zmq_send_data( socket, &data[0], data.size(), more );
  }
  void zmq_send_p_uint8_with_sz_t( p_void const & socket, p_uint8_with_sz_t const & data, bool const more ) {
    zmq_send_data( socket, data.get(), data.sz, more );
  }


  void zmq_msg_close_check( zmq_msg_t * const & msg ) {
    int const ret = zmq_msg_close( msg );
    if( ret != 0 ) { rt_err( "zmg_msg_close() failed" ); }
  }

  typedef shared_ptr< zmq_msg_t > p_zmq_msg_t;
  p_zmq_msg_t make_p_zmq_msg_t( void ) {
    // sigh. seems to be no way around some tricky manual alloc/free/cleanup logic here.
    zmq_msg_t * msg = new zmq_msg_t;
    int const init_ret = zmq_msg_init( msg );
    if( init_ret != 0 ) { delete msg; rt_err( "zmg_msg_init() failed" ); }
    return p_zmq_msg_t( msg, zmq_msg_close_check );
  }

  void zmq_must_recv_msg( p_void const & socket, p_zmq_msg_t const & msg ) {
    int const ret = zmq_msg_recv(msg.get(), socket.get(), 0);
    zmq_check_call_ret(ret, "zmq_must_recv_msg");
    assert_st( (size_t)ret == zmq_msg_size(msg.get()) );
  }

  string zmq_msg_as_string( p_zmq_msg_t const & msg ) {
    uint8_t * const data = (uint8_t *)zmq_msg_data(msg.get());
    return string(data, data+zmq_msg_size(msg.get()));
  }

  // since the underlying zmg_msg_t can be modified after this call, this isn't the safest function. FIXME: somehow make sure the
  p_uint8_with_sz_t zmq_msg_as_p_uint8_with_sz_t( p_zmq_msg_t const & msg ) {
    return p_uint8_with_sz_t(msg, (uint8_t *)zmq_msg_data(msg.get()), zmq_msg_size(msg.get())); // alias ctor to bind lifetime to zmq msg
  }


#if 0
  // for now, we only have the endpoint as a zmq options. but if we add/use more, and they are shared, we could wrap
  // them up in a NESI class:
  struct zmq_opts_t : virtual public nesi //XNESI(help="zmq connection options") 
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string endpoint; //XNESI(default="",help="zmq endpoint url string")
    // TODO/FIXME: add send/recv HWM option settings here? it's not clear we need them, or why exactly they are special
  };
  struct zmq_opts_t; typedef shared_ptr< zmq_opts_t > p_zmq_opts_t; 
  // use as: p_zmq_opts_t zmq_opts; //XNESI(help="server zmq options (including endpoint)")
#endif
  
  struct zmq_hello_server_t : virtual public nesi, public has_main_t // NESI(
                              // help="simple ZMQ test server ",
                              // bases=["has_main_t"], type_id="zmq-hello-server")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    string endpoint; //NESI(default="ipc:///tmp/boda-zmq-test-ipc-endpoint",help="zmq endpoint url string")

    void main( nesi_init_arg_t * nia ) {
      //  Socket to talk to clients
      p_void context = make_p_zmq_context();
      p_void responder = make_p_zmq_socket(context, ZMQ_REP);
      int rc = zmq_bind(responder.get(), endpoint.c_str());
      assert(rc == 0);

      while (1) {
        char buffer[10];
        zmq_recv(responder.get(), buffer, 10, 0);
        printf("Received Hello\n");
        sleep(1);          //  Do some 'work'
        zmq_send(responder.get(), "World", 5, 0);
      }
    }
  };

  struct zmq_hello_client_t : virtual public nesi, public has_main_t // NESI(
                              // help="simple ZMQ test client  ",
                              // bases=["has_main_t"], type_id="zmq-hello-client")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    string endpoint; //NESI(default="ipc:///tmp/boda-zmq-test-ipc-endpoint",help="zmq endpoint url string")

    void main( nesi_init_arg_t * nia ) {
      printf ("Connecting to hello world server…\n");
      p_void context = make_p_zmq_context();
      p_void requester = make_p_zmq_socket(context, ZMQ_REQ);
      zmq_connect(requester.get(), endpoint.c_str());

      int request_nbr;
      for(request_nbr = 0; request_nbr != 3; request_nbr++) {
        char buffer[10];
        printf("Sending Hello %d…\n", request_nbr);
        zmq_send(requester.get(), "Hello", 5, 0);
        zmq_recv(requester.get(), buffer, 10, 0);
        printf("Received World %d\n", request_nbr);
      }
    }
  };

  struct zmq_det_client_t : virtual public nesi, public has_main_t // NESI(
                              // help="zmq detection inference test client  ",
                              // bases=["has_main_t"], type_id="zmq-det-client")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    string endpoint; //NESI(default="ipc:///tmp/det-infer",help="zmq endpoint url string")
    filename_t image_fn; //NESI(default="%(boda_test_dir)/plasma_100.png",help="image file to send to server")

    void main( nesi_init_arg_t * nia ) {
      p_uint8_with_sz_t image_data = map_file_ro_as_p_uint8( image_fn );
      printf( "connecting to endpoint=%s and sending image_fn=%s\n", str(endpoint).c_str(), str(image_fn.exp).c_str() );
      p_void context = make_p_zmq_context();
      p_void requester = make_p_zmq_socket(context, ZMQ_REQ);
      zmq_connect(requester.get(), endpoint.c_str());
      zmq_send_p_uint8_with_sz_t( requester, image_data, 0 );
      p_zmq_msg_t msg = make_p_zmq_msg_t();
      zmq_must_recv_msg(requester, msg);
      string const boxes_dims_str = zmq_msg_as_string(msg);
      if( !zmq_socket_has_more(requester) ) { rt_err("expected another message part"); }
      zmq_must_recv_msg(requester, msg);
      p_uint8_with_sz_t boxes_data = zmq_msg_as_p_uint8_with_sz_t(msg);
      if( zmq_socket_has_more(requester) ) { rt_err("unexpected extra message part"); }
      dims_t boxes_dims;
      dims_t_set_from_string(boxes_dims, boxes_dims_str);

      p_nda_t boxes = make_shared<nda_t>(boxes_dims, boxes_data);
      printf( "boxes=%s\n", str(boxes).c_str() );
    }
  };


#include"gen/zmq-util.cc.nesi_gen.cc"

}
