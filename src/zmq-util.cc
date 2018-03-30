// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"timers.H"
#include"img_io.H"
#include"zmq-util.H"
#include<zmq.h>

namespace boda {


  p_void make_p_zmq_context( void ) { return p_void( zmq_ctx_new(), zmq_ctx_destroy ); }
  p_void make_p_zmq_socket( p_void const & context, int const & type ) { return p_void( zmq_socket( context.get(), type ), zmq_close ); }

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


#include"gen/zmq-util.cc.nesi_gen.cc"

}
