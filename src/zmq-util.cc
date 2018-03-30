// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"timers.H"
#include"img_io.H"
#include"zmq-util.H"
#include<zmq.h>

namespace boda {

  struct zmq_hello_server_t : virtual public nesi, public has_main_t // NESI(
                              // help="simple ZMQ test server ",
                              // bases=["has_main_t"], type_id="zmq-hello-server")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    void main( nesi_init_arg_t * nia ) {
      //  Socket to talk to clients
      void *context = zmq_ctx_new ();
      void *responder = zmq_socket (context, ZMQ_REP);
      int rc = zmq_bind (responder, "tcp://*:5555");
      assert (rc == 0);

      while (1) {
        char buffer [10];
        zmq_recv (responder, buffer, 10, 0);
        printf ("Received Hello\n");
        sleep (1);          //  Do some 'work'
        zmq_send (responder, "World", 5, 0);
      }
    }
  };

  struct zmq_hello_client_t : virtual public nesi, public has_main_t // NESI(
                              // help="simple ZMQ test client  ",
                              // bases=["has_main_t"], type_id="zmq-hello-client")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    void main( nesi_init_arg_t * nia ) {
      printf ("Connecting to hello world server…\n");
      void *context = zmq_ctx_new ();
      void *requester = zmq_socket (context, ZMQ_REQ);
      zmq_connect (requester, "tcp://localhost:5555");

      int request_nbr;
      for (request_nbr = 0; request_nbr != 10; request_nbr++) {
        char buffer [10];
        printf ("Sending Hello %d…\n", request_nbr);
        zmq_send (requester, "Hello", 5, 0);
        zmq_recv (requester, buffer, 10, 0);
        printf ("Received World %d\n", request_nbr);
      }
      zmq_close (requester);
      zmq_ctx_destroy (context);
    }
  };


#include"gen/zmq-util.cc.nesi_gen.cc"

}
