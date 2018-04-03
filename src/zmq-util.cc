// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"timers.H"
#include"img_io.H"
#include"zmq-util.H"
#include<zmq.h>

#include"nesi.H" // for dims_t_set_from_string()
#include"data-stream.H"

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
  void zmq_recv_check_expect_more( p_void const & socket, bool const & expect_more, string const & tag ) {
    if( expect_more && (!zmq_socket_has_more(socket)) ) { rt_err("expected another message part after " + tag); }
    if( (!expect_more) && zmq_socket_has_more(socket) ) { rt_err("unexpected extra message part after " + tag); }
  }

  typedef shared_ptr< zmq_msg_t > p_zmq_msg_t;
  void zmq_msg_close_check( zmq_msg_t * const & msg ) {
    int const ret = zmq_msg_close( msg );
    if( ret != 0 ) { rt_err( "zmg_msg_close() failed" ); }
  }
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


  void zmq_send_data( p_void const & socket, void const * data, uint32_t const & sz, bool const more ) {
    int const ret = zmq_send( socket.get(), data, sz, more ? ZMQ_SNDMORE : 0 );
    if( ret != (int)sz ) { assert( ret == -1 ); zmq_error_raise( "zmq_send_data" ); }
  }
  void zmq_send_str( p_void const & socket, string const & data, bool const more ) {
    zmq_send_data( socket, &data[0], data.size(), more );
  }
  // NOTE/FIXME: we could make this no-copy, but we'd need to either rely on the client to keep data valid, or do
  // something fance to cooperate with ZMQ to reference count the data (i.e. using zmq_msg_init_data()), but it's not
  // clear how we can make that cooperate with shared_ptr<> -- we'd probably need a custom wrapper and/or to use
  // intrusive_ptr<>
  void zmq_send_p_uint8_with_sz_t( p_void const & socket, p_uint8_with_sz_t const & data, bool const more ) {
    zmq_send_data( socket, data.get(), data.sz, more );
  }
  void zmq_send_nda( p_void const & socket, p_nda_t const & nda, bool const more ) {
    zmq_send_str( socket, nda->dims.param_str(1), 1 );
    zmq_send_data( socket, nda->rp_elems(), nda->dims.bytes_sz(), more );
  }

  string zmq_recv_str( p_void const & socket, bool const & expect_more ) {
    p_zmq_msg_t msg = make_p_zmq_msg_t();
    zmq_must_recv_msg( socket, msg );
    uint8_t * const data = (uint8_t *)zmq_msg_data(msg.get());
    zmq_recv_check_expect_more(socket, expect_more, "zmq_recv_str()");
    return string(data, data+zmq_msg_size(msg.get()));
  }
p_uint8_with_sz_t zmq_recv_p_uint8_with_sz_t( p_void const & socket, bool const & expect_more ) {
    p_zmq_msg_t msg = make_p_zmq_msg_t();
    zmq_must_recv_msg( socket, msg );
    zmq_recv_check_expect_more(socket, expect_more, "zmq_recv_p_uint8_with_sz_t()");

    return p_uint8_with_sz_t(msg, (uint8_t *)zmq_msg_data(msg.get()), zmq_msg_size(msg.get())); // alias ctor to bind lifetime to zmq msg
  }
  p_nda_t zmq_recv_nda( p_void const & socket, bool const & expect_more ) {
    string const nda_dims_str = zmq_recv_str(socket, 1);
    dims_t nda_dims;
    dims_t_set_from_string(nda_dims, nda_dims_str);
    p_uint8_with_sz_t const nda_data = zmq_recv_p_uint8_with_sz_t(socket, expect_more);
    return make_shared<nda_t>(nda_dims, nda_data);
  }


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

  struct zmq_det_t : virtual public nesi // NESI( help="zmq detection client + external interface" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string endpoint; //NESI(req=1,help="zmq endpoint url string")
    string image_type; //NESI(default="imdecode",help="image data encoding type: either imdecode or raw_RGBA_32")
    float nms_thresh; //NESI(default="0.0",help="NMS threshold (0 to disable NMS)")
    uint32_t net_short_side_image_size; //NESI(default="576",
    // help="resize input images so their short side is this length (presvering aspet ratio). "
    //        "this determines the size of image that will be processed by the model "
    //        "(and thus the minimum detetable object size).")

    p_void context;
    p_void requester;

    void ensure_init( void ) {
      if(context) { return; } // already init/connected
      context = make_p_zmq_context();
      requester = make_p_zmq_socket(context, ZMQ_REQ);
      zmq_connect(requester.get(), endpoint.c_str());
    }

    p_nda_t do_det( p_nda_t const & image_data ) {
      ensure_init();
      string const opts_str = strprintf(
        "(net_short_side_image_size=%s,image_type=%s,nms_thresh=%s)",
        str(net_short_side_image_size).c_str(), str(image_type).c_str(), str(nms_thresh).c_str() );
      zmq_send_str(requester, opts_str, 1);
      zmq_send_nda(requester, image_data, 0);
      p_nda_t boxes = zmq_recv_nda(requester, 0);
      return boxes;
    }
  };
  struct zmq_det_t; typedef shared_ptr< zmq_det_t > p_zmq_det_t;

  struct zmq_det_client_t : virtual public nesi, public has_main_t // NESI(
                              // help="zmq detection inference test client",
                              // bases=["has_main_t"], type_id="zmq-det-client")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    p_zmq_det_t zmq_det; //NESI(default="(endpoint=ipc:///tmp/det-infer,nms_thresh=0.5,net_short_side_image_size=576)",help="zmq det options")
    filename_t image_fn; //NESI(default="%(boda_test_dir)/plasma_100.png",help="image file to send to server")

    void main( nesi_init_arg_t * nia ) {
      printf( "connecting to endpoint=%s and sending image_fn=%s\n", str(zmq_det->endpoint).c_str(), str(image_fn.exp).c_str() );
      p_uint8_with_sz_t image_data = map_file_ro_as_p_uint8( image_fn );
      p_nda_uint8_t image_nda = make_shared<nda_uint8_t>(
        dims_t{ vect_uint32_t{(uint32_t)image_data.sz}, "uint8_t"}, image_data );
      p_nda_t boxes = zmq_det->do_det(image_nda);
      printf( "boxes=%s\n", str(boxes).c_str() );
    }
  };

  struct zmq_det_stub_server_t : virtual public nesi, public has_main_t // NESI(
                              // help="zmq detection stub server",
                              // bases=["has_main_t"], type_id="zmq-det-stub-server")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string endpoint; //NESI(req=1,help="zmq endpoint url string")
    p_void context;
    p_void socket;

    void init_and_bind( void ) {
      if(context) { return; } // already init/connected
      context = make_p_zmq_context();
      socket = make_p_zmq_socket(context, ZMQ_REP);
      zmq_bind(socket.get(), endpoint.c_str());
    }
    void serve_forever( void ) { while( 1 ) { serve_one_request(); } }
    void serve_one_request( void ) {
      string const opts_str = zmq_recv_str(socket, 1 );
      p_nda_t const image_data = zmq_recv_nda(socket, 0);
      p_nda_float_t bboxes_nda = make_shared<nda_float_t>( dims_t{ vect_uint32_t{1,5}, // X,Y,W,H,Confidence
          {"obj","bbox_wit_confidence"}, "float" } );
      bboxes_nda->at2(0,0) = 100;
      bboxes_nda->at2(0,1) = 100;
      bboxes_nda->at2(0,2) = 200;
      bboxes_nda->at2(0,3) = 200;
      bboxes_nda->at2(0,4) = 0.98;
      zmq_send_nda(socket, bboxes_nda, 0);
    }
    void main( nesi_init_arg_t * nia ) {
      init_and_bind();
      serve_forever();
    }
  };


  struct data_stream_zmq_det_t : virtual public nesi, public data_stream_t // NESI(help="run detection on img in data block using zmq det client, annotate block with results",
                             // bases=["data_stream_t"], type_id="zmq-det")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string anno_meta; //NESI(default="boxes",help="use this string as the meta for added annotations")
    uint32_t skip_det; //NESI(default="0",help="if non-zero, skip doing detection (for profiling)")
    p_zmq_det_t zmq_det; //NESI(default="(endpoint=ipc:///tmp/det-infer,nms_thresh=0.5,net_short_side_image_size=576,image_type=raw_RGBA_32)",help="zmq det options")

    virtual string get_pos_info_str( void ) { return "zmq_det: <no-state>\n"; }

    // annotate a block with data from the zmq det server
    virtual data_block_t proc_block( data_block_t const & db ) {
      data_block_t ret = db;
      if(!ret.as_img) { rt_err( "zmq-det: expected input data block to have valid as_img field" ); }
      p_nda_uint8_t image_nda = ret.as_img->as_packed_RGBA_nda();
      if( skip_det ) { return ret; }
      // do lookup
      p_nda_t boxes = zmq_det->do_det(image_nda);
      assert_st( boxes->dims.size() == 2 );
      assert_st( boxes->dims.dims(1) == 5 ); // X,Y,W,H,confidence
      assert_st( boxes->dims.tn == "float" );
      p_nda_float_t boxes_float( make_shared<nda_float_t>(boxes) );

      uint32_t num_res = boxes->dims.dims(0);

      p_nda_float_t anno_nda = make_shared<nda_float_t>( dims_t{ vect_uint32_t{num_res,(uint32_t)4}, // X,Y,W,H
          {"obj","attr"}, "float" } );
      for( uint32_t i = 0; i != num_res; ++i ) {
        float const & x1 = boxes_float->at2(i,0);
        float const & y1 = boxes_float->at2(i,1);
        float const & x2 = boxes_float->at2(i,2);
        float const & y2 = boxes_float->at2(i,3);
        anno_nda->at2(i, 0) = x1;
        anno_nda->at2(i, 1) = y1;
        anno_nda->at2(i, 2) = x2 - x1;
        anno_nda->at2(i, 3) = y2 - y1;
      }
      ret.ensure_has_subblocks();
      data_block_t adb;
      adb.meta = anno_meta;
      adb.nda = anno_nda;
      ret.subblocks->push_back( adb );
      return ret;
    }

    virtual void data_stream_init( nesi_init_arg_t * nia ) { }
  };

#include"gen/zmq-util.cc.nesi_gen.cc"

}
