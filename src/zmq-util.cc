// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"timers.H"
#include"img_io.H"
#include"zmq-util.H"
#include"zmq-wrap.H"

#include"data-stream.H"


namespace boda {

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
          {"obj","bbox_with_confidence"}, "float" } );
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
      if( boxes->dims.dims_prod() == 0 ) { return ret; } // no results for this frame, but FIXME(?) an odd way to tell?
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
