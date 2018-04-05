// Copyright (c) 2018, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_base.H"
#include"str_util.H"
#include"zmq-wrap.H"

// for standalone distribution, compile with:
// g++ -fuse-ld=gold -Wall -O3 -g -std=c++0x -rdynamic -fPIC -fopenmp -Wall zmq-det-standalone.cc boda_base.cc str_util.cc stacktrace_util_gnu.cc -o zmq-det-standalone -lzmq -lboost_filesystem -lboost_system -lboost_iostreams

// cross-compilation for PX2, assuming a arm64 native chroot exists, and the stock cross compiler is installed:
// aarch64-linux-gnu-g++ --sysroot=/var/lib/schroot/chroots/xenial-arm64 -Wall -O3 -g -std=c++0x -rdynamic -fPIC -fopenmp -Wall zmq-det-standalone.cc boda_base.cc str_util.cc stacktrace_util_gnu.cc -o zmq-det-standalone -lzmq -lboost_filesystem -lboost_system -lboost_iostreams

// (or: inside boda tree compile with:)
// g++ -fuse-ld=gold -Wall -O3 -g -std=c++0x -rdynamic -fPIC -fopenmp -Wall -I.  -I../src  ../src/ext/zmq-det-standalone.cc ../src/boda_base.cc ../src/str_util.cc ../src/stacktrace_util_gnu.cc -o zmq-det-standalone -lzmq -lboost_filesystem -lboost_system -lboost_iostreams



namespace boda {

  // NOTE: this is a limited stub version of the 'real' boda/NESI version of this function, designed to handle the
  // limited cases we need here. we use this to avoid needing to pull in the entire lexp/NESI parts boda. this function
  // is needed by zmq-wrap in order to recv boda::nda_t's (boda's ND-Array class).
  void dims_t_set_from_string( dims_t & dims, string const & dims_str ) {
    assert( dims.empty() ); // should only be called on uninit/empty dims.
    assert_st( dims_str.size() > 1 );
    assert_st( dims_str[0] == '(' );
    assert_st( dims_str[dims_str.size()-1] == ')' );
    vect_string parts = split( dims_str.substr(1,dims_str.size()-2), ',');
    for( vect_string::const_iterator i = parts.begin(); i != parts.end(); ++i ) {
      if( !i->size() ) { continue; } // allow/ignore empty parts (really should only allow at end ...)
      vect_string kv = split(*i, '=');
      assert_st( kv.size() == 2 );
      if( kv[0] == "__tn__" ) { dims.tn = kv[1]; }
      else { dims.add_dims( kv[0], lc_str_u32(kv[1]) ); }
    }
    dims.calc_strides();
  }

  // NOTE: similarly to above, this is a de-NESI'd verison of the zmq_det_t class from zmq-util.cc
  struct zmq_det_t // : virtual public nesi // NESI( help="zmq detection client + external interface" )
  {
    // virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default=0,help="if non-zero, print verbose status messages.")
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
      if(verbose) { printf( "connecting to endpoint=%s ... \n", str(endpoint).c_str() ); }
      zmq_connect(requester.get(), endpoint.c_str());
      if(verbose) { printf( "... connected to endpoint=%s\n", str(endpoint).c_str() ); }
    }

    p_nda_t do_det( p_nda_t const & image_data ) {
      ensure_init();
      string const opts_str = strprintf(
        "(net_short_side_image_size=%s,image_type=%s,nms_thresh=%s)",
        str(net_short_side_image_size).c_str(), str(image_type).c_str(), str(nms_thresh).c_str() );
      if(verbose) { printf( "sending request with opts_str=%s and image_data\n", str(opts_str).c_str() ); }
      zmq_send_str(requester, opts_str, 1);
      zmq_send_nda(requester, image_data, 0);
      if(verbose) { printf( "waiting to recieve reply ...\n" ); }
      p_nda_t boxes = zmq_recv_nda(requester, 0);
      if(verbose) { printf( "... recieved reply.\n" ); }
      return boxes;
    }
  };

  void do_zmq_det_fn( string const & fn ) {
    zmq_det_t zmq_det;
    zmq_det.verbose = 1;
    zmq_det.endpoint = "ipc:///tmp/det-infer";
    zmq_det.image_type = "imdecode";
    zmq_det.nms_thresh = 0.1;
    zmq_det.net_short_side_image_size = 576;

    p_uint8_with_sz_t image_data = map_file_ro_as_p_uint8( fn );
    p_nda_uint8_t image_nda = make_shared<nda_uint8_t>(
      dims_t{ vect_uint32_t{(uint32_t)image_data.sz}, "uint8_t"}, image_data );
    p_nda_t boxes = zmq_det.do_det(image_nda);
    printf( "boxes=%s\n", str(boxes).c_str() );
  }
  
}

int main( int argc, char **argv ) {
  if( argc != 2 ) {
    printf( "error: expected argc==2, but got argc=%s\nusage: zmq-det-standalone image.png\n", boda::str(argc).c_str() );
    return 1;
  }
  boda::do_zmq_det_fn( std::string( argv[1] ) );
  return 0;
}
