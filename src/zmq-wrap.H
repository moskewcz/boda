#ifndef _ZMQ_WRAP_H_
#define _ZMQ_WRAP_H_

#include"boda_base.H"
#include"str_util.H"
#include<zmq.h>

namespace boda {

  void dims_t_set_from_string( dims_t & dims, std::string const & dims_str );

  inline p_void make_p_zmq_context( void ) { return p_void( zmq_ctx_new(), zmq_ctx_destroy ); }
  inline p_void make_p_zmq_socket( p_void const & context, int const & type ) { return p_void( zmq_socket( context.get(), type ), zmq_close ); }

  inline void zmq_error_raise( string const & tag ) {
    int const zmq_err = zmq_errno();
    rt_err( strprintf( "%s() failed with ret=%s (%s)", tag.c_str(), str(zmq_err).c_str(), zmq_strerror(zmq_err) ) );
  }
  
  inline void zmq_check_call_ret( int const & call_ret, string const & tag ) { if( call_ret == -1 ) { zmq_error_raise( tag ); } }

  inline bool zmq_socket_has_more( p_void const & socket ) {
    int more;
    size_t more_size = sizeof(more);
    int const ret = zmq_getsockopt(socket.get(), ZMQ_RCVMORE, &more, &more_size);
    zmq_check_call_ret(ret, "zmq_socket_has_more");
    assert_st( ret == 0 );
    assert_st( more_size == sizeof(more) );
    return more;
  }
  inline void zmq_recv_check_expect_more( p_void const & socket, bool const & expect_more, string const & tag ) {
    if( expect_more && (!zmq_socket_has_more(socket)) ) { rt_err("expected another message part after " + tag); }
    if( (!expect_more) && zmq_socket_has_more(socket) ) { rt_err("unexpected extra message part after " + tag); }
  }

  typedef shared_ptr< zmq_msg_t > p_zmq_msg_t;
  inline void zmq_msg_close_check( zmq_msg_t * const & msg ) {
    int const ret = zmq_msg_close( msg );
    if( ret != 0 ) { rt_err( "zmg_msg_close() failed" ); }
  }
  inline p_zmq_msg_t make_p_zmq_msg_t( void ) {
    // sigh. seems to be no way around some tricky manual alloc/free/cleanup logic here.
    zmq_msg_t * msg = new zmq_msg_t;
    int const init_ret = zmq_msg_init( msg );
    if( init_ret != 0 ) { delete msg; rt_err( "zmg_msg_init() failed" ); }
    return p_zmq_msg_t( msg, zmq_msg_close_check );
  }
  inline void zmq_must_recv_msg( p_void const & socket, p_zmq_msg_t const & msg ) {
    int const ret = zmq_msg_recv(msg.get(), socket.get(), 0);
    zmq_check_call_ret(ret, "zmq_must_recv_msg");
    assert_st( (size_t)ret == zmq_msg_size(msg.get()) );
  }


  inline void zmq_send_data( p_void const & socket, void const * data, uint32_t const & sz, bool const more ) {
    int const ret = zmq_send( socket.get(), data, sz, more ? ZMQ_SNDMORE : 0 );
    if( ret != (int)sz ) { assert( ret == -1 ); zmq_error_raise( "zmq_send_data" ); }
  }
  inline void zmq_send_str( p_void const & socket, string const & data, bool const more ) {
    zmq_send_data( socket, &data[0], data.size(), more );
  }
  // NOTE/FIXME: we could make this no-copy, but we'd need to either rely on the client to keep data valid, or do
  // something fance to cooperate with ZMQ to reference count the data (i.e. using zmq_msg_init_data()), but it's not
  // clear how we can make that cooperate with shared_ptr<> -- we'd probably need a custom wrapper and/or to use
  // intrusive_ptr<>
  inline void zmq_send_p_uint8_with_sz_t( p_void const & socket, p_uint8_with_sz_t const & data, bool const more ) {
    zmq_send_data( socket, data.get(), data.sz, more );
  }
  inline void zmq_send_nda( p_void const & socket, p_nda_t const & nda, bool const more ) {
    zmq_send_str( socket, nda->dims.param_str(1), 1 );
    zmq_send_data( socket, nda->rp_elems(), nda->dims.bytes_sz(), more );
  }

  inline string zmq_recv_str( p_void const & socket, bool const & expect_more ) {
    p_zmq_msg_t msg = make_p_zmq_msg_t();
    zmq_must_recv_msg( socket, msg );
    uint8_t * const data = (uint8_t *)zmq_msg_data(msg.get());
    zmq_recv_check_expect_more(socket, expect_more, "zmq_recv_str()");
    return string(data, data+zmq_msg_size(msg.get()));
  }
  inline p_uint8_with_sz_t zmq_recv_p_uint8_with_sz_t( p_void const & socket, bool const & expect_more ) {
    p_zmq_msg_t msg = make_p_zmq_msg_t();
    zmq_must_recv_msg( socket, msg );
    zmq_recv_check_expect_more(socket, expect_more, "zmq_recv_p_uint8_with_sz_t()");

    return p_uint8_with_sz_t(msg, (uint8_t *)zmq_msg_data(msg.get()), zmq_msg_size(msg.get())); // alias ctor to bind lifetime to zmq msg
  }
  inline p_nda_t zmq_recv_nda( p_void const & socket, bool const & expect_more ) {
    string const nda_dims_str = zmq_recv_str(socket, 1);
    dims_t nda_dims;
    dims_t_set_from_string(nda_dims, nda_dims_str);
    p_uint8_with_sz_t const nda_data = zmq_recv_p_uint8_with_sz_t(socket, expect_more);
    return make_shared<nda_t>(nda_dims, nda_data);
  }

}

#endif /* _ZMQ_WRAP_H_ */
