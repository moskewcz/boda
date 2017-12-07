#ifndef STREAM_UTIL_H
#define STREAM_UTIL_H

#include"boda_base.H"

namespace boda
{
  // ABC so we can abstract over fd_stream_t and sock_stream_t inside the ipc master and worker note that the
  // template-based send/recv code doesn't know or care about this ABC in particular; it will (try to) use any class
  // with the pos_type marker typedef.
  struct stream_t {
    virtual void wait_for_worker( void ) = 0; // for setup, used only in server
    // used for stream communication by templates in boda_base.H
    virtual void write( char const * const & d, size_t const & sz ) = 0;
    virtual void read( char * const & d, size_t const & sz ) = 0;
    virtual bool good( void ) = 0;
    virtual void flush( void ) = 0;
    typedef void pos_type; // flag class as IOStream-like for metaprogramming/template conditionals in boda_base.H
  };
  struct stream_t; typedef shared_ptr< stream_t > p_stream_t; 
  p_stream_t make_stream_t( string const & boda_parent_addr, bool const & is_worker );


}

#endif /* STREAM_UTIL_H */
