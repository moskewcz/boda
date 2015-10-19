#include"boda_base.H"

#include"str_util.H"
#include"geom_prim.H"
#include"img_io.H"

#include<boost/asio.hpp>
#include<boost/bind.hpp>
#include<boost/date_time/posix_time/posix_time.hpp>

#include<sys/mman.h>
#include<sys/stat.h>


// note: unlike some of the other _util headers, this file doesn't
// attempt to hide/encapsulate asio. instead, it just groups together
// common includes, typedefs, and asio-related access functions in one
// place.

namespace boda {

  typedef boost::system::error_code error_code;
  namespace errc = boost::system::errc;
  typedef boost::asio::io_service io_service_t;
  typedef shared_ptr< io_service_t > p_io_service_t; 
  typedef boost::asio::deadline_timer deadline_timer_t;
  using boost::asio::buffer;
  using boost::asio::local::stream_protocol;
  typedef boost::asio::null_buffers null_buffers_t;
  typedef shared_ptr< deadline_timer_t > p_deadline_timer_t;
  typedef boost::asio::posix::stream_descriptor asio_fd_t;
  typedef shared_ptr< asio_fd_t > p_asio_fd_t; 
  typedef boost::asio::local::stream_protocol::socket asio_alss_t;
  typedef shared_ptr< asio_alss_t > p_asio_alss_t; 
  typedef vector< p_asio_alss_t > vect_p_asio_alss_t;
  typedef boost::posix_time::time_duration time_duration;
  using boost::posix_time::microseconds;
  using boost::posix_time::pos_infin;
  class disp_win_t;
  io_service_t & get_io( disp_win_t * const dw );
  deadline_timer_t & get_quit_event( disp_win_t * const dw );

  struct lb_event_t : public deadline_timer_t {
    lb_event_t( io_service_t & io ) : deadline_timer_t(io) { }
    uint32_t img_ix; // image event was inside or uint32_t_const_max if event not in an image
    i32_pt_t xy; // if img_ix is valid, coord withing image for event
    bool is_lb; // if event was left mouse button down
    bool is_key; // if event was key down
    int32_t keycode; // actually an SDL keycode from event.key.keysym.sym
    void set_is_lb( void ) { is_lb = 1; is_key = 0; }
    void set_is_key( int32_t const keycode_ ) { is_lb = 0; is_key = 1; keycode = keycode_; }
  };
  lb_event_t & get_lb_event( disp_win_t * const dw );

  template< typename CAP, typename MFUNC, typename OBJ > 
  void setup_capture_on_read( CAP & cap_afd, MFUNC const & mfunc, OBJ const  & obj ) {
    cap_afd.async_read_some( null_buffers_t(), bind( mfunc, obj, _1 ) ); }
  template< typename T, typename MFUNC, typename OBJ > 
  void register_quit_handler( T & has_quit, MFUNC const & mfunc, OBJ const  & obj ) {
    get_quit_event( &has_quit ).async_wait( bind( mfunc, obj, _1 ) ); }
  template< typename T, typename MFUNC, typename OBJ > 
  void register_lb_handler( T & has_quit, MFUNC const & mfunc, OBJ const  & obj ) {
    get_lb_event( &has_quit ).async_wait( bind( mfunc, obj, _1 ) ); }

  struct multi_alss_t : public vect_p_asio_alss_t { typedef void has_bread_bwrite; };

  inline void bwrite_bytes( multi_alss_t & out, char const * const & d, size_t const & sz ) {
    for( vect_p_asio_alss_t::const_iterator i = out.begin(); i != out.end(); ++i ) { bwrite_bytes( **i, d, sz ); } }
  inline void bread_bytes( multi_alss_t & in, char * const & d, size_t const & sz ) {
    char * const buf = (char *)alloca( sz );
    for( vect_p_asio_alss_t::const_iterator i = in.begin(); i != in.end(); ++i ) { 
      if( i == in.begin() ) { bread_bytes( **i, d, sz ); } // read from first stream to final output
      else { // read from other streams to buffer and check result is same
	bread_bytes( **i, buf, sz ); 
	if( memcmp( d, buf, sz ) ) { rt_err( "interprocess comm logic error: multi-read not equal across dests" ); }
      }
    }
  }

  template< typename AsioWritable, typename check_T<typename AsioWritable::lowest_layer_type>::int_ = 0 > inline 
  void bwrite_bytes( AsioWritable & out, char const * const & d, size_t const & sz ) { write( out, buffer( d, sz ) ); }
  template< typename AsioReadable, typename check_T<typename AsioReadable::lowest_layer_type>::int_ = 0 > inline
  void bread_bytes( AsioReadable & in, char * const & d, size_t const & sz ) { read( in, buffer( d, sz ) ); }

  // shm related functionality. break this up more? de-templatize and/or explicitly instantiate to reduce header code?

  string get_boda_shm_filename( void );

  template< typename STREAM > p_uint8_t make_and_share_p_uint8_t( STREAM & out, uint32_t const sz ) {
    string const fn = get_boda_shm_filename();
    shm_unlink( fn.c_str() ); // we don't want to use any existing shm, so try to remove it if it exists.
    // note that, in theory, we could have just unlinked an in-use shm, and thus broke some other
    // processes. also note that between the unlink above and the shm_open() below, someone could
    // create shm with the name we want, and then we will fail. note that, generally speaking,
    // shm_open() doesn't seem secure/securable (maybe one could use a random shm name here?), but
    // we're mainly trying to just be robust -- including being non-malicious
    // errors/bugs/exceptions.
    int const fd = shm_open( fn.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_EXCL, S_IRUSR | S_IWUSR );
    if( fd == -1 ) { rt_err( strprintf( "send-end shm_open() failed with errno=%s", str(errno).c_str() ) ); }
    neg_one_fail( ftruncate( fd, sz ), "ftruncate" );
    p_uint8_t ret = make_mmap_shared_p_uint8_t( fd, sz, 0 );
    bwrite( out, fn ); 
    bwrite( out, sz );
    uint8_t done;
    bread( out, done );
    // we're done with the shm segment name now, so free it. notes: (1) we could have freed it
    // earlier and used SCM_RIGHTS to xfer the fd. (2) we could make a better effort to unlink here
    // in the event of errors in the above xfer.
    neg_one_fail( shm_unlink( fn.c_str() ), "shm_unlink" );
    return ret;
  }

  template< typename STREAM > p_uint8_t recv_shared_p_uint8_t( STREAM & in ) {
    string fn;
    bread( in, fn ); // note: currently always == get_boda_shm_filename() ...
    uint32_t sz = 0;
    bread( in, sz );
    assert_st( sz );
    int const fd = shm_open( fn.c_str(), O_RDWR, S_IRUSR | S_IWUSR );
    if( fd == -1 ) { rt_err( strprintf( "recv-end shm_open() failed with errno=%s", str(errno).c_str() ) ); }
    neg_one_fail( ftruncate( fd, sz ), "ftruncate" );
    p_uint8_t ret = make_mmap_shared_p_uint8_t( fd, sz, 0 );
    uint8_t const done = 1;
    bwrite( in, done );
    return ret;
  }

  template< typename STREAM > p_img_t make_and_share_p_img_t( STREAM & out, u32_pt_t const & sz ) {
    bwrite( out, sz );
    p_img_t img( new img_t );
    img->set_sz( sz );
    img->pels = make_and_share_p_uint8_t( out, img->sz_raw_bytes() ); // FIXME: check img->row_align wrt map page sz?
    return img;
  }

  template< typename STREAM > p_img_t recv_shared_p_img_t( STREAM & stream ) {
    p_img_t img( new img_t );
    u32_pt_t img_res;
    bread( stream, img_res );
    img->set_sz( img_res );
    img->pels = recv_shared_p_uint8_t( stream ); // FIXME: check img->row_align wrt map page sz?
    return img;
  }

  std::string create_boda_worker_fifo( vect_string const & args, string const & fifo_fn, bool const & dry_run );
  int get_single_parent_fd( string const & bpa ); // FIXME: transitional function for ipc code only supporting socketpair() parent addrs
  int create_boda_worker( vect_string const & args );
  void create_boda_worker( io_service_t & io, p_asio_alss_t & alss, vect_string const & args );

}
