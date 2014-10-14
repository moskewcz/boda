#include<boost/asio.hpp>
#include<boost/bind.hpp>
#include<boost/date_time/posix_time/posix_time.hpp>
#include"boda_base.H"

// note: unlike some of the other _util headers, this file doesn't
// attempt to hide/encapsulate asio. instead, it just groups together
// common includes, typedefs, and asio-related access functions in one
// place.

namespace boda {

  typedef boost::system::error_code error_code;
  typedef boost::asio::io_service io_service_t;
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

  template< typename CAP, typename MFUNC, typename OBJ > 
  void setup_capture_on_read( CAP & cap_afd, MFUNC const & mfunc, OBJ const  & obj ) {
    cap_afd.async_read_some( null_buffers_t(), bind( mfunc, obj, _1 ) );
  }

}
