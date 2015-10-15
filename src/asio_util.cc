#include"boda_tu_base.H"
#include"asio_util.H"
#include"build_info.H"

namespace boda 
{

  string get_boda_shm_filename( void ) { return strprintf( "/boda-rev-%s-pid-%s-top.shm", get_build_rev(), 
							   str(getpid()).c_str() ); }
  void create_boda_worker_fifo( vect_string const & args, string const & fifo_fn ) {
    vect_string fin_args = args;
    fin_args.push_back( "--boda-parent-socket-fd=-1" );
    fin_args.push_back( strprintf("--boda-parent-fifo=%s",fifo_fn.c_str() ) );
    fork_and_exec_self( fin_args );
  }
  int create_boda_worker( vect_string const & args ) {
    int sp_fds[2];
    neg_one_fail( socketpair( AF_LOCAL, SOCK_STREAM, 0, sp_fds ), "socketpair" );
    set_fd_cloexec( sp_fds[0], 0 ); // we want the parent fd closed in our child
    vect_string fin_args = args;
    fin_args.push_back( strprintf("--boda-parent-socket-fd=%s",str(sp_fds[1]).c_str() ) );
    fork_and_exec_self( fin_args );
    neg_one_fail( close( sp_fds[1] ), "close" ); // in the parent, we close the socket child will use
    return sp_fds[0];
  }
  void create_boda_worker( io_service_t & io, p_asio_alss_t & alss, vect_string const & args ) {
    int const fd = create_boda_worker( args );
    alss.reset( new asio_alss_t(io) );
    alss->assign( stream_protocol(), fd );
  }


}
