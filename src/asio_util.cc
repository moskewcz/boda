#include"boda_tu_base.H"
#include"asio_util.H"
#include"build_info.H"

namespace boda 
{

  string get_boda_shm_filename( void ) { return strprintf( "/boda-rev-%s-pid-%s-top.shm", get_build_rev(), 
							   str(getpid()).c_str() ); }
  string create_boda_worker_fifo( vect_string const & args, string const & fifo_fn, bool const & dry_run ) {
    vect_string fin_args = args;
    string const bpa = strprintf("fns:%s:%s", (fifo_fn+"_to_parent").c_str(), (fifo_fn+"_to_worker").c_str() );
    fin_args.push_back( "--boda-parent-addr=" + bpa );
    if( dry_run ) { 
      for( vect_string::iterator i = fin_args.begin(); i != fin_args.end(); ++i ) { *i = "\"" + *i + "\""; }
      printstr( join( fin_args, " " ) + "\n" ); 
    }
    else { fork_and_exec_self( fin_args ); }
    return bpa;
  }
  int create_boda_worker_socketpair( vect_string const & args ) {
    int sp_fds[2];
    neg_one_fail( socketpair( AF_LOCAL, SOCK_STREAM, 0, sp_fds ), "socketpair" );
    set_fd_cloexec( sp_fds[0], 0 ); // we want the parent fd closed in our child
    vect_string fin_args = args;
    string const bpa = strprintf("fds:%s:%s", str(sp_fds[1]).c_str(), str(sp_fds[1]).c_str() );
    fin_args.push_back( "--boda-parent-addr=" + bpa );
    //fin_args.push_back( strprintf("--boda-parent-socket-fd=%s",str(sp_fds[1]).c_str() ) );
    fork_and_exec_self( fin_args );
    neg_one_fail( close( sp_fds[1] ), "close" ); // in the parent, we close the socket child will use
    return sp_fds[0];
  }
  void create_boda_worker( io_service_t & io, p_asio_alss_t & alss, vect_string const & args ) {
    int const fd = create_boda_worker_socketpair( args );
    alss.reset( new asio_alss_t(io) );
    alss->assign( stream_protocol(), fd );
  }

  // FIXME: transitional function for ipc code only supporting socketpair() parent addrs
  int get_single_parent_fd( string const & bpa ) {
    vect_string bpa_parts = split( bpa, ':' );
    if( bpa_parts.size() != 3 ) { rt_err( "boda_parent_addr must consist of three ':' seperated fields, in the form method:to_parent:to_worker"
					  " where method is 'fns' of 'fds' (sorry, no ':' allowed in filenames for the fns method)." ); }
    if( bpa_parts[0] != "fds" ) { rt_err( "this worker only supports the 'fds' parent addr method" ); }
    uint32_t const fd0 = lc_str_u32( bpa_parts[1] );
    uint32_t const fd1 = lc_str_u32( bpa_parts[2] );
    if( fd0 != fd1 ) { rt_err( "this worker only supports the case where the read and write fds are the same (the socketpair() case)" ); }
    return fd0;
  }



}
