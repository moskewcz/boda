// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"timers.H"
#include"str_util.H"
#include<time.h>
// NOTE: we use pre-2.0 sparsehash include path for compatiblity with
// ubuntu 14.04 (which ships sparsehash 1.10) for now, but this should
// eventually become <sparsehash/dense_hash_map>.
#include<google/dense_hash_map> 

namespace boda
{
  using std::pair;
  using google::dense_hash_map;
  // note: in general, these timer routines are not thread safe, as
  // they use statics and globals. sigh. but, with some rules/effort,
  // we can work around the issues. current rules: (1) call
  // get_cur_time() once at startup (before any other threads that may
  // call it start running) to set the clock type and init_ts.

  // returns nanoseconds since an arbitrary-but-during-this-process time (first call +0-1 seconds).
  uint64_t get_cur_time( void ) {
    static clockid_t cur_clock = CLOCK_MONOTONIC_RAW;
    static timespec init_ts = {0};
    static bool init_done = 0;
    timespec ts = {0};
    while( clock_gettime( cur_clock, &ts ) ) {
      if( errno == EINVAL ) { 
	if( cur_clock != CLOCK_MONOTONIC ) { rt_err("neither CLOCK_MONOTONIC_RAW nor CLOCK_MONOTONIC supported"); }
	cur_clock = CLOCK_MONOTONIC; // try CLOCK_MONOTONIC
      } else {	rt_err( "clock_gettime() unexpectedly failed with errno = " + str(errno) ); }
    }
    assert_st( ts.tv_sec >= 0 ); // no negative seconds, please
    if( !init_done ) { init_done = 1; init_ts = ts; }
    assert_st( ts.tv_sec >= init_ts.tv_sec ); // clock should actually be monotonic
    ts.tv_sec -= init_ts.tv_sec; // keep tv_sec reasonable small.
    assert_st( ts.tv_sec < 1000*1000*1000 ); // sanity check: process run time < 1B seconds. keeps total nsec in 64 bits
    assert_st( ts.tv_nsec >= 0 ); // no negative nano seconds, please
    uint64_t ret = ts.tv_nsec;
    ret += ts.tv_sec * 1000*1000*1000;
    return ret;
  }

  struct tlog_elem_t {
    uint64_t min_bt;
    uint64_t cnt;
    uint64_t tot_dur;
    tlog_elem_t( void ) : min_bt(uint64_t_const_max), cnt(0), tot_dur(0) { }
  };
  typedef dense_hash_map< string, tlog_elem_t > tlog_map_t;
  typedef pair< string, tlog_elem_t > tlog_map_val_t;
  typedef vector< tlog_map_val_t > vect_tlog_elem_t;

  struct tlog_map_val_by_bt_comp { 
    bool operator () ( tlog_map_val_t const & a, tlog_map_val_t const & b ) const { return a.second.min_bt < b.second.min_bt; }
  };

  string pre_pad_to_sz( string const & s, uint32_t const & sz ) {
    if( s.size() >= sz ) { return s; }
    return string( sz-s.size(), ' ' ) + s;
  }

  string pretty_format_nsecs( uint64_t const nsecs )
  {
    uint64_t const a_mil = (1000*1000*1000);
    uint64_t const secs = nsecs / a_mil ;
    uint32_t const msecs = (nsecs-secs*a_mil) / (1000*1000);
    uint32_t const usecs = (nsecs-secs*a_mil-msecs*1000*1000) / (1000);
    if( secs ) { return strprintf( "%s.%03us", str(secs).c_str(), msecs ); }
    else { return strprintf( "%s.%03ums", str(msecs).c_str(), usecs ); }
  }

  struct timer_log_t {
    tlog_map_t tlog_map;
    bool disable_finalize;
    timer_log_t( void ) : disable_finalize(0) { tlog_map.set_empty_key( string() ); }
    void finalize( void ) {
      if( disable_finalize ) { return; }
      if( tlog_map.empty() ) { return; } // don't print column headings if no timer data
      vect_tlog_elem_t tlog( tlog_map.begin(), tlog_map.end() );
      std::sort( tlog.begin(), tlog.end(), tlog_map_val_by_bt_comp() );
      printf( "TIMERS:  CNT     TOT_DUR      AVG_DUR    TAG  \n" ); 
      for( vect_tlog_elem_t::const_iterator i = tlog.begin(); i != tlog.end(); ++i) {
	printf( "%s %s %s    %s\n", 
		pre_pad_to_sz( str(i->second.cnt), 12 ).c_str(), 
		pre_pad_to_sz( pretty_format_nsecs(i->second.tot_dur), 12 ).c_str(),
		pre_pad_to_sz( pretty_format_nsecs(i->second.tot_dur / i->second.cnt), 12).c_str(),
		str(i->first).c_str()
		);
      }
    }
    void log_timer( timer_t const * const t ) {
      assert_st( t->ended );
#pragma omp critical 
      {
	tlog_elem_t & te = tlog_map[ t->tag ];
	min_eq( te.min_bt, t->bt );
	te.cnt += 1;
	te.tot_dur += t->dur;
      }
    }
  };

  timer_log_t global_timer_log;
  void global_timer_log_finalize( void ) {
    global_timer_log.finalize();
  }
  void global_timer_log_set_disable_finalize( bool const disable_finalize ) {
    global_timer_log.disable_finalize = disable_finalize;
  }
  
  

  timer_t::timer_t( string const & tag_, timer_log_t * const tlog_ ) : 
    tag(tag_), tlog(tlog_), bt( get_cur_time() ), ended(0), et(0) { }
  void timer_t::stop( void ) { 
    assert_st( !ended ); 
    et = get_cur_time(); 
    assert_st( et >= bt ); 
    dur = et - bt; 
    ended = 1;
    if( tlog ) { tlog->log_timer( this ); }
    //printf( "(double(dur)/(1000*1000))=%s\n", str((double(dur)/(1000*1000))).c_str() ); }
  }
  uint64_t timer_t::cur( void ) const { return get_cur_time() - bt; }
  timer_t::~timer_t( void ) { if( !ended ) { stop(); } } 

}
