// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#ifndef _TIMERS_H_
#define _TIMERS_H_

namespace boda
{
  uint64_t get_cur_time( void );
  struct timer_log_t;
  extern timer_log_t global_timer_log; // for address only used in default below
  void global_timer_log_finalize( void );
  struct timer_t {
    string tag; // name of timer
    timer_log_t * tlog; // borrowed pointer, may be null
    uint64_t bt; // start time 
    bool ended; // true if stop() has been called
    uint64_t et; // end time, valid only if ended
    uint64_t dur; // (et-bt), valid only if ended
    timer_t( string const & tag_, timer_log_t * const tlog_ = &global_timer_log );
    void stop( void );
    ~timer_t( void );
  };

}

#endif /* _TIMERS_H_ */
