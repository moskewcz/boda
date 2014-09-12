#ifndef _BODA_MUTEX_H_
#define _BODA_MUTEX_H_

//#include <boost/thread/mutex.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/anonymous_shared_memory.hpp>

namespace boda 
{
  typedef boost::interprocess::interprocess_mutex ip_mutex;
  typedef boost::interprocess::scoped_lock<ip_mutex> sl_ip_mutex;
  struct ip_mutex_t { ip_mutex m; };
}

#endif /* _BODA_MUTEX_H_ */
