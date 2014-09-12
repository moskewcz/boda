#ifndef _BODA_MUTEX_H_
#define _BODA_MUTEX_H_

#include<boost/thread/mutex.hpp>

namespace boda 
{
  using namespace boost;
  typedef unique_lock<mutex> ul_mutex;
  struct mutex_t { mutex m; };
}

#endif /* _BODA_MUTEX_H_ */
