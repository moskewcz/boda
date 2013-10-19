#ifndef _NESI_H_
#define _NESI_H_
#include<cstdint>

struct tinfo_t
{
  uint32_t tid;
  char const * tname;
};

void nesi_init( void );
tinfo_t * get_tinfo( uint32_t const tid );

#endif /* _NESI_H_ */
