#include"nesi.H"
#include<cassert>

#include"gen/tinfos.cc"

void nesi_init( void )
{
  
}

tinfo_t * get_tinfo( uint32_t const tid ) {  assert( tid < num_tinfos ); return &tinfos[tid]; }
