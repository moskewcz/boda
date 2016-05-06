// Copyright (c) 2016, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"timers.H"
#include"lexp.H"
#include"has_main.H"
#include"str_util.H"

#include <qblas_cblas3.h>

namespace boda 
{
  struct qblas_test_t : virtual public nesi, public has_main_t // NESI(help="test qblas",
		      // bases=["has_main_t"], type_id="qblas-test" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    uint32_t MNK; //NESI(default=1024,help="M/N/K size for sgemm")

    uint32_t iters; //NESI(default=1,help="iters for sgemm")

    virtual void main( nesi_init_arg_t * nia );
  };
 
  void qblas_test_t::main( nesi_init_arg_t * nia ) {
    printf( "MNK=%s\n", str(MNK).c_str() );
    uint32_t const matrixSize = MNK*MNK;
    
    float *A = new float[matrixSize];
    float *B = new float[matrixSize];
    float *C = new float[matrixSize];
    
    for(uint32_t i=0; i < matrixSize; i++) { A[i] = B[i] = C[i] = 1.0; }

    {
      timer_t t("sgemm");
      for( uint32_t i = 0; i != iters; ++i ) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MNK, MNK, MNK,
                    1.0, A, MNK, B, MNK, 0.0, C, MNK);
      }
    }
    printf( "C[0]=%s\n", str(C[0]).c_str() );
  }

#include"gen/qblas-test.cc.nesi_gen.cc"

}
