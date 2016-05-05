// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"culibs-wrap.H"
#include"str_util.H"
#include"cublas_v2.h"
#include"cudnn.h"
namespace boda 
{

#define CES( x ) case x: return #x
  char const * cublasGetErrorString( cublasStatus_t const & ret ) {
    switch( ret ) {
      CES(CUBLAS_STATUS_SUCCESS);
      CES(CUBLAS_STATUS_NOT_INITIALIZED);
      CES(CUBLAS_STATUS_ALLOC_FAILED);
      CES(CUBLAS_STATUS_INVALID_VALUE);   
      CES(CUBLAS_STATUS_ARCH_MISMATCH);   
      CES(CUBLAS_STATUS_MAPPING_ERROR);   
      CES(CUBLAS_STATUS_EXECUTION_FAILED);
      CES(CUBLAS_STATUS_INTERNAL_ERROR);  
      CES(CUBLAS_STATUS_NOT_SUPPORTED);   
      CES(CUBLAS_STATUS_LICENSE_ERROR);
    default: return "UNKNOWN_TO_BODA_CUBLAS_ERROR_CODE";
    }
  }
#undef CES  

  void cublas_err_chk( cublasStatus_t const & ret, char const * const func_name ) {
    if( ret != CUBLAS_STATUS_SUCCESS ) { rt_err( strprintf( "%s() failed with ret=%s (%s)", func_name, str(ret).c_str(), cublasGetErrorString(ret) ) ); } }
  void cudnn_err_chk( cudnnStatus_t const & ret, char const * const func_name ) {
    if( ret != CUDNN_STATUS_SUCCESS ) { rt_err( strprintf( "%s() failed with ret=%s (%s)", func_name, str(ret).c_str(), cudnnGetErrorString(ret) ) ); } }

  struct culibs_wrap_t { 
    cublasHandle_t cbh;
    cudnnHandle_t cdh;
    culibs_wrap_t( void ) { 
      cublas_err_chk( cublasCreate(&cbh), "cublasCreate" ); 
      cudnn_err_chk( cudnnCreate(&cdh), "cublasCreate" ); 
      // FIXME: can we assume the default pointer mode is host? docs don't seem to say.
      //cublasPointerMode_t mode;
      //cublas_err_chk( cublasGetPointerMode( cbh, &mode ), "cublasGetPointerMode" );
    }
    ~culibs_wrap_t( void ) { 
      cublas_err_chk( cublasDestroy(cbh), "cublasDestroy" ); 
      cudnn_err_chk( cudnnDestroy(cdh), "cudnnDestroy" ); 
    }
  };
  p_culibs_wrap_t culibs_wrap_init( void ) { return make_shared< culibs_wrap_t >(); }
  void cublas_sgemm_wrap( p_culibs_wrap_t const & cw, uint64_t const & M, uint64_t const & N, uint64_t const & K, 
			  vect_rp_void const & args ) {
    assert_st( args.size() == 3 ); 
    // our inputs are row-major: a:KxM (pre-transposed), b:KxN; we want an output of c:MxN (row major);
    // if interpret our inputs as column-major, they are: at:MxK, b:NxK; so for col-major sgemm, we want -->
    // opA(A)=b opB(B)=a' --> b*a' = C:NxM (col major) --> 
    // so if we interpret C as row major, we get the desired c:MxN (row major)
    // so we want A=b opA=N, B=a opB=T, M=(our)N, N=(our)M, K=(our)K
    float const alpha = 1.0f;
    float const beta = 0.0f;
    cublas_err_chk( cublasSgemm( cw->cbh, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, 
				 &alpha,
				 *(float const **)(args[1]),  K, //const float           *A, int lda,
				 *(float const **)(args[0]),  K, //const float           *B, int ldb,
				 &beta,
				 *(float **)args[2],  N)  //float           *C, int ldc)
		    ,"cublasSgemm" );
  }

}
