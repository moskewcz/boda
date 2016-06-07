// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"culibs-wrap.H"
#include"str_util.H"
#include"cublas_v2.h"
#include"cudnn.h"
#include"rtc_compute.H" // for rtc_func_info_t
#include<cuda.h> // FIXME: use rtc for things from here?
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

  template< typename T > struct cudnn_create_T { typedef cudnnStatus_t (*p_func)( T * ); };
  template< typename T > struct cudnn_destroy_T { typedef cudnnStatus_t (*p_func)( T ); };

  template< typename T, typename cudnn_create_T<T>::p_func CREATE, typename cudnn_destroy_T<T>::p_func DESTROY > 
  struct cudnn_wrap_t {
    T v;
    cudnn_wrap_t( void ) { cudnn_err_chk( CREATE(&v), "cudnnCreate<T>(&v)" ); }
    ~cudnn_wrap_t( void ) { cudnn_err_chk( DESTROY(v), "cudnnDestroy<T>(v)" ); }
  };
  
  typedef cudnn_wrap_t< cudnnTensorDescriptor_t,cudnnCreateTensorDescriptor,cudnnDestroyTensorDescriptor > cudnn_tensor_t;
  typedef cudnn_wrap_t< cudnnFilterDescriptor_t,cudnnCreateFilterDescriptor,cudnnDestroyFilterDescriptor > cudnn_filter_t;
  typedef cudnn_wrap_t< cudnnConvolutionDescriptor_t,cudnnCreateConvolutionDescriptor,cudnnDestroyConvolutionDescriptor > cudnn_convolution_t;

  typedef vector< int > vect_int;

  void set_cudnn_tensor_from_dims_t( cudnn_tensor_t & cu_v, dims_t const & dims ) {
    vect_int dims_;
    vect_int strides;
    for( uint32_t i = 0; i != dims.size(); ++i ) { dims_.push_back( dims.dims(i) );strides.push_back( dims.strides(i) ); }
    cudnn_err_chk( cudnnSetTensorNdDescriptor( cu_v.v, CUDNN_DATA_FLOAT, dims.size(), &dims_[0], &strides[0] ), 
                   "cudnnSetTensorNdDescriptor");
  }
  void set_cudnn_filter_from_dims_t( cudnn_filter_t & cu_v, dims_t const & dims ) { // like tensor, but no strides
    vect_int dims_;
    for( uint32_t i = 0; i != dims.size(); ++i ) { dims_.push_back( dims.dims(i) ); }
    cudnn_err_chk( cudnnSetFilterNdDescriptor( cu_v.v, CUDNN_DATA_FLOAT, dims.size(), &dims_[0] ), 
                   "cudnnSetFilterNdDescriptor");
  }

  // FIXME: dup'd with nvrtc_util.H 
  void cu_err_chk( CUresult const & ret, char const * const func_name );
  template< typename T >  struct cup_T {
    typedef T element_type;
    CUdeviceptr p;
    uint32_t sz;
    void set_to_zero( void ) { cu_err_chk( cuMemsetD8(  p, 0, sz * sizeof(element_type) ), "cuMemsetD8" ); }
    cup_T( uint32_t const sz_ ) : p(0), sz(sz_) { 
      cu_err_chk( cuMemAlloc( &p,    sz * sizeof(element_type) ), "cuMemAlloc" ); 
      set_to_zero();
    }
    ~cup_T( void ) { cu_err_chk( cuMemFree( p ), "cuMemFree" ); }
  };
  typedef cup_T< uint8_t > cup_uint8_t;
  typedef shared_ptr< cup_uint8_t > p_cup_uint8_t; 


  struct culibs_wrap_t { 
    cublasHandle_t cbh;
    cudnnHandle_t cdh;
    p_cup_uint8_t cu_work;
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
    void call( rtc_func_info_t const & fi, p_map_str_p_nda_t const & args ) { 
      string const & fn = fi.func_name;
      if( 0 ) {}
      else if( startswith( fn, "cublas_sgemm" ) ) { sgemm( args ); }
      else if( startswith( fn, "cudnn_conv" ) ) { conv( fi.op, args ); }
      else { rt_err( "unknown/unhandled culibs_wrap function: " + fn ); }
    }

    void conv( op_base_t const & op, p_map_str_p_nda_t const & args ) {
      nda_t const & filts = *must_find(*args,"arg_0");
      nda_t const & biases = *must_find(*args,"arg_1");
      nda_t const & in = *must_find(*args,"arg_2");
      nda_t const & out = *must_find(*args,"arg_3");
      cudnn_filter_t cu_filts;
      cudnn_tensor_t cu_in;
      cudnn_tensor_t cu_out;
      set_cudnn_filter_from_dims_t( cu_filts, filts.dims );
      set_cudnn_tensor_from_dims_t( cu_in, in.dims );
      set_cudnn_tensor_from_dims_t( cu_out, out.dims );
      
      // create 'expanded' biases dims, with dim names/order that match the output, but sizes 1 expect for dims present in the biases
      dims_t biases_dims_exp; 
      biases_dims_exp.tn = biases.dims.tn;
      uint32_t biases_dims_used = 0;
      for( uint32_t i = 0; i != out.dims.size(); ++i ) {
        string dn = out.dims.names(i);
        uint32_t dim_sz = 1;
        dim_t const * const mbd = biases.dims.get_dim_by_name( (dn=="chan")?"out_chan":dn ); // FIXME: maybe dim of biases should be just 'chan'?
        if( mbd ) { ++biases_dims_used; dim_sz = mbd->sz; }
        biases_dims_exp.add_dims( dn, dim_sz );
      }
      assert_st( biases_dims_used == biases.dims.size() );
      biases_dims_exp.calc_strides();
      cudnn_tensor_t cu_biases;
      set_cudnn_tensor_from_dims_t( cu_biases, biases_dims_exp );

      cudnn_convolution_t cu_conv;
      dims_t const & in_pad = op.get_dims( "in_pad" );
      dims_t const & stride = op.get_dims( "stride" );
      assert_st( in_pad.size() == 2 );
      assert_st( stride.size() == 2 );
      cudnn_err_chk( cudnnSetConvolution2dDescriptor(  cu_conv.v,
                                                       in_pad.dsz("y"),    // zero-padding height
                                                       in_pad.dsz("x"),    // zero-padding width
                                                       stride.dsz("y"),        // vertical filter stride
                                                       stride.dsz("x"),        // horizontal filter stride
                                                       1, // upscale the input in x-direction
                                                       1, // upscale the input in y-direction
                                                       CUDNN_CROSS_CORRELATION
                                                       ), "cudnnSetConvolution2dDescriptor" );

      vect_int cu_out_dims;
      cu_out_dims.resize( out.dims.size() );
      cudnn_err_chk( cudnnGetConvolutionNdForwardOutputDim( cu_conv.v, cu_in.v, cu_filts.v, cu_out_dims.size(),
                                                            &cu_out_dims[0] ), "cudnnGetConvolutionNdForwardOutputDim" );
      printf( "out.dims=%s cu_out_dims=%s\n", str(out.dims).c_str(), str(cu_out_dims).c_str() );

      // allow scratch of 4 times in+out+filts bytes
      uint64_t max_scratch = (in.dims.dims_prod()+out.dims.dims_prod()+filts.dims.dims_prod())*4*4;
      cudnnConvolutionFwdAlgo_t cu_conv_algo = (cudnnConvolutionFwdAlgo_t)999;
      cudnn_err_chk( cudnnGetConvolutionForwardAlgorithm( cdh, cu_in.v, cu_filts.v, cu_conv.v, cu_out.v,
                                                          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                          max_scratch,
                                                          &cu_conv_algo ), "cudnnGetConvolutionForwardAlgorithm" );
      size_t need_scratch = -1;
      cudnn_err_chk( cudnnGetConvolutionForwardWorkspaceSize( cdh, cu_in.v, cu_filts.v, cu_conv.v, cu_out.v, cu_conv_algo, 
                                                              &need_scratch ), "cudnnGetConvolutionForwardWorkspaceSize" );
      
      printf( "cu_conv_algo=%s need_scratch=%s\n", str(cu_conv_algo).c_str(), str(need_scratch).c_str() );
      // FIXME: need to make scratch persistent here? need access to rtc?
      if( need_scratch ) {
        if( (!cu_work) || ( cu_work->sz < need_scratch ) ) { cu_work = make_shared<cup_uint8_t>( need_scratch ); }
      }

      float const alpha = 1.0f;
      float const beta = 0.0f;
      cudnn_err_chk( cudnnConvolutionForward( cdh,
                                              &alpha,
                                              cu_in.v,
                                              in.rp_elems(),
                                              cu_filts.v,
                                              filts.rp_elems(),
                                              cu_conv.v,
                                              cu_conv_algo,
                                              need_scratch ? ((void *)(uintptr_t)cu_work->p) : 0,
                                              need_scratch,
                                              &beta,
                                              cu_out.v,
                                              (void *)out.rp_elems()
                                              ), "cudnnConvolutionForward" );

      cudnn_err_chk( cudnnAddTensor_v3( cdh,
                                        &alpha,
                                        cu_biases.v,
                                        biases.rp_elems(),
                                        &alpha, // aka 1
                                        cu_out.v,
                                        (void *)out.rp_elems()), "cudnnAddTensor_v3" );

      bool const conv_has_relu = op.get_u32( "conv_has_relu" );
      if( conv_has_relu ) {
        cudnn_err_chk( cudnnActivationForward( cdh,
                                               CUDNN_ACTIVATION_RELU,
                                               &alpha,
                                               cu_out.v,
                                               out.rp_elems(),
                                               &beta,
                                               cu_out.v,
                                               (void *)out.rp_elems()), "cudnnActivationForward" );
      }
    }

    void sgemm( p_map_str_p_nda_t const & args ) { 
      nda_t const & a = *must_find(*args,"arg_0");
      nda_t const & b = *must_find(*args,"arg_1");
      nda_t const & c = *must_find(*args,"arg_2");
      uint64_t const M = a.dims.dsz("M");
      uint64_t const K = a.dims.dsz("K");
      assert_st( b.dims.dsz("K") == K );
      uint64_t const N = b.dims.dsz("N");
      assert_st( c.dims.dsz("M") == M );
      assert_st( c.dims.dsz("N") == N );
      //printf( "calling cublas: a=%s b=%s c=%s\n", str(a).c_str(), str(b).c_str(), str(c).c_str() );
      // our inputs are row-major: a:KxM (pre-transposed), b:KxN; we want an output of c:MxN (row major);
      // if interpret our inputs as column-major, they are: at:MxK, b:NxK; so for col-major sgemm, we want -->
      // opA(A)=b opB(B)=a' --> b*a' = C:NxM (col major) --> 
      // so if we interpret C as row major, we get the desired c:MxN (row major)
      // so we want A=b opA=N, B=a opB=T, M=(our)N, N=(our)M, K=(our)K
      float const alpha = 1.0f;
      float const beta = 0.0f;
      cublas_err_chk( cublasSgemm( cbh, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, 
                                   &alpha,
                                   (float const *)(b.rp_elems()),  K, //const float           *A, int lda,
                                   (float const *)(a.rp_elems()),  K, //const float           *B, int ldb,
                                   &beta,
                                   (float *)(c.rp_elems()),  N)  //float           *C, int ldc)
                      ,"cublasSgemm" );
    }
  };
  void culibs_wrap_call( p_culibs_wrap_t const & cw, rtc_func_info_t const & fi, p_map_str_p_nda_t const & args ) {
    cw->call( fi, args );
  }
  p_culibs_wrap_t culibs_wrap_init( void ) { return make_shared< culibs_wrap_t >(); }
}
