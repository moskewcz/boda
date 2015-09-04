// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"rand_util.H"
#include"has_main.H"
#include"has_conv_fwd.H"
#include"timers.H"
#include<boost/filesystem.hpp>
#include<boost/lexical_cast.hpp>
#include"lexp.H"
#include"CL/cl.hpp"
#include"ocl_err.H"

namespace boda 
{
  using boost::filesystem::path;

  void cl_err_chk( cl_int const & ret, char const * const tag ) {
    if( ret != CL_SUCCESS ) { rt_err( strprintf( "%s() failed with ret=%s (%s)", tag, str(ret).c_str(), get_cl_err_str(ret) ) ); } 
  }


#if 0
  void nvrtc_err_chk( nvrtcResult const & ret, char const * const func_name ) {
    if( ret != NVRTC_SUCCESS ) { rt_err( strprintf( "%s() failed with ret=%s (%s)", func_name, str(ret).c_str(), nvrtcGetErrorString(ret) ) ); } }
  void nvrtcDestroyProgram_wrap( nvrtcProgram p ) { if(!p){return;} nvrtc_err_chk( nvrtcDestroyProgram( &p ), "nvrtcDestroyProgram" ); }
  typedef shared_ptr< _nvrtcProgram > p_nvrtcProgram;

  
  p_nvrtcProgram make_p_nvrtcProgram( string const & cuda_prog_str ) { 
    nvrtcProgram p;
    nvrtc_err_chk( nvrtcCreateProgram( &p, &cuda_prog_str[0], "boda_cuda_gen", 0, 0, 0 ), "nvrtcCreateProgram" );
    return p_nvrtcProgram( p, nvrtcDestroyProgram_wrap ); 
  }
  string nvrtc_get_compile_log( p_nvrtcProgram const & cuda_prog ) {
    string ret;
    size_t ret_sz = 0;
    nvrtc_err_chk( nvrtcGetProgramLogSize( cuda_prog.get(), &ret_sz ), "nvrtcGetProgramLogSize" );
    ret.resize( ret_sz );    
    nvrtc_err_chk( nvrtcGetProgramLog( cuda_prog.get(), &ret[0] ), "nvrtcGetProgramLog" );
    return ret;
  }
  string nvrtc_get_ptx( p_nvrtcProgram const & cuda_prog ) {
    string ret;
    size_t ret_sz = 0;
    nvrtc_err_chk( nvrtcGetPTXSize( cuda_prog.get(), &ret_sz ), "nvrtcGetPTXSize" );
    ret.resize( ret_sz );    
    nvrtc_err_chk( nvrtcGetPTX( cuda_prog.get(), &ret[0] ), "nvrtcGetPTX" );
    return ret;
  }

  // FIXME: add function to get SASS? can use this command sequence:
  // ptxas out.ptx -arch sm_52 -o out.cubin ; nvdisasm out.cubin > out.sass

  string nvrtc_compile( string const & cuda_prog_str, bool const & print_log, bool const & enable_lineinfo ) {
    timer_t t("nvrtc_compile");
    p_nvrtcProgram cuda_prog = make_p_nvrtcProgram( cuda_prog_str );
    vect_string cc_opts = {"--use_fast_math",
			   "--gpu-architecture=compute_52",
			   "--restrict"};
    if( enable_lineinfo ) { cc_opts.push_back("-lineinfo"); }
    auto const comp_ret = nvrtcCompileProgram( cuda_prog.get(), cc_opts.size(), &get_vect_rp_const_char( cc_opts )[0] );
    string const log = nvrtc_get_compile_log( cuda_prog );
    if( print_log ) { printf( "NVRTC COMPILE LOG:\n%s\n", str(log).c_str() ); }
    nvrtc_err_chk( comp_ret, ("nvrtcCompileProgram\n"+log).c_str() ); // delay error check until after getting log
    return nvrtc_get_ptx( cuda_prog );
  }

#ifdef CU_GET_FUNC_ATTR_HELPER_MACRO
#error
#endif
#define CU_GET_FUNC_ATTR_HELPER_MACRO( cf, attr ) cu_get_func_attr( cf, attr, #attr )  
  string cu_get_func_attr( CUfunction const & cf, CUfunction_attribute const & cfa, char const * const & cfa_str ) {
    int cfav = 0;
    cu_err_chk( cuFuncGetAttribute( &cfav, cfa, cf ), "cuFuncGetAttribute" );
    return strprintf( "  %s=%s\n", str(cfa_str).c_str(), str(cfav).c_str() );
  }
  string cu_get_all_func_attrs( CUfunction const & cf ) {
    string ret;
    ret += CU_GET_FUNC_ATTR_HELPER_MACRO( cf, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK );
    ret += CU_GET_FUNC_ATTR_HELPER_MACRO( cf, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES );
    ret += CU_GET_FUNC_ATTR_HELPER_MACRO( cf, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES );
    ret += CU_GET_FUNC_ATTR_HELPER_MACRO( cf, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES );
    ret += CU_GET_FUNC_ATTR_HELPER_MACRO( cf, CU_FUNC_ATTRIBUTE_NUM_REGS );
    ret += CU_GET_FUNC_ATTR_HELPER_MACRO( cf, CU_FUNC_ATTRIBUTE_PTX_VERSION );
    ret += CU_GET_FUNC_ATTR_HELPER_MACRO( cf, CU_FUNC_ATTRIBUTE_BINARY_VERSION );
    ret += CU_GET_FUNC_ATTR_HELPER_MACRO( cf, CU_FUNC_ATTRIBUTE_CACHE_MODE_CA );
    return ret;
  }
#undef CU_GET_FUNC_ATTR_HELPER_MACRO

  
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
  typedef cup_T< float > cup_float;
  typedef shared_ptr< cup_float > p_cup_float; 

  // rp_float <-> cup_float
  void cu_copy_to_cup( p_cup_float const & cup, float const * const v, uint32_t const sz ) {
    cu_err_chk( cuMemcpyHtoD( cup->p, v, sz*sizeof(float) ), "cuMemcpyHtoD" );
  }
  void cu_copy_from_cup( float * const v, p_cup_float const & cup, uint32_t const sz ) {
    cu_err_chk( cuMemcpyDtoH( v, cup->p, sz*sizeof(float) ), "cuMemcpyDtoH" );
  }
  // nda_float <-> cup_float
  void cu_copy_nda_to_cup( p_cup_float const & cup, p_nda_float_t const & nda ) {
    assert_st( nda->elems.sz == cup->sz );
    cu_copy_to_cup( cup, &nda->elems[0], cup->sz );
  }
  void cu_copy_cup_to_nda( p_nda_float_t const & nda, p_cup_float const & cup ) {
    assert_st( nda->elems.sz == cup->sz );
    cu_copy_from_cup( &nda->elems[0], cup, cup->sz );
  }
  // vect_float <-> cup_float
  p_cup_float get_cup_copy( vect_float const & v ) { 
    p_cup_float ret = make_shared<cup_float>( v.size() ); 
    cu_copy_to_cup( ret, &v[0], v.size() ); 
    return ret; 
  }
  void set_from_cup( vect_float & v, p_cup_float const & cup ) {
    assert_st( cup->sz == v.size() );
    cu_copy_to_cup( cup, &v[0], v.size() );
  }
  
#endif

#if 0
/*
def run_mxplusb( prog, queue ):
    func = prog['mxplusb']
    print func
    func.argtypes = (cl.cl_float, cl.cl_mem, cl.cl_float, cl.cl_mem)
    x = cl.array('f', range(100))
    x_buf, in_evt = cl.buffer_from_pyarray(queue, x, blocking=False)
    y_buf = x_buf.empty_like_this()
    run_evt = func(2, x_buf, 5, y_buf).on(queue, len(x), wait_for=in_evt)
    y, evt = cl.buffer_to_pyarray(queue, y_buf, wait_for=run_evt, like=x)
    evt.wait()
    print y[0:10]
*/
#endif
  
  using cl::Platform;
  typedef vector< Platform > vect_Platform;
  using cl::Device;
  typedef vector< Device > vect_Device;
  using cl::Context;
  using cl::Program;
  using cl::Kernel;
  using cl::CommandQueue;
  using cl::Buffer;
  using cl::NDRange;

  void cl_err_chk_build( cl_int const & ret, Program const & program, vect_Device const & use_devices ) {
    if( ret != CL_SUCCESS ) {
      for( vect_Device::const_iterator i = use_devices.begin(); i != use_devices.end(); ++i ) {
	Device const & device = *i;
	string const bs = str(program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device));
	string const bo = program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(device);
	string const bl = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
	printf( "OpenCL build error (for device \"%s\"): build_status=%s build_options=\"%s\" build_log:\n%s\n", 
		device.getInfo<CL_DEVICE_NAME>().c_str(), str(bs).c_str(), str(bo).c_str(), str(bl).c_str() );
      }
      cl_err_chk( ret, "OpenCL build program");
    }
  }

  // create a cl::Buffer() of the same size as the passed vect_float. if cq is non-null, also enqueue a non-blocking write of
  // the contents of v into the returned buffer.
  Buffer make_buf_from_vect_float( Context const & context, vect_float const & v, CommandQueue * const cq ) { 
    cl_int err;
    uint32_t const sz = sizeof(float)*v.size();
    Buffer ret( context, CL_MEM_READ_WRITE, sz, 0, &err ); 
    cl_err_chk( err, "Buffer() from vect_float" );
    if( cq ) { 
      err = cq->enqueueWriteBuffer( ret, 0, 0, sz, &v[0], 0, 0); 
      cl_err_chk( err, "cq->enqueueWriteBuffer()" );
    }
    return ret;
  }

  template< typename V > void set_kernel_arg( Kernel & k, uint32_t const & ai, V const & v ) { 
    cl_int err;
    err = k.setArg( ai, v );
    cl_err_chk( err, "Kernel::setArg()" );
  }

  struct ocl_test_t : virtual public nesi, public has_main_t // NESI(help="test basic usage of openCL",
		      // bases=["has_main_t"], type_id="ocl_test")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t prog_fn; //NESI(default="%(boda_test_dir)/ocl_test_dot.cl",help="cuda program source filename")
    uint32_t data_sz; //NESI(default=10000,help="size in floats of test data")

    boost::random::mt19937 gen;
    
    virtual void main( nesi_init_arg_t * nia ) {
      vect_Platform platforms;
      Platform::get(&platforms);
      if( platforms.empty() ) { rt_err( "no OpenCL platforms found" ); }
      vect_Device use_devices;
      for( vect_Platform::const_iterator i = platforms.begin(); i != platforms.end(); ++i ) {
	vect_Device devices;
	(*i).getDevices( CL_DEVICE_TYPE_GPU, &devices );
	if( !devices.empty() ) { use_devices = vect_Device{devices[0]}; } // pick first device only (arbitrarily)
      }
      if( use_devices.empty() ) { rt_err( "no OpenCL platform had any GPUs (devices of type CL_DEVICE_TYPE_GPU)" ); }
      cl_int err = CL_SUCCESS;  
      Context context( use_devices, 0, 0, 0, &err );
      cl_err_chk( err, "cl::Context()" );
      p_string prog_str = read_whole_fn( prog_fn );
      Program prog( context, *prog_str, 1, &err );
      cl_err_chk_build( err, prog, use_devices );
      Kernel my_dot( prog, "my_dot", &err );
      cl_err_chk( err, "cl::Kernel() (aka clCreateKernel())" );

      // note: after this, we're only using the first device in use_devices, although our context is for all of
      // them. this is arguably not the most sensible thing to do in general.
      CommandQueue cq( context, use_devices[0], 0, &err ); // note: not out of order, no profiling
      cl_err_chk( err, "cl::CommandQueue()" );

      vect_float a( data_sz, 0.0f );
      rand_fill_vect( a, 2.5f, 7.5f, gen );
      vect_float b( data_sz, 0.0f );
      rand_fill_vect( b, 2.5f, 7.5f, gen );
      vect_float c( data_sz, 123.456f );

      Buffer d_a = make_buf_from_vect_float( context, a, &cq );
      Buffer d_b = make_buf_from_vect_float( context, b, &cq );
      Buffer d_c = make_buf_from_vect_float( context, c, &cq );

      uint32_t const n = a.size();
      set_kernel_arg( my_dot, 0, d_a );
      set_kernel_arg( my_dot, 1, d_b );
      set_kernel_arg( my_dot, 2, d_c );
      set_kernel_arg( my_dot, 3, n );

      err = cq.enqueueNDRangeKernel( my_dot, cl::NullRange, NDRange(n), cl::NullRange, 0, 0 );
      cl_err_chk( err, "cl::CommandQueue::enqueueNDRangeKernel()" );
       
#if 0
      p_cup_float cu_a = get_cup_copy(a);
      p_cup_float cu_b = get_cup_copy(b);
      p_cup_float cu_c = get_cup_copy(c); // or, for no init: make_shared<cup_float>( c.size() );

      uint32_t const tpb = 256;
      uint32_t const num_blocks = u32_ceil_div( data_sz, tpb );
      vect_rp_void cu_func_args{ &cu_a->p, &cu_b->p, &cu_c->p, &data_sz };
      {
	timer_t t("cu_launch_and_sync");
	cu_err_chk( cuLaunchKernel( cu_func,
				    num_blocks, 1, 1, // grid x,y,z dims
				    tpb, 1, 1, // block x,y,z dims
				    0, 0, // smem_bytes, stream_ix
				    &cu_func_args[0], // cu_func's args
				    0 ), "cuLaunchKernel" ); // unused 'extra' arg-passing arg
	cu_err_chk( cuCtxSynchronize(), "cuCtxSynchronize" );
      }
      set_from_cup( c, cu_c );
#endif

      err = cq.enqueueReadBuffer( d_c, 1, 0, sizeof(float)*c.size(), &c[0], 0, 0 ); // note: blocking_read=1
      cl_err_chk( err, "cq->enqueueReadBuffer()" );

      assert_st( b.size() == a.size() );
      assert_st( c.size() == a.size() );
      for( uint32_t i = 0; i != c.size(); ++i ) {
	if( fabs((a[i]+b[i]) - c[i]) > 1e-6f ) {
	  printf( "bad res: a[i]=%s b[i]=%s c[i]=%s\n", str(a[i]).c_str(), str(b[i]).c_str(), str(c[i]).c_str() );
	  break;
	}
      }
    }
  };

  
#include"gen/ocl_util.cc.nesi_gen.cc"
}
