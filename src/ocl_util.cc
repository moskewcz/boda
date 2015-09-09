// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"

// FIXME: remove after test prog removed
#include"rand_util.H"
#include"has_main.H"

#include"CL/cl.hpp"
#include"ocl_err.H"

namespace boda 
{

  void cl_err_chk( cl_int const & ret, char const * const tag ) {
    if( ret != CL_SUCCESS ) { rt_err( strprintf( "%s() failed with ret=%s (%s)", tag, str(ret).c_str(), get_cl_err_str(ret) ) ); } 
  }

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
