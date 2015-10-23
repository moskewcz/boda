// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"CL/cl.hpp"
#include"ocl_err.H"
#include"rtc_compute.H"

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
  using cl::Event;

  typedef shared_ptr< Event > p_Event; 
  typedef vector< p_Event > vect_p_Event; 

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

  template< typename V > void set_kernel_arg( Kernel & k, uint32_t const & ai, V const & v ) { 
    cl_int err;
    err = k.setArg( ai, v );
    cl_err_chk( err, "Kernel::setArg()" );
  }

  typedef map< string, Kernel > map_str_Kernel_t;
  typedef shared_ptr< map_str_Kernel_t > p_map_str_Kernel_t;

  typedef map< string, Buffer > map_str_Buffer_t;
  typedef shared_ptr< map_str_Buffer_t > p_map_str_Buffer_t;

  struct cl_var_info_t {
    Buffer buf;
    dims_t dims;
    // p_void ev; // when ready
    cl_var_info_t( Buffer const & buf_, dims_t const & dims_ ) : buf(buf_), dims(dims_) {} // , ev( new Event ) {}
  };
  typedef map< string, cl_var_info_t > map_str_cl_var_info_t;
  typedef shared_ptr< map_str_cl_var_info_t > p_map_str_cl_var_info_t;

  string ocl_base_decls = R"rstr(
//typedef unsigned uint32_t;
typedef int int32_t;
//typedef long long int64_t;
#define CUCL_GLOBAL_KERNEL kernel
#define GASQ global
#define GLOB_ID_1D get_global_id(0)
#define LOC_ID_1D get_local_id(0)
#define GRP_ID_1D get_group_id(0)
#define LOC_SZ_1D get_local_size(0)
#define LOCSHAR_MEM local
#define LSMASQ local
#define BARRIER_SYNC barrier(CLK_LOCAL_MEM_FENCE)

// note: it seems OpenCL doesn't provide powf(), but instead overloads pow() for double and float. 
// so, we use this as a compatibility wrapper. 
// the casts should help uses that might expect implict casts from double->float when using powf() 
// ... or maybe that's a bad idea?
#define powf(v,e) pow((float)v,(float)e)

)rstr";


  struct ocl_compute_t : virtual public nesi, public rtc_compute_t // NESI(help="OpenCL based rtc support",
			   // bases=["rtc_compute_t"], type_id="ocl" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    vect_Platform platforms;
    vect_Device use_devices;
    Context context;
    CommandQueue cq;
    zi_bool init_done;
    void init( void ) {
      assert_st( !init_done.v );
      Platform::get(&platforms);
      if( platforms.empty() ) { rt_err( "no OpenCL platforms found" ); }
      for( vect_Platform::const_iterator i = platforms.begin(); i != platforms.end(); ++i ) {
	vect_Device devices;
	(*i).getDevices( CL_DEVICE_TYPE_GPU, &devices );
	if( !devices.empty() ) { use_devices = vect_Device{devices[0]}; } // pick first device only (arbitrarily)
      }
      if( use_devices.empty() ) { rt_err( "no OpenCL platform had any GPUs (devices of type CL_DEVICE_TYPE_GPU)" ); }
      cl_int err = CL_SUCCESS;  
      context = Context( use_devices, 0, 0, 0, &err );
      cl_err_chk( err, "cl::Context()" );
      // note: after this, we're only using the first device in use_devices, although our context is for all of
      // them. this is arguably not the most sensible thing to do in general.
      cq = CommandQueue( context, use_devices[0], CL_QUEUE_PROFILING_ENABLE, &err ); // note: not out of order
      cl_err_chk( err, "cl::CommandQueue()" );
      init_done.v = 1;
    }

    Program prog;
    zi_bool prog_valid;
    void compile( string const & cucl_src, bool const show_compile_log, bool const enable_lineinfo ) {
      string const src = ocl_base_decls + cucl_src;
      assert( init_done.v );
      assert( !prog_valid.v );
      write_whole_fn( "out.cl", src );
      cl_int err;
      prog = Program( context, src, 0, &err );
      cl_err_chk( err, "Program::Program() (just ctor, no build (build=0))" );
      err = prog.build( use_devices, "-cl-fast-relaxed-math -cl-denorms-are-zero" );
      cl_err_chk_build( err, prog, use_devices );
      prog_valid.v = 1;
    }

    p_map_str_cl_var_info_t vis;
    p_map_str_Kernel_t kerns;

    void copy_to_var( string const & vn, float const * const v ) {
      Buffer const & buf = must_find( *vis, vn ).buf;
      cl_int const err = cq.enqueueWriteBuffer( buf, 1, 0, buf.getInfo<CL_MEM_SIZE>(), &v[0], 0, 0);  // note: blocking write
      cl_err_chk( err, "cq->enqueueWriteBuffer()" );
    }
    void copy_from_var( float * const v, string const & vn ) {
      Buffer const & buf = must_find( *vis, vn ).buf;
      cl_int const err = cq.enqueueReadBuffer( buf, 1, 0, buf.getInfo<CL_MEM_SIZE>(), &v[0], 0, 0 ); // note: blocking_read=1
      cl_err_chk( err, "cq->enqueueReadBuffer()" );
    }
    void create_var_with_dims_floats( string const & vn, dims_t const & dims ) { 
      uint32_t const sz = dims.dims_prod();
      uint32_t const bytes_sz = sizeof(float)*sz;
      cl_int err;
      Buffer buf( context, CL_MEM_READ_WRITE, bytes_sz, 0, &err ); 
      cl_err_chk( err, "Buffer() from vect_float" );
      must_insert( *vis, vn, cl_var_info_t{buf,dims} ); 
    }
    uint32_t get_var_sz_floats( string const & vn ) { 
      uint32_t const bytes_sz = must_find( *vis, vn ).buf.getInfo<CL_MEM_SIZE>(); 
      assert_st( (bytes_sz%sizeof(float)) == 0 );
      return bytes_sz / sizeof(float);
    }
    void set_var_to_zero( string const & vn ) { 
      Buffer const & buf = must_find( *vis, vn ).buf;
#if 0
      cl_int const err = cq.enqueueFillBuffer( buf, ... ); // need OpenCL 1.2 ...
      cl_err_chk( err, "cq->enqueueFillBuffer()" );
#else
      uint32_t const bytes_sz = buf.getInfo<CL_MEM_SIZE>(); 
      vect_uint8_t zeros( bytes_sz, 0 );
      cl_int const err = cq.enqueueWriteBuffer( buf, 1, 0, bytes_sz, &zeros[0], 0, 0);  // note: blocking write
      cl_err_chk( err, "cq->enqueueWriteBuffer()" );
#endif
    }
    
    ocl_compute_t( void ) : vis( new map_str_cl_var_info_t ), kerns( new map_str_Kernel_t ) { }

    // note: post-compilation, MUST be called exactly once on all functions that will later be run()
    void check_runnable( string const name, bool const show_func_attrs ) {
      assert_st( prog_valid.v );
      cl_int err = 0;
      Kernel kern( prog, name.c_str(), &err );
      cl_err_chk( err, "cl::Kernel() (aka clCreateKernel())" );
      if( show_func_attrs ) {
	// FIXME: TODO
      }
      must_insert( *kerns, name, kern );
    }

    vect_p_Event call_evs;
    Event & get_call_ev( uint32_t const & call_id ) { assert_st( call_id < call_evs.size() ); return *call_evs[call_id]; }
    uint32_t alloc_call_id( void ) { call_evs.push_back( p_Event( new Event ) ); return call_evs.size() - 1; }
    virtual void release_per_call_id_data( void ) { call_evs.clear(); } // invalidates all call_ids inside rtc_func_call_t's

    virtual float get_dur( rtc_func_call_t const & b, rtc_func_call_t const & e ) {
      float compute_dur = 0.0f;
      cl_int err = 0;
      cl_ulong const et = get_call_ev(e.call_id).getProfilingInfo<CL_PROFILING_COMMAND_END>(&err);
      cl_err_chk( err, "cl::Event::getProfilingInfo() (end time)" );
      err = 0;
      cl_ulong const bt = get_call_ev(b.call_id).getProfilingInfo<CL_PROFILING_COMMAND_START>(&err);
      cl_err_chk( err, "cl::Event::getProfilingInfo() (start time)" );
      compute_dur = float(et - bt) / 1e6;
      //cu_err_chk( cuEventElapsedTime( &compute_dur, *(CUevent*)b.b_ev.get(), *(CUevent*)e.e_ev.get() ), "cuEventElapsedTime" );
      return compute_dur;
    }

    virtual float get_var_compute_dur( string const & vn ) { return 0; }
    virtual float get_var_ready_delta( string const & vn1, string const & vn2 ) { return 0; }

    void add_args( vect_string const & args, Kernel & kern, uint32_t & cur_arg_ix ) {
      for( vect_string::const_iterator i = args.begin(); i != args.end(); ++i ) {
	Buffer & buf = must_find( *vis, *i ).buf;
	set_kernel_arg( kern, cur_arg_ix, buf );
	++cur_arg_ix;
      }
    }

    void run( rtc_func_call_t & rfc ) {
      Kernel & kern = must_find( *kerns, rfc.rtc_func_name.c_str() );
      uint32_t cur_arg_ix = 0;
      add_args( rfc.in_args, kern, cur_arg_ix );
      add_args( rfc.inout_args, kern, cur_arg_ix );
      add_args( rfc.out_args, kern, cur_arg_ix );
      for( vect_uint32_t::iterator i = rfc.u32_args.begin(); i != rfc.u32_args.end(); ++i ) { 
	set_kernel_arg( kern, cur_arg_ix, *i );
	++cur_arg_ix;
      }

      rfc.call_id = alloc_call_id();
      cl_int const err = cq.enqueueNDRangeKernel( kern, cl::NullRange, NDRange(rfc.tpb.v*rfc.blks.v), NDRange(rfc.tpb.v), 0, 
						  &get_call_ev(rfc.call_id) );
      cl_err_chk( err, "cl::CommandQueue::enqueueNDRangeKernel()" );
    }

    void finish_and_sync( void ) { cl_err_chk( cq.finish(), "CommandQueue::finish()" ); }

    // FIXME: TODO
    void profile_start( void ) { }
    void profile_stop( void ) { }
  };
  
#include"gen/ocl_util.cc.nesi_gen.cc"
}
