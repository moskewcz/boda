// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"CL/cl.h"
#include"ocl_err.H"
#include"rtc_compute.H"
#include"timers.H"

namespace boda 
{
  void cl_err_chk( cl_int const & ret, char const * const tag ) {
    if( ret != CL_SUCCESS ) { rt_err( strprintf( "%s() failed with ret=%s (%s)", tag, str(ret).c_str(), get_cl_err_str(ret) ) ); } 
  }

  // let's experiment with using the C bindings directly. it may be that this is easier/cleaner for us than using the
  // 'default' C++ wrappers? hmm. i keep getting lost in the 12K+ lines of cl.hpp ... here we only seem to need ~100
  // lines of wrappers, some of which we needed even when using the official cl.hpp wrappers.
  template< typename T > struct cl_release_T { typedef cl_int (*p_func)( T ); };
  template< typename T, typename cl_release_T<T>::p_func RELEASE, typename cl_release_T<T>::p_func RETAIN > struct cl_wrap_t {
    T v;
    cl_wrap_t( void ) : v(0) {}
    bool valid( void ) const { return v != 0; }
    void reset( void ) { 
      if( valid() ) { cl_int const ret = RELEASE(v); v = 0; cl_err_chk( ret, "clRelease<T>" ); } 
    }
    void reset( T const & v_ ) { reset(); v = v_; } // assume ref cnt == 1
    ~cl_wrap_t( void ) { reset(); }
    cl_wrap_t( cl_wrap_t const & o ) { if(o.valid()) { RETAIN(o.v); v = o.v; } }
    void operator = ( cl_wrap_t const & o ) { reset(); if(o.valid()) { RETAIN(o.v); v = o.v; }}
  };
  
  typedef cl_wrap_t< cl_program, clReleaseProgram, clRetainProgram > cl_program_t;
  typedef cl_wrap_t< cl_context, clReleaseContext, clRetainContext > cl_context_t;
  typedef cl_wrap_t< cl_kernel, clReleaseKernel, clRetainKernel > cl_kernel_t;
  typedef cl_wrap_t< cl_mem, clReleaseMemObject, clRetainMemObject > cl_mem_t;
  typedef cl_wrap_t< cl_command_queue, clReleaseCommandQueue, clRetainCommandQueue > cl_command_queue_t;
  typedef cl_wrap_t< cl_event, clReleaseEvent, clRetainEvent > cl_event_t;

  typedef vector< cl_event_t > vect_cl_event_t; 

  // wrappers around various clGet___Info funcitons (for use with get_info function)
  struct ProgramBuild_t {
    cl_program p; cl_device_id d; cl_program_build_info i;
    ProgramBuild_t( cl_program const & p_, cl_device_id const & d_, cl_program_build_info const & i_ ):p(p_),d(d_),i(i_) {}
    cl_int operator ()( size_t const & pvs, void  * const & pv, size_t * const & pvsr ) const {
      return clGetProgramBuildInfo( p, d, i, pvs, pv, pvsr ); }
  };
  struct Device_t {
    cl_device_id d; cl_device_info i;
    Device_t( cl_device_id const & d_, cl_device_info const & i_ ): d(d_),i(i_) {}
    cl_int operator ()( size_t const & pvs, void  * const & pv, size_t * const & pvsr ) const {
      return clGetDeviceInfo( d, i, pvs, pv, pvsr ); }
  };
  struct MemObject_t {
    cl_mem v; cl_mem_info i;
    MemObject_t( cl_mem const & v_, cl_mem_info const & i_ ): v(v_),i(i_) {}
    cl_int operator ()( size_t const & pvs, void  * const & pv, size_t * const & pvsr ) const {
      return clGetMemObjectInfo( v, i, pvs, pv, pvsr ); }
  };

  // get_info<T>() for (all) fixed-size types + get_info_str() for (var-sized) strings (i.e. char[] in OpenCL spec)
  template< typename T, typename GI > T get_info( GI const & gi ) {
    T ret; cl_int err = gi( sizeof(ret), &ret, 0 );
    cl_err_chk( err, "clGet...Info()" );
    return ret;
  }
  template< typename GI > string get_info_str( GI const & gi ) {
    string ret; size_t sz; cl_int err; 
    err = gi( 0, 0, &sz ); cl_err_chk( err, "clGet...Info(get_size_of_str)" );
    ret.resize(sz); err = gi( sz, &ret[0], 0 ); 
    cl_err_chk( err, "clGet...Info(get_str)" );
    return ret;
  }

  cl_ulong get_prof_info( cl_event_t const & event, cl_profiling_info const & pn ) {
    cl_ulong ret; cl_int err = clGetEventProfilingInfo( event.v, pn, sizeof(ret), &ret, 0 );
    cl_err_chk( err, "clGetEventProfilingInfo()" );
    return ret;
  }

  typedef vector< cl_platform_id > vect_cl_platform_id;
  void cl_get_platforms( vect_cl_platform_id & out ) {
    assert_st( out.empty() );
    cl_uint sz;
    cl_int err;
    err = clGetPlatformIDs( 0, 0, &sz ); cl_err_chk( err, "clGetPlatformIDs(size)" );
    out.resize( sz );
    err = clGetPlatformIDs( sz, &out[0], 0 ); cl_err_chk( err, "clGetPlatformIDs(out)" );
  } 


  typedef vector< cl_device_id > vect_cl_device_id;  
  void cl_get_devices( vect_cl_device_id & out, cl_platform_id platform, cl_device_type device_type ) {
    assert_st( out.empty() );
    cl_uint sz;
    cl_int err;
    err = clGetDeviceIDs( platform, device_type, 0, 0, &sz );
    if( err == CL_DEVICE_NOT_FOUND ) { return; } // leave out empty for no devices case
    cl_err_chk( err, "clGetDeviceIDs(size)" );
    out.resize( sz );
    err = clGetDeviceIDs( platform, device_type, sz, &out[0], 0 );
    cl_err_chk( err, "clGetDeviceIDs(out)" );
  } 

  void cl_err_chk_build( cl_int const & ret, cl_program const & program, vect_cl_device_id const & use_devices ) {
    if( ret != CL_SUCCESS ) {
      for( vect_cl_device_id::const_iterator i = use_devices.begin(); i != use_devices.end(); ++i ) {
	cl_device_id const & device = *i;
	string const bs = str(get_info<cl_build_status>(ProgramBuild_t(program,device,CL_PROGRAM_BUILD_STATUS)));
	string const bo = get_info_str(ProgramBuild_t(program,device,CL_PROGRAM_BUILD_OPTIONS));
	string const bl = get_info_str(ProgramBuild_t(program,device,CL_PROGRAM_BUILD_LOG));
	printf( "OpenCL build error (for device \"%s\"): build_status=%s build_options=\"%s\" build_log:\n%s\n", 
		get_info_str(Device_t(device,CL_DEVICE_NAME)).c_str(), str(bs).c_str(), str(bo).c_str(), str(bl).c_str() );
      }
      cl_err_chk( ret, "OpenCL build program");
    }
  }

  template< typename V > void set_kernel_arg( cl_kernel_t const & k, uint32_t const & ai, V const & v ) { 
    cl_int err; err = clSetKernelArg( k.v, ai, sizeof(v), &v );
    cl_err_chk( err, "clSetKernelArg()" );
  }

  typedef map< string, cl_kernel_t > map_str_cl_kernel_t;
  typedef shared_ptr< map_str_cl_kernel_t > p_map_str_cl_kernel_t;

  struct cl_var_info_t {
    cl_mem_t buf;
    dims_t dims;
    // p_void ev; // when ready
    cl_var_info_t( cl_mem_t const & buf_, dims_t const & dims_ ) : buf(buf_), dims(dims_) {} // , ev( new Event ) {}
  };
  typedef map< string, cl_var_info_t > map_str_cl_var_info_t;
  typedef shared_ptr< map_str_cl_var_info_t > p_map_str_cl_var_info_t;

  string ocl_base_decls = R"rstr(
typedef unsigned uint32_t;
__constant uint32_t const U32_MAX = 0xffffffff;
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
    vect_cl_platform_id platforms;
    vect_cl_device_id use_devices;
    cl_context_t context;
    cl_command_queue_t cq;
    zi_bool init_done;
    void init( void ) {
      assert_st( !init_done.v );
      cl_get_platforms( platforms );
      if( platforms.empty() ) { rt_err( "no OpenCL platforms found" ); }
      for( vect_cl_platform_id::const_iterator i = platforms.begin(); i != platforms.end(); ++i ) {
	vect_cl_device_id devices;
	cl_get_devices( devices, *i, CL_DEVICE_TYPE_GPU );
	if( !devices.empty() ) { use_devices = vect_cl_device_id{devices[0]}; } // pick first device only (arbitrarily)
      }
      if( use_devices.empty() ) { rt_err( "no OpenCL platform had any GPUs (devices of type CL_DEVICE_TYPE_GPU)" ); }
      cl_int err = CL_SUCCESS;  
      context.reset( clCreateContext( 0, use_devices.size(), &use_devices[0], 0, 0, &err ) );
      cl_err_chk( err, "clCreateContext()" );
      // note: after this, we're only using the first device in use_devices, although our context is for all of
      // them. this is arguably not the most sensible thing to do in general.
      cq.reset( clCreateCommandQueue( context.v, use_devices[0], CL_QUEUE_PROFILING_ENABLE, &err ) ); // note: not out of order
      cl_err_chk( err, "cl::CommandQueue()" );
      init_done.v = 1;
    }

    // note: post-compilation, MUST be called exactly once on all functions that will later be run()
    void check_runnable( cl_program_t const & prog, string const name, bool const show_func_attrs ) {
      assert_st( prog.valid() );
      cl_int err = 0;
      cl_kernel_t kern;
      kern.reset( clCreateKernel( prog.v, name.c_str(), &err ) );
      cl_err_chk( err, "clCreateKernel()" );
      if( show_func_attrs ) {
	// FIXME: TODO
      }
      must_insert( *kerns, name, kern );
    }

    void compile( string const & src, bool const show_compile_log, bool const enable_lineinfo,
		  vect_string const & func_names, bool const show_func_attrs ) {
      timer_t t("ocl_compile");
      string const ocl_src = ocl_base_decls + src;
      assert( init_done.v );
      write_whole_fn( "out.cl", ocl_src );
      char const * rp_src = ocl_src.c_str();
      cl_int err;
      cl_program_t prog;
      prog.reset( clCreateProgramWithSource( context.v, 1, &rp_src, 0, &err ) );
      cl_err_chk( err, "clCreateProgramWithSource" );
      err = clBuildProgram( prog.v, use_devices.size(), &use_devices[0], "-cl-fast-relaxed-math -cl-denorms-are-zero", 0, 0 );
      cl_err_chk_build( err, prog.v, use_devices );
      for( vect_string::const_iterator i = func_names.begin(); i != func_names.end(); ++i ) {
	check_runnable( prog, *i, show_func_attrs );
      }
    }

    cl_mem_t null_buf; // inited to 0; used to pass null device pointers to kernels. note, however, that the value is
                       // generally unused, so the value doesn't really matter currently. it might later of course.
    p_map_str_cl_var_info_t vis;
    p_map_str_cl_kernel_t kerns;
    virtual void release_all_funcs( void ) { kerns->clear(); }


    void copy_to_var( string const & vn, float const * const v ) {
      cl_mem_t const & buf = must_find( *vis, vn ).buf;
      cl_int const err = clEnqueueWriteBuffer( cq.v, buf.v, 1, 
					       0, get_info<size_t>(MemObject_t(buf.v,CL_MEM_SIZE)), &v[0], 
					       0, 0, 0);  // note: blocking write
      cl_err_chk( err, "clEnqueueWriteBuffer()" );
    }
    void copy_from_var( float * const v, string const & vn ) {
      cl_mem_t const & buf = must_find( *vis, vn ).buf;
      cl_int const err = clEnqueueReadBuffer( cq.v, buf.v, 1, 
					      0, get_info<size_t>(MemObject_t(buf.v,CL_MEM_SIZE)), &v[0], 
					      0, 0, 0 ); // note: blocking_read=1
      cl_err_chk( err, "clEnqueueReadBuffer()" );
    }
    void create_var_with_dims_floats( string const & vn, dims_t const & dims ) { 
      uint32_t const sz = dims.dims_prod();
      uint32_t const bytes_sz = sizeof(float)*sz;
      cl_int err;
      cl_mem_t buf;
      buf.reset( clCreateBuffer( context.v, CL_MEM_READ_WRITE, bytes_sz, 0, &err ) );
      cl_err_chk( err, "Buffer() from vect_float" );
      must_insert( *vis, vn, cl_var_info_t{buf,dims} ); 
      set_var_to_zero( vn );
    }
    dims_t get_var_dims_floats( string const & vn ) { return must_find( *vis, vn ).dims; }
    void set_var_to_zero( string const & vn ) { 
      cl_mem_t const & buf = must_find( *vis, vn ).buf;
#if 0
      cl_int const err = cq.enqueueFillBuffer( buf, ... ); // need OpenCL 1.2 ...
      cl_err_chk( err, "cq->enqueueFillBuffer()" );
#else
      uint32_t const bytes_sz = get_info<size_t>(MemObject_t(buf.v,CL_MEM_SIZE));
      vect_uint8_t zeros( bytes_sz, 0 );
      cl_int const err = clEnqueueWriteBuffer( cq.v, buf.v, 1, 
					       0, bytes_sz, &zeros[0],
					       0, 0, 0);  // note: blocking write
      cl_err_chk( err, "clEnqueueWriteBuffer()" );
#endif
    }
    
    ocl_compute_t( void ) : vis( new map_str_cl_var_info_t ), kerns( new map_str_cl_kernel_t ) { }

    vect_cl_event_t call_evs;
    cl_event_t & get_call_ev( uint32_t const & call_id ) { assert_st( call_id < call_evs.size() ); return call_evs[call_id]; }
    uint32_t alloc_call_id( void ) { call_evs.push_back( cl_event_t() ); return call_evs.size() - 1; }
    virtual void release_per_call_id_data( void ) { call_evs.clear(); } // invalidates all call_ids inside rtc_func_call_t's

    virtual float get_dur( uint32_t const & b, uint32_t const & e ) {
      float compute_dur = 0.0f;
      cl_ulong const et = get_prof_info( get_call_ev(e), CL_PROFILING_COMMAND_END );
      cl_ulong const bt = get_prof_info( get_call_ev(b), CL_PROFILING_COMMAND_START );
      compute_dur = float(et - bt) / 1e6;
      return compute_dur;
    }

    virtual float get_var_compute_dur( string const & vn ) { return 0; }
    virtual float get_var_ready_delta( string const & vn1, string const & vn2 ) { return 0; }

    void add_args( vect_string const & args, cl_kernel_t const & kern, uint32_t & cur_arg_ix ) {
      for( vect_string::const_iterator i = args.begin(); i != args.end(); ++i ) {
	cl_mem * buf = 0;
	if( *i == "<NULL>" ) { buf = &null_buf.v; }
	else { buf = &must_find( *vis, *i ).buf.v; }
	set_kernel_arg( kern, cur_arg_ix, *buf );
	++cur_arg_ix;
      }
    }

    void run( rtc_func_call_t & rfc ) {
      cl_kernel_t const & kern = must_find( *kerns, rfc.rtc_func_name.c_str() );
      uint32_t cur_arg_ix = 0;
      add_args( rfc.in_args, kern, cur_arg_ix );
      add_args( rfc.inout_args, kern, cur_arg_ix );
      add_args( rfc.out_args, kern, cur_arg_ix );
      for( vect_uint32_t::iterator i = rfc.u32_args.begin(); i != rfc.u32_args.end(); ++i ) { 
	set_kernel_arg( kern, cur_arg_ix, *i );
	++cur_arg_ix;
      }

      rfc.call_id = alloc_call_id();
      size_t const glob_work_sz = rfc.tpb.v*rfc.blks.v;
      size_t const loc_work_sz = rfc.tpb.v;
      cl_int const err = clEnqueueNDRangeKernel( cq.v, kern.v, 1, 0, &glob_work_sz, &loc_work_sz, 0, 0, &get_call_ev(rfc.call_id).v);
      cl_err_chk( err, "clEnqueueNDRangeKernel()" );
    }

    void finish_and_sync( void ) { cl_err_chk( clFinish( cq.v ), "clFinish()" ); }

    // FIXME: TODO
    void profile_start( void ) { }
    void profile_stop( void ) { }
  };
  
#include"gen/ocl_util.cc.nesi_gen.cc"
}
