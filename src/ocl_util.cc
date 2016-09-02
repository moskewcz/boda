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


  template< typename T > void debug_rr( string const & tag, T const  & v ) {
    //printf( "%s %s v=%s\n", cxx_demangle(typeid(T).name()).c_str(), str(tag).c_str(), str(v).c_str() );
  }
  typedef vector< size_t > vect_size_t;
  // let's experiment with using the C bindings directly. it may be that this is easier/cleaner for us than using the
  // 'default' C++ wrappers? hmm. i keep getting lost in the 12K+ lines of cl.hpp ... here we only seem to need ~100
  // lines of wrappers, some of which we needed even when using the official cl.hpp wrappers.
  template< typename T > struct cl_release_T { typedef cl_int (*p_func)( T ); };
  template< typename T, typename cl_release_T<T>::p_func RELEASE, typename cl_release_T<T>::p_func RETAIN > struct cl_wrap_t {
    T v;
    cl_wrap_t( void ) : v(0) {}
    bool valid( void ) const { return v != 0; }
    void reset( void ) { 
      if( valid() ) { 
        debug_rr( "RELEASE (reset)", v );
        cl_int const ret = RELEASE(v); v = 0; cl_err_chk( ret, "clRelease<T>" );
      } 
    }
    void reset( T const & v_ ) { reset(); v = v_; } // assume ref cnt == 1
    ~cl_wrap_t( void ) { 
      debug_rr( "DTOR ", v );
      reset(); 
    }
    cl_wrap_t( cl_wrap_t const & o ) : v(0) { 
      if(o.valid()) { 
        debug_rr( "RETAIN (copy ctor)", o.v );
        cl_err_chk( RETAIN(o.v), "clRetain<T>");
        v = o.v; 
      } 
    }
    void operator = ( cl_wrap_t const & o ) { 
      if(this == &o) { return; } 
      debug_rr( "OP = (about to reset)", v );
      reset(); 
      if(o.valid()) { 
        debug_rr( "RETAIN (OP =)", o.v );
        cl_err_chk( RETAIN(o.v), "clRetain<T>");
        v = o.v; 
      }
    }
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
  struct Program_t {
    cl_program p; cl_program_info i;
    Program_t( cl_program const & p_, cl_program_info const & i_ ):p(p_),i(i_) {}
    cl_int operator ()( size_t const & pvs, void  * const & pv, size_t * const & pvsr ) const {
      return clGetProgramInfo( p, i, pvs, pv, pvsr ); }
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
  struct KernelWorkGroup_t {
    cl_kernel k; cl_device_id d; cl_kernel_work_group_info i;
    KernelWorkGroup_t( cl_kernel const & k_, cl_device_id const & d_, cl_kernel_work_group_info const & i_ ): k(k_),d(d_),i(i_) {}
    cl_int operator ()( size_t const & pvs, void  * const & pv, size_t * const & pvsr ) const {
      return clGetKernelWorkGroupInfo( k, d, i, pvs, pv, pvsr ); }
  };

  // get_info<T>() for (all) fixed-size types + get_info_str() for (var-sized) strings (i.e. char[] in OpenCL spec)
  template< typename T, typename GI > T get_info( GI const & gi ) {
    T ret; cl_int err = gi( sizeof(ret), &ret, 0 );
    cl_err_chk( err, "clGet...Info()" );
    return ret;
  }
  template< typename T, typename GI > T get_info_vect( GI const & gi ) {
    T ret; size_t sz; cl_int err; 
    err = gi( 0, 0, &sz ); cl_err_chk( err, "clGet...Info(get_size_of_var_sized_thing)" );
    size_t const elems = sz / sizeof(typename T::value_type);
    assert_st( elems*sizeof(typename T::value_type) == sz );
    ret.resize(elems); 
    err = gi( sz, &ret[0], 0 ); 
    cl_err_chk( err, "clGet...Info(get_var_sized_thing)" );
    return ret;
  }
  template< typename GI > string get_info_str( GI const & gi ) { return get_info_vect< string, GI >( gi ); }

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

  struct ocl_func_info_t {
    rtc_func_info_t info;
    cl_kernel_t kern;
  };
 
  typedef map< string, ocl_func_info_t > map_str_ocl_func_info_t;
  typedef shared_ptr< map_str_ocl_func_info_t > p_map_str_ocl_func_info_t;

  struct cl_var_info_t {
    cl_mem_t buf;
    dims_t dims;
    // p_void ev; // when ready
    cl_var_info_t( cl_mem_t const & buf_, dims_t const & dims_ ) : buf(buf_), dims(dims_) {} // , ev( new Event ) {}
    cl_var_info_t( cl_var_info_t const & src_vi, dims_t const & dims_ ) : buf(src_vi.buf), dims(dims_) {
      assert_st( dims.bytes_sz() == src_vi.dims.bytes_sz() );
    } 
  };
  typedef map< string, cl_var_info_t > map_str_cl_var_info_t;
  typedef shared_ptr< map_str_cl_var_info_t > p_map_str_cl_var_info_t;

  string ocl_base_decls = R"rstr(
typedef unsigned uint32_t;
typedef int int32_t;
typedef unsigned char uint8_t;

#define CUCL_BACKEND_IX 2
__constant uint32_t const U32_MAX = 0xffffffff;
//typedef long long int64_t;
#define CUCL_GLOBAL_KERNEL kernel
#define CUCL_DEVICE 
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
#define store_float_to_rp_half( val, ix, p ) vstore_half( val, ix, p )
#define store_float_to_rp_float( val, ix, p ) p[ix] = val

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
    void check_runnable( cl_program_t const & prog, rtc_func_info_t const & info, bool const show_func_attrs ) {
      assert_st( prog.valid() );
      cl_int err = 0;
      cl_kernel_t kern;
      kern.reset( clCreateKernel( prog.v, info.func_name.c_str(), &err ) );
      cl_err_chk( err, "clCreateKernel()" );
      if( show_func_attrs ) {
	// FIXME: TODO
      }
      must_insert( *kerns, info.func_name, ocl_func_info_t{info,kern} );
    }

    zi_uint32_t compile_call_ix;
    void compile( vect_rtc_func_info_t const & func_infos, rtc_compile_opts_t const & opts ) {
      timer_t t("ocl_compile");
      assert( init_done.v );
      if( func_infos.empty() ) { return; } // no work to do? don't compile just the base decls to no effect (slow).
      vect_rp_const_char srcs{ ocl_base_decls.c_str(), get_rtc_base_decls().c_str() };
      for( vect_rtc_func_info_t::const_iterator i = func_infos.begin(); i != func_infos.end(); ++i ) {
        srcs.push_back( i->func_src.c_str() );
      }      
      if( gen_src ) {
	ensure_is_dir( gen_src_output_dir.exp, 1 );
        p_ofstream out = ofs_open( strprintf( "%s/out_%s.cl", gen_src_output_dir.exp.c_str(), str(compile_call_ix.v).c_str() ));
        for( vect_rp_const_char::const_iterator i = srcs.begin(); i != srcs.end(); ++i ) { (*out) << (*i); }
      }
      cl_int err;
      cl_program_t prog;
      prog.reset( clCreateProgramWithSource( context.v, srcs.size(), &srcs[0], 0, &err ) );
      cl_err_chk( err, "clCreateProgramWithSource" );
      err = clBuildProgram( prog.v, use_devices.size(), &use_devices[0], "-cl-fast-relaxed-math -cl-denorms-are-zero", 0, 0 );
      cl_err_chk_build( err, prog.v, use_devices );
      if( gen_src ) {
        // get binaries
        cl_uint const num_devs = get_info<cl_uint>(Program_t(prog.v,CL_PROGRAM_NUM_DEVICES));
        assert_st( num_devs == 1 ); // FIXME: should only be one device here ever, right?
        vect_size_t pb_szs;
        pb_szs.resize( num_devs );

        cl_int const sz_err = Program_t(prog.v,CL_PROGRAM_BINARY_SIZES)( pb_szs.size()*sizeof(pb_szs[0]), &pb_szs[0], 0 );
        cl_err_chk( sz_err, "clGetProgramInfo(prog,CL_PROGRAM_BINARY_SIZES)" );

        string ocl_bin;
        ocl_bin.resize( pb_szs[0] );
        vect_rp_char pbs{ &ocl_bin[0] };
        cl_int const err = Program_t(prog.v,CL_PROGRAM_BINARIES)( pbs.size()*sizeof(pbs[0]), &pbs[0], 0 );
        cl_err_chk( err, "clGetProgramInfo(prog,CL_PROGRAM_BINARIES)" );
        
	write_whole_fn( strprintf( "%s/out_%s.clb", gen_src_output_dir.exp.c_str(), str(compile_call_ix.v).c_str() ), ocl_bin );
      }
      for( vect_rtc_func_info_t::const_iterator i = func_infos.begin(); i != func_infos.end(); ++i ) {
	check_runnable( prog, *i, opts.show_func_attrs );
      }
      ++compile_call_ix.v;
    }

    cl_mem_t null_buf; // inited to 0; used to pass null device pointers to kernels. note, however, that the value is
                       // generally unused, so the value doesn't really matter currently. it might later of course.
    p_map_str_cl_var_info_t vis;
    p_map_str_ocl_func_info_t kerns;
    virtual void release_all_funcs( void ) { kerns->clear(); }

    void copy_nda_to_var( string const & vn, p_nda_t const & nda ) {
      cl_var_info_t const & vi = must_find( *vis, vn );
      assert_st( nda->dims == vi.dims );
      size_t const buf_sz = get_info<size_t>(MemObject_t(vi.buf.v,CL_MEM_SIZE));
      assert_st( buf_sz == nda->dims.bytes_sz() );
      cl_int const err = clEnqueueWriteBuffer( cq.v, vi.buf.v, 1, 0, buf_sz, nda->rp_elems(), 0, 0, 0);  // note: blocking write
      cl_err_chk( err, "clEnqueueWriteBuffer()" );
    }
    void copy_var_to_nda( p_nda_t const & nda, string const & vn ) {
      cl_var_info_t const & vi = must_find( *vis, vn );
      assert_st( nda->dims == vi.dims );
      size_t const buf_sz = get_info<size_t>(MemObject_t(vi.buf.v,CL_MEM_SIZE));
      assert_st( buf_sz == nda->dims.bytes_sz() );
      cl_int const err = clEnqueueReadBuffer( cq.v, vi.buf.v, 1, 0, buf_sz, nda->rp_elems(), 0, 0, 0 ); // note: blocking_read=1
      cl_err_chk( err, "clEnqueueReadBuffer()" );
    }
    p_nda_t get_var_raw_native_pointer( string const & vn ) {
      rt_err( "ocl_compute_t: get_var_raw_native_pointer(): not implemented");
    }

    void create_var_with_dims( string const & vn, dims_t const & dims ) { 
      cl_int err;
      cl_mem_t buf;
      buf.reset( clCreateBuffer( context.v, CL_MEM_READ_WRITE, dims.bytes_sz(), 0, &err ) );
      cl_err_chk( err, "Buffer() from dims" );
      must_insert( *vis, vn, cl_var_info_t{buf,dims} ); 
      set_var_to_zero( vn );
    }
    void create_var_with_dims_as_reshaped_view_of_var( string const & vn, dims_t const & dims, string const & src_vn ) {
      cl_var_info_t const & src_vi = must_find( *vis, src_vn );
      rtc_reshape_check( dims, src_vi.dims );
      must_insert( *vis, vn, cl_var_info_t( src_vi, dims ) );
    }

    void release_var( string const & vn ) { must_erase( *vis, vn ); }
    dims_t get_var_dims( string const & vn ) { return must_find( *vis, vn ).dims; }
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
    
    ocl_compute_t( void ) : vis( new map_str_cl_var_info_t ), kerns( new map_str_ocl_func_info_t ) { }

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

    // FIXME: semi-dupe'd with nvrtc version, factor out somehow?
    void add_arg( rtc_arg_t const & arg, cl_kernel_t const & kern, uint32_t & cur_arg_ix ) {
      assert_st( arg.is_valid() );
      if( arg.is_var() ) { // pass-by-reference case
        set_kernel_arg( kern, cur_arg_ix, must_find( *vis, arg.n ).buf.v ); 
      } else if( arg.is_nda() ) { // pass-by-value and null-reference cases (yes, an odd pairing ...)
        assert_st( arg.v );
        if( !arg.v->rp_elems() ) { set_kernel_arg( kern, cur_arg_ix, null_buf.v ); } // null case (REFs, optional vars)
        else { // pass-by-value case
          cl_int err; err = clSetKernelArg( kern.v, cur_arg_ix, arg.v->dims.bytes_sz(), arg.v->rp_elems() );
          cl_err_chk( err, "clSetKernelArg() [by-value]" );
        }
      } else { assert_st(0); }
      ++cur_arg_ix;
    }

    void release_func( string const & func_name ) { must_erase( *kerns, func_name ); }

    uint32_t run( rtc_func_call_t const & rfc ) {
      ocl_func_info_t const & ofi = must_find( *kerns, rfc.rtc_func_name.c_str() );
      uint32_t cur_arg_ix = 0;
      for( vect_string::const_iterator i = ofi.info.arg_names.begin(); i != ofi.info.arg_names.end(); ++i ) {
        map_str_rtc_arg_t::const_iterator ai = rfc.arg_map.find( *i );
        // this error is almost an internal error, since the rtc_codegen_t level should ensure it doesn't happen:
        if( ai == rfc.arg_map.end() ) { rt_err( strprintf( "ocl_compute_t: arg '%s' not found in arg_map for call.\n",
                                                           str((*i)).c_str() ) ); }
        add_arg( ai->second, ofi.kern, cur_arg_ix );
      }
      rtc_launch_check_blks_and_tpb( rfc.rtc_func_name, rfc.blks.v, rfc.tpb.v );
      uint32_t const call_id = alloc_call_id();
      size_t const glob_work_sz = rfc.tpb.v*rfc.blks.v;
      size_t const loc_work_sz = rfc.tpb.v;
      size_t const kwgs = get_info<size_t>(KernelWorkGroup_t(ofi.kern.v,use_devices[0],CL_KERNEL_WORK_GROUP_SIZE));
      // printf( "kwgs=%s\n", str(kwgs).c_str() ); // might be handy to see; might indicate occupancy limits for kernel
      if( loc_work_sz > kwgs ) {
        rt_err( strprintf( "Error: can't run kernel: loc_work_sz is %s but OpenCL says max is %s for this kernel+device.\n", 
                           str(loc_work_sz).c_str(), str(kwgs).c_str() ) );
      }
      cl_event ev = 0;
      cl_int const err = clEnqueueNDRangeKernel( cq.v, ofi.kern.v, 1, 0, &glob_work_sz, &loc_work_sz, 0, 0, &ev);
      cl_err_chk( err, "clEnqueueNDRangeKernel()" );
      get_call_ev(call_id).reset(ev);
      return call_id;
    }

    void finish_and_sync( void ) { cl_err_chk( clFinish( cq.v ), "clFinish()" ); }

    // FIXME: TODO
    void profile_start( void ) { }
    void profile_stop( void ) { }
  };
  
#include"gen/ocl_util.cc.nesi_gen.cc"
}
