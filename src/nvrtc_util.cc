// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"nvrtc_util.H"
#include"str_util.H"
#include"timers.H"
#include<nvrtc.h>
#include<cuda.h>
#include<cudaProfiler.h>
#include"rtc_compute.H"
#include"culibs-wrap.H"

namespace boda 
{

  void nvrtc_err_chk( nvrtcResult const & ret, char const * const func_name ) {
    if( ret != NVRTC_SUCCESS ) { rt_err( strprintf( "%s() failed with ret=%s (%s)", func_name, str(ret).c_str(), nvrtcGetErrorString(ret) ) ); } }
  void nvrtcDestroyProgram_wrap( nvrtcProgram p ) { if(!p){return;} nvrtc_err_chk( nvrtcDestroyProgram( &p ), "nvrtcDestroyProgram" ); }
  typedef shared_ptr< _nvrtcProgram > p_nvrtcProgram;

  void cu_err_chk( CUresult const & ret, char const * const func_name ) {
    if( ret != CUDA_SUCCESS ) { 
      char const * ret_name;
      char const * ret_str;
      assert_st( cuGetErrorName( ret, &ret_name ) == CUDA_SUCCESS );
      assert_st( cuGetErrorString( ret, &ret_str ) == CUDA_SUCCESS );
      rt_err( strprintf( "%s() failed with ret=%s (%s)", func_name, ret_name, ret_str ) );
    }
  }
  
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

  
  struct cup_t {
    CUdeviceptr p;
    uint64_t sz;
    void set_to_zero( void ) { cu_err_chk( cuMemsetD8(  p, 0, sz ), "cuMemsetD8" ); }
    cup_t( uint64_t const sz_ ) : p(0), sz(sz_) { 
      cu_err_chk( cuMemAlloc( &p, sz ), "cuMemAlloc" ); 
      set_to_zero();
    }
    ~cup_t( void ) { cu_err_chk( cuMemFree( p ), "cuMemFree" ); }
  };
  typedef shared_ptr< cup_t > p_cup_t; 

  typedef map< string, p_cup_t > map_str_p_cup_t;
  typedef shared_ptr< map_str_p_cup_t > p_map_str_p_cup_t;

  typedef shared_ptr< CUevent > p_CUevent; 
  typedef vector< p_CUevent > vect_p_CUevent; 
  void cuEventDestroy_wrap( CUevent const * const p ) { 
    if(!p){return;} 
    cu_err_chk( cuEventDestroy( *p ), "cuEventDestroy" ); 
  }
  p_CUevent make_p_CUevent( void ) {
    CUevent ret;
    cu_err_chk( cuEventCreate( &ret, CU_EVENT_DEFAULT ), "cuEventCreate" );
    return p_CUevent( new CUevent( ret ), cuEventDestroy_wrap ); 
  }

  typedef shared_ptr< CUmodule > p_CUmodule; 
  void cuModuleUnload_wrap( CUmodule const * const p ) { 
    if(!p){return;}
    cu_err_chk( cuModuleUnload( *p ), "cuModuleUnload" ); // FIXME/NOTE: breaks cuda-memcheck, but otherwise no effect?
  }
  p_CUmodule make_p_CUmodule( CUmodule to_own ) { return p_CUmodule( new CUmodule( to_own ), cuModuleUnload_wrap ); }

  // unlink opencl, functions implicity tied to the lifetime of thier module. if we use one-module-per-function, we need
  // to keep a ref to the module alongside the func so we can free the module when we want to free the function. also,
  // note that there is no reference counting for the module in the CUDA driver API, just load/unload, so we'd need to
  // do that ourselves. this might not be needed currently, but allows for arbitrary funcs->modules mappings.
  struct nv_func_info_t {
    rtc_func_info_t info;
    CUfunction func;
    p_CUmodule mod;
  };

  typedef map< string, nv_func_info_t > map_str_nv_func_info_t;
  typedef shared_ptr< map_str_nv_func_info_t > p_map_str_nv_func_info_t; 

  struct call_ev_t {
    p_CUevent b_ev;
    p_CUevent e_ev;
    call_ev_t( void ) : b_ev( make_p_CUevent() ), e_ev( make_p_CUevent() ) { }
  };
  typedef vector< call_ev_t > vect_call_ev_t; 

  struct var_info_t {
    p_cup_t cup;
    dims_t dims;
    //p_void ev; // when ready
    var_info_t( dims_t const & dims_ ) : cup( make_shared<cup_t>( dims_.bytes_sz() ) ), dims(dims_) {} // , ev( make_p_CUevent() ) { }
    var_info_t( var_info_t const & src_vi, dims_t const & dims_ ) : cup( src_vi.cup ), dims(dims_) {
      assert_st( dims.bytes_sz() == src_vi.dims.bytes_sz() );
    } 
  };

  typedef map< string, var_info_t > map_str_var_info_t;
  typedef shared_ptr< map_str_var_info_t > p_map_str_var_info_t;

  string cu_base_decls = R"rstr(
#define CUCL_BACKEND_IX 1
typedef unsigned uint32_t;
uint32_t const U32_MAX = 0xffffffffU;
typedef int int32_t;
//typedef long long int64_t;
float const FLT_MAX = /*0x1.fffffep127f*/ 340282346638528859811704183484516925440.0f;
float const FLT_MIN = 1.175494350822287507969e-38f;
#define CUCL_GLOBAL_KERNEL extern "C" __global__
#define CUCL_DEVICE extern "C" __device__
#define GASQ
#define GLOB_ID_1D (blockDim.x * blockIdx.x + threadIdx.x)
#define LOC_ID_1D (threadIdx.x)
#define GRP_ID_1D (blockIdx.x)
#define LOC_SZ_1D (blockDim.x)
#define LOCSHAR_MEM __shared__
#define LSMASQ
#define BARRIER_SYNC __syncthreads()

#define store_float_to_rp_half( val, ix, p ) vstore_half( val, ix, p )
#define store_float_to_rp_float( val, ix, p ) p[ix] = val

)rstr";

  struct nvrtc_compute_t : virtual public nesi, public rtc_compute_t // NESI(help="libnvrtc based rtc support (i.e. CUDA)",
			   // bases=["rtc_compute_t"], type_id="nvrtc" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    // FIXME: can/should we init these cu_* vars?
    CUdevice cu_dev;
    CUcontext cu_context;
    p_culibs_wrap_t cw; // wrapper for handle to nVidia closed libs (if availible)
    zi_bool init_done;
    void init( void ) {
      assert_st( !init_done.v );
      null_cup = 0;

      cu_err_chk( cuInit( 0 ), "cuInit" ); 
      cu_err_chk( cuDeviceGet( &cu_dev, 0 ), "cuDeviceGet" );
      //cu_err_chk( cuCtxCreate( &cu_context, 0, cu_dev ), "cuCtxCreate" );
      cu_err_chk( cuDevicePrimaryCtxRetain( &cu_context, cu_dev ), "cuDevicePrimaryCtxRetain" );
      cu_err_chk( cuCtxSetCurrent( cu_context ), "cuCtxSetCurrent" ); // is this always needed/okay?
      // cu_err_chk( cuCtxSetCacheConfig( CU_FUNC_CACHE_PREFER_L1 ), "cuCtxSetCacheConfig" ); // does nothing?
      cw = culibs_wrap_init(); // creates cublas handle, cudnn handle, etc ...
      init_done.v = 1;
    }

    zi_uint32_t compile_call_ix;
    void compile( vect_rtc_func_info_t const & func_infos, rtc_compile_opts_t const & opts ) {
#if 0
      // for now, this is disabled, since:
      // 1) it doesn't really help (much) yet (we still need to compile other function in all flows
      // 2) we now need to create the nv_func_info object in all cases to cache the func info (even if we need not compile)
      bool all_funcs_culibs = 1;
      for( vect_rtc_func_info_t::const_iterator i = func_infos.begin(); i != func_infos.end(); ++i ) {
        string const & fn = i->func_name;
        if( !(startswith( "cublas_", fn ) || startswith( fn, "cudnn_" ) ) ) { all_funcs_culibs = 0; }
      }
      if( all_funcs_culibs ) { return; } // skip unneeded compilation if all funcs to compile are culibs stubs
#endif
      string cucl_src = cu_base_decls;
      for( vect_rtc_func_info_t::const_iterator i = func_infos.begin(); i != func_infos.end(); ++i ) {
        cucl_src += i->func_src;
      }
      assert( init_done.v );
      if( gen_src ) { ensure_is_dir( gen_src_output_dir.exp, 1 ); }
      if( gen_src ) {
	write_whole_fn( strprintf( "%s/out_%s.cu", gen_src_output_dir.exp.c_str(), str(compile_call_ix.v).c_str() ), cucl_src );
      }
      string const prog_ptx = nvrtc_compile( cucl_src, opts.show_compile_log, opts.enable_lineinfo );
      if( gen_src ) {      
	write_whole_fn( strprintf( "%s/out_%s.ptx", gen_src_output_dir.exp.c_str(), str(compile_call_ix.v).c_str() ), prog_ptx );
      }
      CUmodule new_cu_mod;
      cu_err_chk( cuModuleLoadDataEx( &new_cu_mod, prog_ptx.c_str(), 0, 0, 0 ), "cuModuleLoadDataEx" );
      p_CUmodule cu_mod = make_p_CUmodule( new_cu_mod );
      for( vect_rtc_func_info_t::const_iterator i = func_infos.begin(); i != func_infos.end(); ++i ) {
	check_runnable( cu_mod, *i, opts.show_func_attrs );
      }
      ++compile_call_ix.v;
    }

    // note: post-compilation, MUST be called exactly once on all functions that will later be run()
    void check_runnable( p_CUmodule const & cu_mod, rtc_func_info_t const & info, bool const show_func_attrs ) {
      assert_st( cu_mod );
      CUfunction cu_func;
      cu_err_chk( cuModuleGetFunction( &cu_func, *cu_mod, info.func_name.c_str() ), "cuModuleGetFunction" );
      // FIXME: i'd like to play with enabling L1 caching for these kernels, but it's not clear how to do that
      // cu_err_chk( cuFuncSetCacheConfig( cu_func, CU_FUNC_CACHE_PREFER_L1 ), "cuFuncSetCacheConfig" ); // does nothing?
      if( show_func_attrs ) {
	string rfas = cu_get_all_func_attrs( cu_func );
	printf( "%s: \n%s", info.func_name.c_str(), str(rfas).c_str() );
      }
      must_insert( *cu_funcs, info.func_name, nv_func_info_t{ info, cu_func, cu_mod } );
    }

    CUdeviceptr null_cup; // inited to 0; used to pass null device pointers to kernels. note, however, that the value is
			  // generally unused, so the value doesn't really matter currently. it might later of course.
    p_map_str_var_info_t vis;
    p_map_str_nv_func_info_t cu_funcs;
    virtual void release_all_funcs( void ) { 
      // FIXME/NOTE: it's unclear if syncs are needed here, but we added them when debugging cuda-memcheck +
      // cumoduleunload. there's no evidence they do anything, but i guess they shouldn't hurt for now.
      finish_and_sync();
      cu_funcs->clear();
      finish_and_sync(); 
    }

    vect_call_ev_t call_evs;
    call_ev_t & get_call_ev( uint32_t const & call_id ) { assert_st( call_id < call_evs.size() ); return call_evs[call_id]; }
    uint32_t alloc_call_id( void ) { call_evs.push_back( call_ev_t() ); return call_evs.size() - 1; }
    virtual void release_per_call_id_data( void ) { call_evs.clear(); } // invalidates all call_ids inside rtc_func_call_t's

    virtual float get_dur( uint32_t const & b, uint32_t const & e ) {
      float compute_dur = 0.0f;
      cu_err_chk( cuEventElapsedTime( &compute_dur, *get_call_ev(b).b_ev, *get_call_ev(e).e_ev ), "cuEventElapsedTime" );
      return compute_dur;
    }

    void copy_nda_to_var( string const & vn, p_nda_t const & nda ) {
      var_info_t const & vi = must_find( *vis, vn );
      assert_st( vi.dims == nda->dims );
      assert_st( vi.cup->sz == nda->dims.bytes_sz() );
      cu_err_chk( cuMemcpyHtoDAsync( vi.cup->p, nda->rp_elems(), vi.cup->sz, 0 ), "cuMemcpyHtoD" );
      //record_event( vi.ev );
    }
    void copy_var_to_nda( p_nda_t const & nda, string const & vn ) {
      var_info_t const & vi = must_find( *vis, vn );
      assert_st( vi.dims == nda->dims );
      assert_st( vi.cup->sz == nda->dims.bytes_sz() );
      cu_err_chk( cuMemcpyDtoH( nda->rp_elems(), vi.cup->p, vi.cup->sz ), "cuMemcpyDtoH" );
    }
    void create_var_with_dims( string const & vn, dims_t const & dims ) { must_insert( *vis, vn, var_info_t( dims ) ); }
    void create_var_with_dims_as_reshaped_view_of_var( string const & vn, dims_t const & dims, string const & src_vn ) {
      var_info_t const & src_vi = must_find( *vis, src_vn );
      rtc_reshape_check( dims, src_vi.dims );
      must_insert( *vis, vn, var_info_t( src_vi, dims ) );
    }

    void release_var( string const & vn ) { must_erase( *vis, vn ); }
    dims_t get_var_dims( string const & vn ) { return must_find( *vis, vn ).dims; }
    void set_var_to_zero( string const & vn ) { must_find( *vis, vn ).cup->set_to_zero(); }
    
    nvrtc_compute_t( void ) : vis( new map_str_var_info_t ), cu_funcs( new map_str_nv_func_info_t ) { }

    void add_args( vect_string const & args, vect_rp_void & cu_func_args, p_map_str_p_nda_t const & func_args ) {
      for( vect_string::const_iterator i = args.begin(); i != args.end(); ++i ) {
        // FIXME: we really want the func arg names from the layer above this here, but we'll make do for now: 
        string const an = "arg_" + str( func_args->size() ); 
	if( *i == "<NULL>" ) { 
	  cu_func_args.push_back( &null_cup );
          must_insert( *func_args, an, make_shared<nda_t>() );
	} else {
	  var_info_t const & vi = must_find( *vis, *i );
	  cu_func_args.push_back( &vi.cup->p );
          // note that we assume here both host and device pointers are 64 bits (or at least the same size ... or maybe
          // just that the host size is >= the device size). curious about conversion between CUdeviceptr and (void *)?
          // take a look here:
          // http://www.cudahandbook.com/2013/08/why-does-cuda-cudeviceptr-use-unsigned-int-instead-of-void/
          must_insert( *func_args, an, make_shared<nda_t>(vi.dims,(void *)(uintptr_t)vi.cup->p) );
	}
      }
    }
#if 0
    void record_var_events( vect_string const & vars, rtc_func_call_t const & rfc ) {
      for( vect_string::const_iterator i = vars.begin(); i != vars.end(); ++i ) { must_find( *vis, *i ).ev = rfc.e_ev; }
    }
#endif
    bool has_func_by_name( string const & func_name ) { return has( *cu_funcs, func_name ); }

    void run( rtc_func_call_t & rfc ) {
      timer_t t("cu_launch_and_sync");
      string const & fn = rfc.rtc_func_name;
      // FIXME/NOTE: for now, for interfacing with culibs, we create an extra/redundant argument map 'func_args':
      p_map_str_p_nda_t func_args = make_shared<map_str_p_nda_t>();
      vect_rp_void cu_func_args;
      for( vect_vect_string::const_iterator i = rfc.args.begin(); i != rfc.args.end(); ++i ) {
        add_args( *i, cu_func_args, func_args );
      }
      for( vect_p_nda_t::iterator i = rfc.nda_args.begin(); i != rfc.nda_args.end(); ++i ) { 
        assert_st( (*i)->elems_sz() == 1 );
        cu_func_args.push_back( (*i)->rp_elems() );
      }
      if( rfc.has_cucl_arg_info.v ) { cu_func_args.push_back( &rfc.cucl_arg_info[0] ); }
      else { assert_st( rfc.cucl_arg_info.empty() ); }

      rfc.call_id = alloc_call_id();
      record_event( get_call_ev(rfc.call_id).b_ev );
      nv_func_info_t & nfi = must_find( *cu_funcs, fn.c_str() );
      if( startswith( fn, "cublas_" ) || startswith( fn, "cudnn_" )) { culibs_wrap_call( cw, nfi.info, func_args ); } 
      else {
        rtc_launch_check_blks_and_tpb( fn, rfc.blks.v, rfc.tpb.v );
        cu_err_chk( cuLaunchKernel( nfi.func,
				    rfc.blks.v, 1, 1, // grid x,y,z dims
				    rfc.tpb.v, 1, 1, // block x,y,z dims
				    0, 0, // smem_bytes, stream_ix
				    &cu_func_args[0], // cu_func's args
				    0 ), "cuLaunchKernel" ); // unused 'extra' arg-passing arg
      }
      record_event( get_call_ev(rfc.call_id).e_ev );
      //record_var_events( rfc.inout_args, rfc );
      //record_var_events( rfc.out_args, rfc );
    }

    void finish_and_sync( void ) { cu_err_chk( cuCtxSynchronize(), "cuCtxSynchronize" ); }

    void profile_start( void ) { cuProfilerStart(); }
    void profile_stop( void ) { cuProfilerStop(); }

  protected:
    void record_event( p_void const & ev ) { cu_err_chk( cuEventRecord( *(CUevent*)ev.get(), 0 ), "cuEventRecord" ); }

  };
  struct nvrtc_compute_t; typedef shared_ptr< nvrtc_compute_t > p_nvrtc_compute_t; 

#include"gen/nvrtc_util.cc.nesi_gen.cc"
}
