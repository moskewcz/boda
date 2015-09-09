// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"nvrtc_util.H"
#include"str_util.H"

// FIXME: try to remove after nvrtc_test is removed.
#include"rand_util.H"
#include"has_main.H"

#include"has_conv_fwd.H"
#include"timers.H"
#include<nvrtc.h>
#include<cuda.h>
#include<cudaProfiler.h>

#include"rtc_compute.H"

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

  typedef map< string, p_cup_float > map_str_p_cup_float_t;
  typedef shared_ptr< map_str_p_cup_float_t > p_map_str_p_cup_float_t;


  typedef shared_ptr< CUevent > p_CUevent; 
  void cuEventDestroy_wrap( CUevent const * const p ) { 
    if(!p){return;} 
    cu_err_chk( cuEventDestroy( *p ), "cuEventDestroy" ); 
  }
  p_CUevent make_p_CUevent( void ) {
    CUevent ret;
    cu_err_chk( cuEventCreate( &ret, CU_EVENT_DEFAULT ), "cuEventCreate" );
    return p_CUevent( new CUevent( ret ), cuEventDestroy_wrap ); 
  }

  typedef map< string, CUfunction > map_str_CUfunction_t;
  typedef shared_ptr< map_str_CUfunction_t > p_map_str_CUfunction_t;


  struct nvrtc_compute_t : virtual public nesi, public rtc_compute_t // NESI(help="libnvrtc based rtc support (i.e. CUDA)",
			   // bases=["rtc_compute_t"], type_id="nvrtc" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    // FIXME: can/should we init these cu_* vars?
    CUdevice cu_dev;
    CUcontext cu_context;
    zi_bool init_done;
    void init( void ) {
      assert_st( !init_done.v );

      cu_err_chk( cuInit( 0 ), "cuInit" ); 
      cu_err_chk( cuDeviceGet( &cu_dev, 0 ), "cuDeviceGet" );
      //cu_err_chk( cuCtxCreate( &cu_context, 0, cu_dev ), "cuCtxCreate" );
      cu_err_chk( cuDevicePrimaryCtxRetain( &cu_context, cu_dev ), "cuDevicePrimaryCtxRetain" );
      cu_err_chk( cuCtxSetCurrent( cu_context ), "cuCtxSetCurrent" ); // is this always needed/okay?
      // cu_err_chk( cuCtxSetCacheConfig( CU_FUNC_CACHE_PREFER_L1 ), "cuCtxSetCacheConfig" ); // does nothing?

      init_done.v = 1;
    }

    CUmodule cu_mod;
    zi_bool mod_valid;
    void compile( string const & src, bool const show_compile_log, bool const enable_lineinfo ) {
      assert( init_done.v );
      write_whole_fn( "out.cu", src );
      string const prog_ptx = nvrtc_compile( src, show_compile_log, enable_lineinfo );
      write_whole_fn( "out.ptx", prog_ptx );
      assert( !mod_valid.v );
      cu_err_chk( cuModuleLoadDataEx( &cu_mod, prog_ptx.c_str(), 0, 0, 0 ), "cuModuleLoadDataEx" );
      mod_valid.v = 1;
    }

    p_map_str_p_cup_float_t cups;
    p_map_str_CUfunction_t cu_funcs;

    void copy_to_var( string const & vn, float const * const v ) {
      p_cup_float const & cup = must_find( *cups, vn );
      cu_err_chk( cuMemcpyHtoD( cup->p, v, cup->sz*sizeof(float) ), "cuMemcpyHtoD" );
    }
    void copy_from_var( float * const v, string const & vn ) {
      p_cup_float const & cup = must_find( *cups, vn );
      cu_err_chk( cuMemcpyDtoH( v, cup->p, cup->sz*sizeof(float) ), "cuMemcpyDtoH" );
    }
    void create_var_with_sz_floats( string const & vn, uint32_t const & sz ) { must_insert( *cups, vn, make_shared<cup_float>( sz ) ); }
    uint32_t get_var_sz( string const & vn ) { return must_find( *cups, vn )->sz; }
    void set_var_to_zero( string const & vn ) { must_find( *cups, vn )->set_to_zero(); }
    
    nvrtc_compute_t( void ) : cups( new map_str_p_cup_float_t ), cu_funcs( new map_str_CUfunction_t ) { }

    // note: post-compilation, MUST be called exactly once on all functions that will later be run()
    void check_runnable( string const name, bool const show_func_attrs ) {
      assert_st( mod_valid.v );
      CUfunction cu_func;
      cu_err_chk( cuModuleGetFunction( &cu_func, cu_mod, name.c_str() ), "cuModuleGetFunction" );
      // FIXME: i'd like to play with enabling L1 caching for these kernels, but it's not clear how to do that
      // cu_err_chk( cuFuncSetCacheConfig( cu_func, CU_FUNC_CACHE_PREFER_L1 ), "cuFuncSetCacheConfig" ); // does nothing?
      if( show_func_attrs ) {
	string rfas = cu_get_all_func_attrs( cu_func );
	printf( "%s: \n%s", name.c_str(), str(rfas).c_str() );
      }
      must_insert( *cu_funcs, name, cu_func );
    }

    p_void make_event( void ) { return make_p_CUevent(); }
    void record_event( p_void const & ev ) { cu_err_chk( cuEventRecord( *(CUevent*)ev.get(), 0 ), "cuEventRecord" ); }
    float get_event_dur( p_void const & b_ev, p_void const & e_ev ) {
      float compute_dur = 0.0f;
      cu_err_chk( cuEventElapsedTime( &compute_dur, *(CUevent*)b_ev.get(), *(CUevent*)e_ev.get() ), "cuEventElapsedTime" );
      return compute_dur;
    }

    void run( rtc_func_call_t & rfc ) {
      CUfunction & cu_func = must_find( *cu_funcs, rfc.rtc_func_name.c_str() );
      vect_rp_void cu_func_args;
      for( vect_string::const_iterator i = rfc.args.begin(); i != rfc.args.end(); ++i ) {
	p_cup_float const & cu_v = must_find( *cups, *i );
	cu_func_args.push_back( &cu_v->p );
      }
      for( vect_uint32_t::iterator i = rfc.u32_args.begin(); i != rfc.u32_args.end(); ++i ) { cu_func_args.push_back( &(*i) ); }
      if( !rfc.b_ev ) { rfc.b_ev = make_event(); } 
      if( !rfc.e_ev ) { rfc.e_ev = make_event(); }

      timer_t t("cu_launch_and_sync");
      record_event( rfc.b_ev );
      cu_err_chk( cuLaunchKernel( cu_func,
				  rfc.blks.v, 1, 1, // grid x,y,z dims
				  rfc.tpb.v, 1, 1, // block x,y,z dims
				  0, 0, // smem_bytes, stream_ix
				  &cu_func_args[0], // cu_func's args
				  0 ), "cuLaunchKernel" ); // unused 'extra' arg-passing arg
      record_event( rfc.e_ev );
    }

    void finish_and_sync( void ) { cu_err_chk( cuCtxSynchronize(), "cuCtxSynchronize" ); }

    void profile_start( void ) { cuProfilerStart(); }
    void profile_stop( void ) { cuProfilerStop(); }

  };

  struct nvrtc_compute_t; typedef shared_ptr< nvrtc_compute_t > p_nvrtc_compute_t; 

  
  struct nvrtc_test_t : virtual public nesi, public has_main_t // NESI(help="test basic usage of cuda nvrtc library",
			// bases=["has_main_t"], type_id="nvrtc_test")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t prog_fn; //NESI(default="%(boda_test_dir)/nvrtc_test_dot.cu",help="cuda program source filename")
    uint32_t data_sz; //NESI(default=10000,help="size in floats of test data")

    boost::random::mt19937 gen;

    virtual void main( nesi_init_arg_t * nia ) { 
      p_rtc_compute_t rtc( new nvrtc_compute_t );
      rtc->init();
      p_string prog_str = read_whole_fn( prog_fn );
      rtc->compile( *prog_str, 0, 0 );

      vect_float a( data_sz, 0.0f );
      rand_fill_vect( a, 2.5f, 7.5f, gen );
      vect_float b( data_sz, 0.0f );
      rand_fill_vect( b, 2.5f, 7.5f, gen );
      vect_float c( data_sz, 123.456f );

      rtc->init_var_from_vect_float( "a", a );
      rtc->init_var_from_vect_float( "b", b );
      rtc->init_var_from_vect_float( "c", c );
      
      rtc_func_call_t rfc{ "dot", {"a","b","c"}, {data_sz} }; 
      rfc.tpb.v = 256;
      rfc.blks.v = u32_ceil_div( data_sz, rfc.tpb.v );

      rtc->check_runnable( rfc.rtc_func_name, 0 );

      rtc->run( rfc );
      rtc->finish_and_sync();
      rtc->set_vect_float_from_var( c, "c" );
      assert_st( b.size() == a.size() );
      assert_st( c.size() == a.size() );
      for( uint32_t i = 0; i != c.size(); ++i ) {
	if( fabs((a[i]+b[i]) - c[i]) > 1e-6f ) {
	  printf( "bad res: i=%s a[i]=%s b[i]=%s c[i]=%s\n", str(i).c_str(), str(a[i]).c_str(), str(b[i]).c_str(), str(c[i]).c_str() );
	  break;
	}
      }
    }
  };

#include"gen/nvrtc_util.cc.nesi_gen.cc"
}
