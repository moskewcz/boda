// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"nvrtc_util.H"
#include"str_util.H"
#include"rand_util.H"
#include"has_main.H"
#include"has_conv_fwd.H"
#include"timers.H"
#include<nvrtc.h>
#include<cuda.h>
#include<boost/filesystem.hpp>
#include<boost/lexical_cast.hpp>
#include"lexp.H"
#include<cudaProfiler.h>
// for conv_pipe_fwd_t
#include"conv_util.H"

namespace boda 
{
using boost::filesystem::path;

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
  string nvrtc_compile( string const & cuda_prog_str, bool const & print_log ) {
    timer_t t("nvrtc_compile");
    p_nvrtcProgram cuda_prog = make_p_nvrtcProgram( cuda_prog_str );
    vect_string cc_opts = {"--use_fast_math","--gpu-architecture=compute_52","--restrict","-lineinfo"};
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
    cup_T( uint32_t const sz_ ) : p(0), sz(sz_) { cu_err_chk( cuMemAlloc( &p, sz * sizeof(element_type) ), "cuMemAlloc" ); }
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
  
  struct nvrtc_test_t : virtual public nesi, public has_main_t // NESI(help="test basic usage of cuda nvrtc library",
			// bases=["has_main_t"], type_id="nvrtc_test")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t prog_fn; //NESI(default="%(boda_test_dir)/nvrtc_test_dot.cu",help="cuda program source filename")
    uint32_t data_sz; //NESI(default=10000,help="size in floats of test data")

    boost::random::mt19937 gen;

    virtual void main( nesi_init_arg_t * nia ) { 
      p_string prog_str = read_whole_fn( prog_fn );
      string const prog_ptx = nvrtc_compile( *prog_str, 0 );

      cu_err_chk( cuInit( 0 ), "cuInit" );
      CUdevice cu_dev;
      cu_err_chk( cuDeviceGet( &cu_dev, 0 ), "cuDeviceGet" );
      CUcontext cu_context;
      cu_err_chk( cuCtxCreate( &cu_context, 0, cu_dev ), "cuCtxCreate" );
      CUmodule cu_mod;
      cu_err_chk( cuModuleLoadDataEx( &cu_mod, &prog_ptx[0], 0, 0, 0 ), "cuModuleLoadDataEx" );
      CUfunction cu_func;
      cu_err_chk( cuModuleGetFunction( &cu_func, cu_mod, "dot" ), "cuModuleGetFunction" );

      vect_float a( data_sz, 0.0f );
      rand_fill_vect( a, 2.5f, 7.5f, gen );
      vect_float b( data_sz, 0.0f );
      rand_fill_vect( b, 2.5f, 7.5f, gen );
      vect_float c( data_sz, 123.456f );

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

  typedef map< string, uint32_t > map_str_u32_t;
  typedef map< string, float > map_str_float_t;

  typedef map< string, p_cup_float > map_str_p_cup_float_t;
  typedef shared_ptr< map_str_p_cup_float_t > p_map_str_p_cup_float_t;

  void copy_named_ndas_to_cups( vect_string const & names, map_str_p_nda_float_t const & ndas, map_str_p_cup_float_t const & cups ) {
    for( vect_string::const_iterator i = names.begin(); i != names.end(); ++i ) {
      string const pyid = as_pyid( *i );
      cu_copy_nda_to_cup( must_find( cups, pyid ), must_find( ndas, pyid ) );
    }
  }
  void copy_named_cups_to_ndas( vect_string const & names, map_str_p_cup_float_t const & cups, map_str_p_nda_float_t & ndas ) {
    for( vect_string::const_iterator i = names.begin(); i != names.end(); ++i ) {
      string const pyid = as_pyid( *i );
      cu_copy_cup_to_nda( must_find( ndas, pyid ), must_find( cups, pyid ) );
    }
  }

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

  struct cu_func_call_t { 
    string cu_func_name; 
    vect_string args; 
    vect_uint32_t u32_args;
    string call_tag;
    // begin and end event from most recent call (currently only for timing/profiling)
    p_CUevent b_ev; 
    p_CUevent e_ev;
    void ensure_evs( void ) { if( !b_ev ) { b_ev = make_p_CUevent(); } if( !e_ev ) { e_ev = make_p_CUevent(); } }
  };
  typedef vector< cu_func_call_t > vect_cu_func_call_t; 
  struct gen_layout_info_t { uint32_t in_chans; uint32_t bix_out_chan_blk_sz; uint32_t tix_out_chan_tile_sz; };
  struct cu_func_t { 
    string name;
    bool finalized;
    vect_uint32_t arg_sizes;
    uint32_t tpb;
    uint32_t blks;
    CUfunction cu_func; 
    gen_layout_info_t gli; // for communication between conv codegen and xpose codegen
  };
  typedef map< string, cu_func_t > cu_funcs_t;

  struct quantize_ops_t : virtual public nesi // NESI(help="per-layer quantization options") 
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string name; //NESI(help="name of node to apply operation to",req=1)
    uint32_t max_val; //NESI(help="clamp value to this maximum",req=1)
    uint32_t keep_bits; //NESI(help="after clamping, keep this many high bits",req=1)
  };
  typedef vector< quantize_ops_t > vect_quantize_ops_t; 
  typedef shared_ptr< quantize_ops_t > p_quantize_ops_t; 
  typedef vector< p_quantize_ops_t > vect_p_quantize_ops_t;

  struct conv_pipe_fwd_t : virtual public nesi, public has_conv_fwd_t // NESI(help="compute conv pipe forward using rtc",
			   // bases=["has_conv_fwd_t"], type_id="nvrtc" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    uint32_t enable_stats; //NESI(default=0,help="if 1, dump stats")
    uint32_t enable_prof; //NESI(default=1,help="if 1, enable profiling")
    string per_call_fn; //NESI(default="",help="if non-empty, write per-call profiling (timing via events) to given file.")
    vect_string def; // NESI(help="#define STR 1 in generated code")
    vect_p_quantize_ops_t quantize; //NESI(help="per-layer quantize options")
    uint32_t quantize_keep_bits; //NESI(default=8,help="number of bits to keep when quantizing")
    uint32_t show_compile_log; //NESI(default=0,help="if 1, print compilation log")
    uint32_t show_rtc_calls; //NESI(default=0,help="if 1, print rtc calls")
    uint32_t show_func_attrs; //NESI(default=0,help="if 1, print func attrs after load")
    uint32_t enable_s1conv; //NESI(default=0,help="if 1, enable experimental s1conv special case")
    uint32_t t_tile_sz; //NESI(default=8,help="register blocking tile size: compute t_tile_sz^2 outputs in registers per thread")

    p_conv_pipe_t cp;
    uint32_t num_imgs;
    p_map_str_p_cup_float_t cups;
    vect_string op_param_names;
    set_string filts_names;

    vect_string stats_names;
    map_str_float_t stats_map;

    //nvrtc/cuda state
    CUdevice cu_dev;
    CUcontext cu_context;
    CUmodule cu_mod;
    CUfunction cu_func;

    string cu_prog_str;
    vect_cu_func_call_t init_calls;
    vect_cu_func_call_t fwd_calls;
    cu_funcs_t cu_funcs;
    

    virtual void init( p_conv_pipe_t const & cp_, uint32_t const & num_imgs_ );
    virtual void run_fwd( p_map_str_p_nda_float_t const & fwd );

    void update_stats( void );
    virtual ~conv_pipe_fwd_t( void );
  protected:
    cu_func_t & gen_op_kern( p_conv_op_t const & cop, conv_io_t const & cio_in, p_conv_node_t const & node_out );
    cu_func_t & gen_op_s1conv( bool const conv_has_relu, uint32_t const & in_pad, uint32_t const kern_sz,
			       conv_io_t const & cio_in, conv_io_t const & cio_out ); // stride 1, kern_sz <= 5 special case
    cu_func_t & gen_op_lrn( p_conv_op_t const & cop, conv_io_t const & cio_in, conv_io_t const & cio_out );
    cu_func_t & gen_op_copy( p_conv_op_t const & cop, conv_io_t const & cio_in, conv_io_t const & cio_out, uint32_t const ocix );
    cu_func_t & gen_op_relu( conv_io_t const & cio_out );
    cu_func_t & gen_op_xpose( p_conv_op_t const & cop, gen_layout_info_t const & gli );
    void gen_xpose( cu_func_t const & cf, p_conv_op_t const & cop, string const & filts_name, gen_layout_info_t const & gli );
    vect_string gen_op_stats( conv_io_t const & cio_in, string const & top_in );
    void gen_op_quantize( conv_io_t const & cio_in, string const & top_in, uint32_t const & max_val, uint32_t const & keep_bits );

    void gen_node( string const & name, p_conv_node_t const & node );
    void add_op_param( string const & name, uint32_t const & sz );
    void gen_op( p_conv_op_t const & cop );
    void gen_ops_rec( string const & node_name );

    void run_cfc( cu_func_call_t & cfc );
  };

  // FIXME: i'm not too happy about the duplication between here and the kernel version
  float stats_reduce( string const & stats_name, float const & v1, float const & v2 ) { 
    if( 0 ) { }
    else if( endswith(stats_name,"min_out_sz_1") ) { return std::min(v1,v2); }
    else if( endswith(stats_name,"max_out_sz_1") ) { return std::max(v1,v2); }
    else if( endswith(stats_name,"sum_out_sz_1") ) { return v1 + v2; }
    else if( endswith(stats_name,"hist_out_sz_1") ) { return v1 + v2; }
    else if( endswith(stats_name,"cnt_out_sz_1") ) { return v1 + v2; }
    else { assert_st(0); }
  }

  void conv_pipe_fwd_t::update_stats( void ) {
    for( vect_string::const_iterator i = stats_names.begin(); i != stats_names.end(); ++i ) {
      string const pyid = as_pyid( *i );
      p_cup_float const & cup = must_find( *cups, pyid );
      dims_t cup_dims( vect_uint32_t{cup->sz} ); 
      cup_dims.calc_strides();
      p_nda_float_t nda = make_shared<nda_float_t>( cup_dims );
      cu_copy_cup_to_nda( nda, cup );
      assert_st( nda->elems.sz == 1 );
      float v = nda->elems[0];
      if( has( stats_map, *i ) ) { v = stats_reduce( *i, v, stats_map[*i] ); }
      stats_map[*i] = v;
    }
  }

  conv_pipe_fwd_t::~conv_pipe_fwd_t( void ) {
    for( map_str_float_t::const_iterator i = stats_map.begin(); i != stats_map.end(); ++i ) {
      printf( "%s=%s\n", str(i->first).c_str(), str(i->second).c_str() );
    }
  }

  void conv_pipe_fwd_t::add_op_param( string const & name, uint32_t const & sz ) {
    string const & name_id = as_pyid( name );
    must_insert( *cups, name_id, make_shared<cup_float>( sz ) ); 
    op_param_names.push_back( name );
  }
  
  void insert_nda_exprs( vect_pair_str_str & mss, string const & ix, vect_string const & dns, vect_uint32_t const & dss,
			 bool const src_is_expr = 0 ) {
    assert_st( dns.size() );
    assert_st( dns.size() == dss.size() );
    string eix = ix;
    if( src_is_expr ) { eix = "%("+eix+")"; }
    uint32_t stride = 1;
    for( int32_t i = dns.size()-1; i >= 0; --i ) {
      mss.push_back( make_pair( ix+"_"+dns[i]+"_dim", str(dss[i]) ) );
      assert_st( stride );
      mss.push_back( make_pair( ix+"_"+dns[i]+"_sz", str(stride) ) );
      string v = (stride > 1) ? "("+eix+"/"+str(stride)+")" : eix;
      mss.push_back( make_pair( ix+"_"+dns[i]+"_nomod", v ) );      
      if( i ) { v = "("+v+"%%"+str(dss[i])+")"; }
      mss.push_back( make_pair( ix+"_"+dns[i], v ) );
      stride *= dss[i];
    }
    mss.push_back( make_pair( ix+"_sz", str(stride) ) );
  }

  // yeah, not the greatest ...
  uint32_t get_sz( vect_pair_str_str & mss, string const & ix ) { 
    for( vect_pair_str_str::const_iterator i = mss.begin(); i != mss.end(); ++i ) {
      if( i->first == (ix+"_sz") ) { return boost::lexical_cast< uint32_t >( i->second ); }
    }
    rt_err( "size not found in tf_exprs for ix:" + ix );
  }


  struct rtc_func_param_info_t { string name; string val; };
  struct rtc_u32_param_info_t { string name; uint32_t val; };
  typedef vector< rtc_func_param_info_t > vect_rtc_func_param_info_t; 
  struct rtc_func_gen_info_t {
    string op_tag;
    vect_rtc_func_param_info_t spec_params;
    // vect_rtc_func_param_info_t pass_params; // TODO
    cu_func_t & init( cu_funcs_t & cu_funcs ) {
      rtc_func_name = op_tag;
      for( vect_rtc_func_param_info_t::const_iterator i = spec_params.begin(); i != spec_params.end(); ++i ) {
	rtc_func_name += "__"+i->name+"_"+as_pyid(i->val);
	tf_exprs.push_back( make_pair( i->name, i->val ) );
      }
      tf_exprs.push_back( make_pair( "cu_func_name", rtc_func_name ) );
      rtc_func_template = read_whole_fn( (path(py_boda_test_dir()) / "rtc" / (op_tag+".cu")).string() );
      cf = &cu_funcs.insert( make_pair( rtc_func_name, cu_func_t{rtc_func_name,0} ) ).first->second;
      //printf( "cf->name=%s\n", str(cf->name).c_str() );
      return *cf;
    }
    vect_pair_str_str tf_exprs;
    cu_func_t *cf;
    string rtc_func_name;
    p_string rtc_func_template;
    void instantiate_template( string & cu_prog_str ) {
      lexp_name_val_map_t tf_nvm{ p_lexp_t() };
      tf_nvm.insert_leafs_from( tf_exprs );
      string rtc_func_str;
      str_format_from_nvm( rtc_func_str, *rtc_func_template, tf_nvm );
      cu_prog_str += rtc_func_str;
      cu_prog_str += "// -- template substituion table used: --\n";
      for( vect_pair_str_str::const_iterator i = tf_exprs.begin(); i != tf_exprs.end(); ++i ) {
	cu_prog_str += strprintf( "/* %s = %s */\n", str(i->first).c_str(), str(i->second).c_str() );
      }
      //printf( "rtc_func_name=%s cf.tpb=%s cf.blks=%s\n", str(rtc_func_name).c_str(), str(cf->tpb).c_str(), str(cf->blks).c_str()); 
      cf->finalized = 1;
    }
  // note: also adds the output as a parameter
    void set_tpb_blks_for_one_output_per_thread( uint32_t out_sz ) {
      // note: cf.arg_sizes might or might not be empty here
      cf->arg_sizes.push_back( out_sz );
      cf->tpb = 256;
      cf->blks = u32_ceil_div( out_sz, cf->tpb );
    }
  };

  cu_func_t & conv_pipe_fwd_t::gen_op_kern( p_conv_op_t const & cop, conv_io_t const & cio_in, p_conv_node_t const & node_out ) {
    bool const is_conv = cop->type == Convolution_str;
    bool const is_pool = cop->type == Pooling_str;
    // if the output node's first in_place op is a ReLU, fuse it into this conv. a matching conditional later will omit the relu
    bool const conv_has_relu = is_conv && (node_out->in_place_ops.size() > 0) && (node_out->in_place_ops[0]->type == ReLU_str);
    // note: cio_in and node_out are derived from cop->bots[0] and cop->tops[0]
    // for now, we only attempt to handle the (common) case of uniform padding, kernel size, and stride
    conv_io_t & cio_out = node_out->cio;
    assert_st( cop->in_pad.bnds_are_same() );
    assert_st( cop->in_pad.p[0].dims_are_same() );
    assert_st( cop->stride.dims_are_same() );
    u32_pt_t kern_sz_ = cop->kern_sz;
    if( kern_sz_.is_zeros() ) { kern_sz_ = cio_in.sz; } // 'global' input special case
    assert_st( kern_sz_.dims_are_same() );
    uint32_t const in_pad = cop->in_pad.p[0].d[0];
    uint32_t const kern_sz = kern_sz_.d[0];
    uint32_t const stride = cop->stride.d[0];
    
    // also, for now, we'll only handle square inputs. however, this is probably too limiting for more than initial tests.
    assert_st( cio_in.sz.dims_are_same() );
    if( is_conv && (stride == 1) && (kern_sz <= 5) && (kern_sz > 1) && (cio_in.sz.d[0] >= 20) && (cio_in.sz.d[0] <= 500) ) { 
      //return gen_op_s1conv( conv_has_relu, in_pad, kern_sz, cio_in, cio_out ); 
    }

    rtc_func_gen_info_t rfgi{"",
      { {"num_imgs",str(num_imgs)},{"in_pad",str(in_pad)},{"in_dim_0",str(cio_in.sz.d[0])},{"in_dim_1",str(cio_in.sz.d[1])}
	,{"conv_has_relu",str(conv_has_relu)},{"kern_sz",str(kern_sz)},{"stride",str(stride)},{"out_chans",str(cio_out.chans)} } };
    if( 0 ) { }
    else if( is_conv ) { rfgi.op_tag="conv"; rfgi.spec_params.push_back( rtc_func_param_info_t{"in_chans",str(cio_in.chans)} ); }
    else if( is_pool ) { rfgi.op_tag="pool"; rfgi.spec_params.push_back( rtc_func_param_info_t{"avg_pool",str(cop->avg_pool)} ); }
    else { rt_err( "unhanded kern op: " + cop->type ); }    
    cu_func_t & cf = rfgi.init( cu_funcs );
    vect_pair_str_str & tf_exprs = rfgi.tf_exprs;
    if( cf.finalized ) { return cf; } // already generated

    tf_exprs.push_back( make_pair( "t_tile_sz", str(t_tile_sz) ) );

    vect_string const cio_dims{"img","chan","y","x"};
    insert_nda_exprs( tf_exprs, "out_ix", cio_dims, vect_uint32_t{num_imgs,cio_out.chans,cio_out.sz.d[1],cio_out.sz.d[0]} );
    uint32_t const out_ix_sz = get_sz( tf_exprs, "out_ix" );
    insert_nda_exprs( tf_exprs, "in_ix", cio_dims, vect_uint32_t{num_imgs,cio_in.chans,cio_in.sz.d[1],cio_in.sz.d[0]} );
    if( is_conv ) {
      // for reg blocking
      uint32_t const out_chan_tile_sz = u32_ceil_div( cio_out.chans, t_tile_sz );
      //insert_nda_exprs( tf_exprs, "filts_xp_ix", vect_string{"in_chan","y","x","out_chan"}, 
      //		vect_uint32_t{cio_in.chans,kern_sz,kern_sz,cio_out.chans} );
      //assert_st( out_chan_tile_sz * t_tile_sz == cio_out.chans ); // FIXME: too strong (need to handle partial tiles)
      uint32_t const patch_sz = u32_ceil_div( out_ix_sz, cio_out.chans );
      assert_st( patch_sz * cio_out.chans == out_ix_sz ); // by construction
      uint32_t const patch_tile_sz = u32_ceil_div( patch_sz, t_tile_sz );
      //insert_nda_exprs( tf_exprs, "tile_ix", vect_string{"patch_tile","out_chan_tile"}, vect_uint32_t{patch_tile_sz,out_chan_tile_sz} );

      insert_nda_exprs( tf_exprs, "t_smem_patch_ix", vect_string{"img","y","x"}, vect_uint32_t{num_imgs,cio_out.sz.d[1],cio_out.sz.d[0]} );
      insert_nda_exprs( tf_exprs, "filts_ix_out_chan_elem", vect_string{"in_chan","y","x"}, 
			vect_uint32_t{cio_in.chans,kern_sz,kern_sz} );
      //printf( "out_chan_tile_sz=%s patch_tile_sz=%s\n", str(out_chan_tile_sz).c_str(), str(patch_tile_sz).c_str() );
      uint32_t const goal_tix_out_chan_tile_sz = 16; // sqrt( cf.tpb ) above, more or less, but tweakable
      // determine block geometry in terms of WxH where the W is over out_chan_tile_sz (typ. ~64-1024+ / 8) and the H is
      // over patch_size (probably large-ish, at least in the cases we care most about perf for). ideally, we want
      // blocks with size sqrt(tpb) tiles. but, we can't (usefully) use a W smaller than the out_chans.
      uint32_t tix_out_chan_tile_sz = std::min( goal_tix_out_chan_tile_sz, out_chan_tile_sz );
      if( tix_out_chan_tile_sz < goal_tix_out_chan_tile_sz ) {
	
      }
      uint32_t tix_patch_tile_sz = 8; // treated as a minimum
      cf.tpb = 128; // treated as a target, but not be exceeded
      while( (tix_patch_tile_sz+1) * tix_out_chan_tile_sz < cf.tpb ) { ++tix_patch_tile_sz; }
      uint32_t const new_tbp = tix_patch_tile_sz * tix_out_chan_tile_sz;// recalculate tpb, should not increase
      assert_st( new_tbp <= cf.tpb );
      cf.tpb = new_tbp;
      //printf( "tix_patch_tile_sz=%s tix_out_chan_tile_sz=%s cf.tpb=%s\n", str(tix_patch_tile_sz).c_str(), str(tix_out_chan_tile_sz).c_str(), str(cf.tpb).c_str() );
      insert_nda_exprs( tf_exprs, "threadIdx.x", vect_string{"patch_tile","out_chan_tile"}, 
			vect_uint32_t{tix_patch_tile_sz,tix_out_chan_tile_sz} );

      uint32_t const bix_out_chan_blk_sz = u32_ceil_div( out_chan_tile_sz, tix_out_chan_tile_sz );
      
      // fill in gli. this indo is used by gen_xpos to generate the xpose op for filters for used by conv
      cf.gli.in_chans = cio_in.chans; 
      cf.gli.tix_out_chan_tile_sz = tix_out_chan_tile_sz; // num out chan tiles (threads) per block (~8-16)
      cf.gli.bix_out_chan_blk_sz = bix_out_chan_blk_sz; // number of blocks in out_chan dim of blocks  

      insert_nda_exprs( tf_exprs, "filts_xp_ix", vect_string{"in_chan","y","x","out_chan_blk","out_chan_reg","out_chan_tile"}, 
			vect_uint32_t{cio_in.chans,kern_sz,kern_sz,bix_out_chan_blk_sz,t_tile_sz,tix_out_chan_tile_sz} );

      // check that we have enough threads per block to load smem using one-elem-per-thread.
      // FIXME: allow for cases when this does not hold
      assert_st( cf.tpb >= (t_tile_sz * tix_out_chan_tile_sz) ); 
      uint32_t const patch_smem_load_iter = u32_ceil_div( (t_tile_sz * tix_patch_tile_sz), cf.tpb );
      tf_exprs.push_back( std::make_pair( "patch_smem_load_iter", str(patch_smem_load_iter) ) );
      // printf( "patch_smem_load_iter=%s\n", str(patch_smem_load_iter).c_str() );
      // assert_st( cf.tpb*2 >= (t_tile_sz * tix_patch_tile_sz) ); // fixed load loop of size 2
      
      uint32_t const bix_patch_blk_sz = u32_ceil_div( patch_tile_sz, tix_patch_tile_sz );
      cf.blks = bix_patch_blk_sz * bix_out_chan_blk_sz; 
      insert_nda_exprs( tf_exprs, "blockIdx.x", vect_string{"patch_blk","out_chan_blk"}, vect_uint32_t{bix_patch_blk_sz,bix_out_chan_blk_sz}); 

      tf_exprs.push_back( std::make_pair( "out_chan_tile", 
					  "(%(threadIdx.x_out_chan_tile)+%(blockIdx.x_out_chan_blk)*%(threadIdx.x_out_chan_tile_dim))"));
      tf_exprs.push_back( std::make_pair( "patch_tile",
					  "(%(threadIdx.x_patch_tile)+%(blockIdx.x_patch_blk)*%(threadIdx.x_patch_tile_dim))"));

      tf_exprs.push_back( std::make_pair( "out_chan_ix","(%(out_chan_tile)*%(t_tile_sz))" ) );
      
      for( uint32_t i = 0; i != t_tile_sz; ++i ) {
	tf_exprs.push_back( std::make_pair( "patch_ix_" + str(i), 
					    strprintf( "(%%(patch_tile)*%%(t_tile_sz)+%s)", str(i).c_str() ) ) );
	insert_nda_exprs( tf_exprs, "patch_ix_" + str(i), 
			  vect_string{"img","y","x"}, vect_uint32_t{num_imgs,cio_out.sz.d[1],cio_out.sz.d[0]},
			  1 );
      }
#if 1
      string const get_in = strprintf( 
	"float v = 0;\n"
        "      int const smem_in_ix_y = %%(t_smem_patch_ix_y)*%%(stride)+%%(filts_ix_out_chan_elem_y) - %%(in_pad);\n"
        "      int const smem_in_ix_x = %%(t_smem_patch_ix_x)*%%(stride)+%%(filts_ix_out_chan_elem_x) - %%(in_pad);\n"
        "      if(smem_in_ix_y >= 0 && smem_in_ix_x >= 0 && \n"
        "         smem_in_ix_x < %%(in_ix_x_dim) && smem_in_ix_y < %%(in_ix_y_dim) ) {\n"
        "        v = in[%%(t_smem_patch_ix_img)*%%(in_ix_img_sz) +\n"
	"          %%(filts_ix_out_chan_elem_in_chan)*%%(in_ix_chan_sz) +\n"
	"          smem_in_ix_y*%%(in_ix_y_sz) +\n"
	"          smem_in_ix_x*%%(in_ix_x_sz)];\n" 
	"      }"
				       );
#else // hack for testing overhead of above
      string const get_in = strprintf("float v = in[threadIdx.x];\n");
#endif				      
      tf_exprs.push_back( std::make_pair( "get_in", get_in ) );
			
    } else if( is_pool ) { 
      cf.tpb = 256;
      cf.blks = u32_ceil_div( out_ix_sz, cf.tpb ); 
    }
    else { assert_st( 0 ); }


    if( is_pool ) {
      tf_exprs.push_back( std::make_pair( "op", cop->avg_pool ? "out_v += v" : "out_v = max( out_v, v )" ) );
      tf_exprs.push_back( std::make_pair( "op_post", cop->avg_pool ? "out_v /= float("+str(kern_sz*kern_sz)+")" : "" ) );
    }
    string t_tile_fmas("// begin t_tile_fmas\n");
    string t_tile_smem_loads("// begin t_tile_smem_loads\n");
    string t_tile_loads("// begin t_tile_loads\n");
    string t_tile_dummy_loads("// begin t_tile_dummy_loads\n");
    string t_tile_stores("// begin t_tile_stores\n");
    string t_tile_dummy_stores("// begin t_tile_dummy_stores\n");
    if( is_conv ) {
      for( uint32_t tx = 0; tx != t_tile_sz; ++tx ) {
	t_tile_dummy_loads += strprintf( "    filts_strip[%s] = filts_smem[(threadIdx.x %%%% 32) + %s];\n", str(tx).c_str(), str(tx).c_str() );
	t_tile_loads += strprintf( "    filts_strip[%s] = filts_smem[%%(threadIdx.x_out_chan_tile)+%s*%%(threadIdx.x_out_chan_tile_dim)];\n",
					str(tx).c_str(), str(tx).c_str() );
      }
      for( uint32_t ty = 0; ty != t_tile_sz; ++ty ) { // note: could merge with above loop, but we want to use ty for consistency
	t_tile_dummy_loads += strprintf( "    in_strip[%s] = in_smem[(threadIdx.x %%%% 32) + %s];\n", str(ty).c_str(), str(ty).c_str() );
	t_tile_loads += strprintf( "    in_strip[%s] = in_smem[%%(t_tile_sz)*%%(threadIdx.x_patch_tile)+%s];\n",
				   str(ty).c_str(), str(ty).c_str() );
      }

      t_tile_stores += "  int32_t tpix[%(t_tile_sz)];\n";
      t_tile_stores += "  int32_t tcix[%(t_tile_sz)];\n";

      // FIXME: should somehow assert that both out_ix and patch_ix_N have the same dims here
      for( uint32_t ty = 0; ty != t_tile_sz; ++ty ) { 
	t_tile_stores += strprintf( "  tpix[%s] = %%(patch_ix_%s_img)*%%(out_ix_img_sz) + \n"
				    "   ( %%(patch_ix_%s) %%%% %%(patch_ix_%s_img_sz) ); // cache out patch ixs\n ",
				    str(ty).c_str(), str(ty).c_str(), str(ty).c_str(), str(ty).c_str() );
      }
      for( uint32_t ty = 0; ty != t_tile_sz; ++ty ) { 
	t_tile_stores += strprintf( "  tcix[%s] = (%%(out_chan_ix)+%s)*%%(out_ix_chan_sz); // cache out chan ixs\n",
				    str(ty).c_str(), str(ty).c_str() );
      }
	
      t_tile_dummy_stores += " out[0] = 0.0f\n";
      for( uint32_t ty = 0; ty != t_tile_sz; ++ty ) {
	t_tile_stores += "  if( %(patch_ix_"+str(ty)+") >= %(patch_ix_0_sz) ) { return; } "
	  "// this patch and the following are off-the-end patches, so don't store them.\n";
	for( uint32_t tx = 0; tx != t_tile_sz; ++tx ) {
	  t_tile_fmas += strprintf( "    out_tile[%s] += filts_strip[%s]*in_strip[%s];\n", 
				    str((ty*t_tile_sz+tx)).c_str(), str(tx).c_str(), str(ty).c_str() );
	  string const ve = strprintf( "%sout_tile[%s] + filts_strip[%s])", conv_has_relu ? "max(0.0f," : "(",
				       str((ty*t_tile_sz+tx)).c_str(), str(tx).c_str() );
	  t_tile_stores += strprintf( "if( tcix[%s] < (%%(out_ix_chan_dim)*%%(out_ix_chan_sz)) ) { "
				      "out[ tpix[%s] + tcix[%s] ] = %s; }\n",
				      str(tx).c_str(), str(ty).c_str(), str(tx).c_str(), ve.c_str() );
	  t_tile_dummy_stores += " + " + ve + "\n";
	}
      }
      t_tile_dummy_stores += ";\n";
    } 

    if( is_pool ) { } // unused for pooling
    // note: newline (and semi-unwanted semi-colon) from src will go after blocks, hence no newline on these lines
    t_tile_fmas += "    // end t_tile_fmas"; 
    t_tile_loads += "    // end t_tile_loads";
    t_tile_dummy_loads += "    // end t_tile_dummy_loads";
    t_tile_stores += "  // end t_tile_stores";
    tf_exprs.push_back( std::make_pair( "t_tile_fmas", t_tile_fmas ) );
    tf_exprs.push_back( std::make_pair( "t_tile_smem_loads", t_tile_smem_loads ) );
    tf_exprs.push_back( std::make_pair( "t_tile_loads", t_tile_loads ) );
    tf_exprs.push_back( std::make_pair( "t_tile_dummy_loads", t_tile_dummy_loads ) );
    tf_exprs.push_back( std::make_pair( "t_tile_stores", t_tile_stores ) );
    tf_exprs.push_back( std::make_pair( "t_tile_dummy_stores", t_tile_dummy_stores ) );

    // for error checking, (re-) calculate the sizes of the arguments (note: in elements, not bytes)
    if( is_conv ) { 
      cf.arg_sizes.push_back( get_sz( tf_exprs, "filts_xp_ix" ) );
      cf.arg_sizes.push_back( cio_out.chans ); // biases_sz
    }
    cf.arg_sizes.push_back( get_sz( tf_exprs, "in_ix" ) );
    cf.arg_sizes.push_back( out_ix_sz );

    rfgi.instantiate_template( cu_prog_str );
    return cf;
  }

  cu_func_t & conv_pipe_fwd_t::gen_op_s1conv( bool const conv_has_relu, uint32_t const & in_pad, uint32_t const kern_sz,
					      conv_io_t const & cio_in, conv_io_t const & cio_out ) 
  {
    rtc_func_gen_info_t rfgi{"",
      { {"num_imgs",str(num_imgs)},{"in_pad",str(in_pad)},{"in_dim_0",str(cio_in.sz.d[0])},{"in_dim_1",str(cio_in.sz.d[1])}
	,{"conv_has_relu",str(conv_has_relu)},{"kern_sz",str(kern_sz)},{"out_chans",str(cio_out.chans)} } };
    rfgi.op_tag="s1conv"; rfgi.spec_params.push_back( rtc_func_param_info_t{"in_chans",str(cio_in.chans)} );
    
    cu_func_t & cf = rfgi.init( cu_funcs );
    vect_pair_str_str & tf_exprs = rfgi.tf_exprs;
    if( cf.finalized ) { return cf; } // already generated

    tf_exprs.push_back( make_pair( "t_tile_sz", str(t_tile_sz) ) );

    vect_string const cio_dims{"img","chan","y","x"};
    insert_nda_exprs( tf_exprs, "out_ix", cio_dims, vect_uint32_t{num_imgs,cio_out.chans,cio_out.sz.d[1],cio_out.sz.d[0]} );
    uint32_t const out_ix_sz = get_sz( tf_exprs, "out_ix" );
    insert_nda_exprs( tf_exprs, "in_ix", cio_dims, vect_uint32_t{num_imgs,cio_in.chans,cio_in.sz.d[1],cio_in.sz.d[0]} );

    // for reg blocking
    uint32_t const out_chan_tile_sz = u32_ceil_div( cio_out.chans, t_tile_sz );
    uint32_t const lines_sz = u32_ceil_div( u32_ceil_div( out_ix_sz, cio_out.sz.d[0] ), cio_out.chans );
    assert_st( lines_sz * cio_out.sz.d[0] * cio_out.chans == out_ix_sz ); // by construction
    
    uint32_t const line_patch_sz = cio_out.sz.d[0];
    uint32_t const line_patch_tile_sz = u32_ceil_div( line_patch_sz, t_tile_sz );

    insert_nda_exprs( tf_exprs, "t_smem_patch_ix", vect_string{"img","y","x"}, vect_uint32_t{num_imgs,cio_out.sz.d[1],cio_out.sz.d[0]} );
    insert_nda_exprs( tf_exprs, "filts_ix_out_chan_elem", vect_string{"in_chan","y","x"}, 
		      vect_uint32_t{cio_in.chans,kern_sz,kern_sz} );
    //printf( "out_chan_tile_sz=%s patch_tile_sz=%s\n", str(out_chan_tile_sz).c_str(), str(patch_tile_sz).c_str() );
    uint32_t const goal_tix_out_chan_tile_sz = 16; // sqrt( cf.tpb ) above, more or less, but tweakable
    // determine block geometry in terms of WxH where the W is over out_chan_tile_sz (typ. ~64-1024+ / 8) and the H is
    // over patch_size (probably large-ish, at least in the cases we care most about perf for). ideally, we want
    // blocks with size sqrt(tpb) tiles. but, we can't (usefully) use a W smaller than the cio_out.chans.
    uint32_t tix_out_chan_tile_sz = std::min( goal_tix_out_chan_tile_sz, out_chan_tile_sz );
    uint32_t tix_patch_tile_sz = line_patch_tile_sz;
    cf.tpb = 512; // treated as a target, but not be exceeded
    uint32_t const new_tbp = tix_patch_tile_sz * tix_out_chan_tile_sz;// recalculate tpb, should not increase
    assert_st( new_tbp <= cf.tpb );
    cf.tpb = new_tbp;
    printf( "cio_out.sz=%s\n", str(cio_out.sz).c_str() );
    printf( "tix_patch_tile_sz=%s tix_out_chan_tile_sz=%s cf.tpb=%s\n", str(tix_patch_tile_sz).c_str(), str(tix_out_chan_tile_sz).c_str(), str(cf.tpb).c_str() );
    insert_nda_exprs( tf_exprs, "threadIdx.x", vect_string{"patch_tile","out_chan_tile"}, 
		      vect_uint32_t{tix_patch_tile_sz,tix_out_chan_tile_sz} );

    uint32_t const bix_out_chan_blk_sz = u32_ceil_div( out_chan_tile_sz, tix_out_chan_tile_sz );
      
    // fill in gli. this indo is used by gen_xpos to generate the xpose op for filters for used by conv
    cf.gli.in_chans = cio_in.chans; 
    cf.gli.tix_out_chan_tile_sz = tix_out_chan_tile_sz; // num out chan tiles (threads) per block (~8-16)
    cf.gli.bix_out_chan_blk_sz = bix_out_chan_blk_sz; // number of blocks in out_chan dim of blocks  

    insert_nda_exprs( tf_exprs, "filts_xp_ix", vect_string{"in_chan","y","x","out_chan_blk","out_chan_reg","out_chan_tile"}, 
		      vect_uint32_t{cio_in.chans,kern_sz,kern_sz,bix_out_chan_blk_sz,t_tile_sz,tix_out_chan_tile_sz} );

    uint32_t const out_chan_smem_load_iter = u32_ceil_div( (t_tile_sz * tix_out_chan_tile_sz), cf.tpb );
    tf_exprs.push_back( std::make_pair( "out_chan_smem_load_iter", str(out_chan_smem_load_iter) ) );
    uint32_t const patch_smem_load_iter = u32_ceil_div( (t_tile_sz * tix_patch_tile_sz), cf.tpb );
    tf_exprs.push_back( std::make_pair( "patch_smem_load_iter", str(patch_smem_load_iter) ) );
      
    uint32_t const bix_patch_blk_sz = lines_sz;
    cf.blks = bix_patch_blk_sz * bix_out_chan_blk_sz; 
    // TODO/FIXME: rework following for block-per-line
    insert_nda_exprs( tf_exprs, "blockIdx.x", vect_string{"patch_blk","out_chan_blk"}, vect_uint32_t{bix_patch_blk_sz,bix_out_chan_blk_sz}); 

    tf_exprs.push_back( std::make_pair( "out_chan_tile", 
					"(%(threadIdx.x_out_chan_tile)+%(blockIdx.x_out_chan_blk)*%(threadIdx.x_out_chan_tile_dim))"));
    tf_exprs.push_back( std::make_pair( "patch_tile",
					"(%(threadIdx.x_patch_tile)+%(blockIdx.x_patch_blk)*%(threadIdx.x_patch_tile_dim))"));

    tf_exprs.push_back( std::make_pair( "out_chan_ix","(%(out_chan_tile)*%(t_tile_sz))" ) );
      
    for( uint32_t i = 0; i != t_tile_sz; ++i ) {
      tf_exprs.push_back( std::make_pair( "patch_ix_" + str(i), 
					  strprintf( "(%%(patch_tile)*%%(t_tile_sz)+%s)", str(i).c_str() ) ) );
      insert_nda_exprs( tf_exprs, "patch_ix_" + str(i), 
			vect_string{"img","y","x"}, vect_uint32_t{num_imgs,cio_out.sz.d[1],cio_out.sz.d[0]},
			1 );
    }
#if 1
    string const get_in = strprintf( 
      "float v = 0;\n"
      "      int const smem_in_ix_y = %%(t_smem_patch_ix_y)+%%(filts_ix_out_chan_elem_y) - %%(in_pad);\n"
      "      int const smem_in_ix_x = %%(t_smem_patch_ix_x)+%%(filts_ix_out_chan_elem_x) - %%(in_pad);\n"
      "      if(smem_in_ix_y >= 0 && smem_in_ix_x >= 0 && \n"
      "         smem_in_ix_x < %%(in_ix_x_dim) && smem_in_ix_y < %%(in_ix_y_dim) ) {\n"
      "        v = in[%%(t_smem_patch_ix_img)*%%(in_ix_img_sz) +\n"
      "          %%(filts_ix_out_chan_elem_in_chan)*%%(in_ix_chan_sz) +\n"
      "          smem_in_ix_y*%%(in_ix_y_sz) +\n"
      "          smem_in_ix_x*%%(in_ix_x_sz)];\n" 
      "      }"
				     );
#else // hack for testing overhead of above
    string const get_in = strprintf("float v = in[threadIdx.x];\n");
#endif				      
    tf_exprs.push_back( std::make_pair( "get_in", get_in ) );
			
    string t_tile_fmas("// begin t_tile_fmas\n");
    string t_tile_smem_loads("// begin t_tile_smem_loads\n");
    string t_tile_loads("// begin t_tile_loads\n");
    string t_tile_dummy_loads("// begin t_tile_dummy_loads\n");
    string t_tile_stores("// begin t_tile_stores\n");
    string t_tile_dummy_stores("// begin t_tile_dummy_stores\n");

    for( uint32_t tx = 0; tx != t_tile_sz; ++tx ) {
      t_tile_dummy_loads += strprintf( "    filts_strip[%s] = filts_smem[(threadIdx.x %%%% 32) + %s];\n", str(tx).c_str(), str(tx).c_str() );
      t_tile_loads += strprintf( "    filts_strip[%s] = filts_smem[%%(threadIdx.x_out_chan_tile)+%s*%%(threadIdx.x_out_chan_tile_dim)];\n",
				 str(tx).c_str(), str(tx).c_str() );
    }
    for( uint32_t ty = 0; ty != t_tile_sz; ++ty ) { // note: could merge with above loop, but we want to use ty for consistency
      t_tile_dummy_loads += strprintf( "    in_strip[%s] = in_smem[(threadIdx.x %%%% 32) + %s];\n", str(ty).c_str(), str(ty).c_str() );
      t_tile_loads += strprintf( "    in_strip[%s] = in_smem[%%(t_tile_sz)*%%(threadIdx.x_patch_tile)+%s];\n",
				 str(ty).c_str(), str(ty).c_str() );
    }

    t_tile_stores += "  int32_t tpix[%(t_tile_sz)];\n";
    t_tile_stores += "  int32_t tcix[%(t_tile_sz)];\n";

    // FIXME: should somehow assert that both out_ix and patch_ix_N have the same dims here
    for( uint32_t ty = 0; ty != t_tile_sz; ++ty ) { 
      t_tile_stores += strprintf( "  tpix[%s] = %%(patch_ix_%s_img)*%%(out_ix_img_sz) + \n"
				  "   ( %%(patch_ix_%s) %%%% %%(patch_ix_%s_img_sz) ); // cache out patch ixs\n ",
				  str(ty).c_str(), str(ty).c_str(), str(ty).c_str(), str(ty).c_str() );
    }
    for( uint32_t ty = 0; ty != t_tile_sz; ++ty ) { 
      t_tile_stores += strprintf( "  tcix[%s] = (%%(out_chan_ix)+%s)*%%(out_ix_chan_sz); // cache out chan ixs\n",
				  str(ty).c_str(), str(ty).c_str() );
    }
	
    t_tile_dummy_stores += " out[0] = 0.0f\n";
    for( uint32_t ty = 0; ty != t_tile_sz; ++ty ) {
      t_tile_stores += "  if( %(patch_ix_"+str(ty)+") >= %(patch_ix_0_sz) ) { return; } "
	"// this patch and the following are off-the-end patches, so don't store them.\n";
      for( uint32_t tx = 0; tx != t_tile_sz; ++tx ) {
	t_tile_fmas += strprintf( "    out_tile[%s] += filts_strip[%s]*in_strip[%s];\n", 
				  str((ty*t_tile_sz+tx)).c_str(), str(tx).c_str(), str(ty).c_str() );
	string const ve = strprintf( "%sout_tile[%s] + filts_strip[%s])", conv_has_relu ? "max(0.0f," : "(",
				     str((ty*t_tile_sz+tx)).c_str(), str(tx).c_str() );
	t_tile_stores += strprintf( "if( tcix[%s] < (%%(out_ix_chan_dim)*%%(out_ix_chan_sz)) ) { "
				    "out[ tpix[%s] + tcix[%s] ] = %s; }\n",
				    str(tx).c_str(), str(ty).c_str(), str(tx).c_str(), ve.c_str() );
	t_tile_dummy_stores += " + " + ve + "\n";
      }
    }
    t_tile_dummy_stores += ";\n";

    // note: newline (and semi-unwanted semi-colon) from src will go after blocks, hence no newline on these lines
    t_tile_fmas += "    // end t_tile_fmas"; 
    t_tile_loads += "    // end t_tile_loads";
    t_tile_dummy_loads += "    // end t_tile_dummy_loads";
    t_tile_stores += "  // end t_tile_stores";
    tf_exprs.push_back( std::make_pair( "t_tile_fmas", t_tile_fmas ) );
    tf_exprs.push_back( std::make_pair( "t_tile_smem_loads", t_tile_smem_loads ) );
    tf_exprs.push_back( std::make_pair( "t_tile_loads", t_tile_loads ) );
    tf_exprs.push_back( std::make_pair( "t_tile_dummy_loads", t_tile_dummy_loads ) );
    tf_exprs.push_back( std::make_pair( "t_tile_stores", t_tile_stores ) );
    tf_exprs.push_back( std::make_pair( "t_tile_dummy_stores", t_tile_dummy_stores ) );

    // for error checking, (re-) calculate the sizes of the arguments (note: in elements, not bytes)
    cf.arg_sizes.push_back( get_sz( tf_exprs, "filts_xp_ix" ) );
    cf.arg_sizes.push_back( cio_out.chans ); // biases_sz
    cf.arg_sizes.push_back( get_sz( tf_exprs, "in_ix" ) );
    cf.arg_sizes.push_back( out_ix_sz );

    rfgi.instantiate_template( cu_prog_str );
    return cf;
  }

  cu_func_t & conv_pipe_fwd_t::gen_op_lrn( p_conv_op_t const & cop, conv_io_t const & cio_in, conv_io_t const & cio_out ) {
    // note: cio_in and cio_out are derived from cop->bots[0] and cop->tops[0]
    assert_st( cio_in.sz == cio_out.sz );
    assert_st( cio_in.chans == cio_out.chans );
    // FIXME: make {alpha, beta, k} into passed params (and support that somehow)
    rtc_func_gen_info_t rfgi{"lrn",
      { {"num_imgs",str(num_imgs)},{"chans",str(cio_in.chans)},{"ysz",str(cio_in.sz.d[1])},{"xsz",str(cio_in.sz.d[0])}
	,{"local_size",str(cop->lrn_local_size)},{"alpha",str(cop->lrn_alpha)},{"beta",str(cop->lrn_beta)},{"k",str(cop->lrn_k)} } };
    cu_func_t & cf = rfgi.init( cu_funcs );
    vect_pair_str_str & tf_exprs = rfgi.tf_exprs;
    if( cf.finalized ) { return cf; } // already generated
    assert_st( cop->lrn_local_size & 1 ); // we're only supporting centerable windows
    vect_string const cio_dims{"img","chan","y","x"};
    insert_nda_exprs( tf_exprs, "tix", vect_string{"img","y","x"}, 
		      vect_uint32_t{num_imgs,cio_out.sz.d[1],cio_out.sz.d[0]} );
    insert_nda_exprs( tf_exprs, "out_ix", cio_dims, 
		      vect_uint32_t{num_imgs,cio_out.chans,cio_out.sz.d[1],cio_out.sz.d[0]} );
    uint32_t const out_ix_sz = get_sz( tf_exprs, "out_ix" );
    cf.tpb = 256;
    cf.blks = u32_ceil_div( out_ix_sz / cio_out.chans, cf.tpb ); // handle one img,y,x per thread (across chans)
    cf.arg_sizes.push_back( out_ix_sz );
    cf.arg_sizes.push_back( out_ix_sz );
    rfgi.instantiate_template( cu_prog_str );
    return cf;
  }

  cu_func_t & conv_pipe_fwd_t::gen_op_copy( p_conv_op_t const & cop, conv_io_t const & cio_in, conv_io_t const & cio_out, uint32_t const ocix ) {
    // note: cio_in and cio_out are derived from cop->bots[bi] and cop->tops[0]
    assert_st( cio_in.sz == cio_out.sz );
    rtc_func_gen_info_t rfgi{"copy",
      { {"num_imgs",str(num_imgs)},{"in_chans",str(cio_in.chans)},{"ysz",str(cio_in.sz.d[1])},{"xsz",str(cio_in.sz.d[0])}
	,{"out_chans",str(cio_out.chans)},{"ocix",str(ocix)} } };
    cu_func_t & cf = rfgi.init( cu_funcs );
    vect_pair_str_str & tf_exprs = rfgi.tf_exprs;
    if( cf.finalized ) { return cf; } // already generated
    vect_string const cio_dims{"img","chan","y","x"};
    insert_nda_exprs( tf_exprs, "in_ix", vect_string{"img","chan","y","x"}, 
		      vect_uint32_t{num_imgs,cio_in.chans,cio_in.sz.d[1],cio_in.sz.d[0]} );
    insert_nda_exprs( tf_exprs, "out_ix", cio_dims, 
		      vect_uint32_t{num_imgs,cio_out.chans,cio_out.sz.d[1],cio_out.sz.d[0]} );
    uint32_t const in_ix_sz = get_sz( tf_exprs, "in_ix" );
    uint32_t const out_ix_sz = get_sz( tf_exprs, "out_ix" );
    cf.tpb = 256;
    cf.blks = u32_ceil_div( in_ix_sz, cf.tpb ); // handle one img,y,x per thread (across chans)
    cf.arg_sizes.push_back( in_ix_sz );
    cf.arg_sizes.push_back( out_ix_sz );
    rfgi.instantiate_template( cu_prog_str );
    return cf;
  }

  cu_func_t & conv_pipe_fwd_t::gen_op_relu( conv_io_t const & cio_out ) {
    uint32_t const out_sz = cio_out.sz.dims_prod() * cio_out.chans * num_imgs;
    rtc_func_gen_info_t rfgi{"relu", { {"out_sz",str(out_sz)} } };
    cu_func_t & cf = rfgi.init( cu_funcs );
    //vect_pair_str_str & tf_exprs = rfgi.tf_exprs;
    if( cf.finalized ) { return cf; } // already generated
    rfgi.set_tpb_blks_for_one_output_per_thread( out_sz );
    rfgi.instantiate_template( cu_prog_str );
    return cf;
  }

  struct red_op_t {
    string tag;
    string iv;
    string ts;
    red_op_t( string const & tag_ ) { 
      tag = tag_; ts = "float"; 
      if( 0 ) { }
      else if( tag == "min" ) { iv = "FLT_MAX"; }
      else if( tag == "max" ) { iv = "-FLT_MAX"; }
      else if( tag == "sum" ) { iv = "0"; }
      else if( tag == "hist" ) { iv = "0"; }
      else if( tag == "cnt" ) { iv = "0"; }
      else { assert_st(0); } // unknown tag/op
    }
    string param_str( void ) { return strprintf( "%s * %s_in, %s * %s_out", ts.c_str(), tag.c_str(), ts.c_str(), tag.c_str() ); }
    string decl_str( void ) { return strprintf( "    %s %s_v = %s; __shared__ %s %s_smem[tbp];", 
						ts.c_str(), tag.c_str(), iv.c_str(), ts.c_str(), tag.c_str() ); }
    string in_proc_str( void ) { 
      if( tag == "hist" ) { return strprintf( " (%s_in[ix]>1000) ", tag.c_str() ); }
      if( tag == "cnt" ) { return strprintf( "1" ); }
      else { return strprintf( " %s_in[ix]", tag.c_str() ); }
    }
    string load_str( void ) { return strprintf( "    if( ix < in_sz ) { "
						"if(primary_in) { %s_v = %s; } else { %s_v = %s_in[ix]; } } %s_smem[tid] = %s_v;", 
						tag.c_str(), in_proc_str().c_str(), tag.c_str(), tag.c_str(), tag.c_str(), 
						tag.c_str() ); }
    string update_v_str( string const & from_expr ) {
      if( tag == "min" || tag == "max" ) {
	return strprintf( "%s_v = %s( %s_v, %s );", tag.c_str(), tag.c_str(), tag.c_str(), from_expr.c_str() ); 
      } else if( tag == "sum" || tag == "hist" || tag == "cnt" ) {
      	return strprintf( "%s_v += %s;", tag.c_str(), from_expr.c_str() ); 
      } else { assert_st(0); }
    }
    string store_str( void ) {
      return strprintf( "    if( !tid ) { %s_out[blockIdx.x] = %s_v; }", tag.c_str(), tag.c_str() ); }

  };
  typedef vector< red_op_t > vect_red_op_t; 

  vect_string conv_pipe_fwd_t::gen_op_stats( conv_io_t const & cio_in, string const & top_in ) {
    vect_red_op_t reds{ red_op_t("min"), red_op_t("max"), red_op_t("sum"), red_op_t("hist"), red_op_t("cnt")  };
    uint32_t in_sz = cio_in.sz.dims_prod() * cio_in.chans * num_imgs;
    uint32_t primary_in = 1;
    assert_st( in_sz );
    vect_string cur_ins;
    for( uint32_t i = 0; i != reds.size(); ++i ) { cur_ins.push_back( top_in ); }
    
    while( in_sz > 1 ) {
      rtc_func_gen_info_t rfgi{"stats", { } };
      cu_func_t & cf = rfgi.init( cu_funcs );
      if( !cf.finalized ) { 
	cf.tpb = 256;
	// FIXME: handle dynamic block sizes better?
	//cf.blks = u32_ceil_div( in_sz, cf.tpb );
	cf.blks = 0;
	vect_string params;
	vect_string body;
	for( uint32_t i = 0; i != reds.size(); ++i ) { 
	  params.push_back(reds[i].param_str());
	  // FIXME: for now, we disable these size checks ...
	  //cf.arg_sizes.push_back( in_sz );
	  //cf.arg_sizes.push_back( cf.blks );
	  body.push_back(reds[i].decl_str());
	  body.push_back(reds[i].load_str());
	}
	body.push_back( "  __syncthreads();" );
	uint32_t const tbp = 256;
	uint32_t const warp_sz = 32;
	for( uint32_t smb = tbp / 2; smb > warp_sz; smb /= 2 ) {
	  body.push_back( strprintf( "  if( tid < %s ) {", str(smb).c_str() ) );
	  for( uint32_t i = 0; i != reds.size(); ++i ) { 
	    body.push_back( strprintf("    %s_smem[tid] = ",reds[i].tag.c_str()) +
			    reds[i].update_v_str( strprintf( "%s_smem[tid+%s]", reds[i].tag.c_str(), str(smb).c_str() )));
	  }
	  body.push_back( "  }" );
	  body.push_back( "  __syncthreads();" );
	}
	body.push_back( strprintf( "  if( tid < %s ) {", str(warp_sz).c_str() ) );
	for( uint32_t i = 0; i != reds.size(); ++i ) {
	  body.push_back( reds[i].update_v_str( strprintf( "%s_smem[tid+%s]", reds[i].tag.c_str(), str(warp_sz).c_str() )));
	  for( uint32_t wb = warp_sz / 2; wb; wb /= 2 ) {
	    body.push_back( reds[i].update_v_str( strprintf( "__shfl_down( %s_v,%s )", reds[i].tag.c_str(), str(wb).c_str() ) ) );
	  }
	} 
	body.push_back( "  }" );
	for( uint32_t i = 0; i != reds.size(); ++i ) { body.push_back( reds[i].store_str() ); }

	rfgi.tf_exprs.push_back( std::make_pair( "params", join(params,", ") ) );
	rfgi.tf_exprs.push_back( std::make_pair( "body", join(body,"\n") ) );

	rfgi.instantiate_template( cu_prog_str );
      }
      uint32_t const out_sz = u32_ceil_div( in_sz, cf.tpb );
      vect_string cur_outs;
      vect_string args;
      for( uint32_t i = 0; i != reds.size(); ++i ) { 
	string cur_out = top_in + "_" + reds[i].tag + "_out_sz_" + str(out_sz);
	must_insert( *cups, cur_out, make_shared<cup_float>( out_sz ) ); 
	cur_outs.push_back( cur_out );
	args.push_back( cur_ins[i] );
	args.push_back( cur_out );
      }
      fwd_calls.push_back( cu_func_call_t{ cf.name, args, {in_sz, primary_in} } );
      cur_ins = cur_outs;
      in_sz = out_sz;
      primary_in = 0;
    }
    assert_st( in_sz == 1 );
    return cur_ins;
  }

  void conv_pipe_fwd_t::gen_op_quantize( conv_io_t const & cio_in, string const & top_in, 
					 uint32_t const & max_val, uint32_t const & keep_bits ) {
    uint32_t drop_bits = 0;
    while( max_val > (1U<<(keep_bits+drop_bits)) ) { ++drop_bits; }
    uint32_t drop_mask = ((1<<drop_bits)-1);

    uint32_t in_sz = cio_in.sz.dims_prod() * cio_in.chans * num_imgs;
    assert_st( in_sz );
    rtc_func_gen_info_t rfgi{"quantize", { } };
    cu_func_t & cf = rfgi.init( cu_funcs );
    if( !cf.finalized ) { 
      cf.tpb = 256;
      // FIXME: handle dynamic block sizes better?
      //cf.blks = u32_ceil_div( in_sz, cf.tpb );
      cf.blks = 0;
      vect_string body;
      rfgi.tf_exprs.push_back( std::make_pair( "body", join(body,"\n") ) );
      rfgi.instantiate_template( cu_prog_str );
    }
    fwd_calls.push_back( cu_func_call_t{ cf.name, {top_in}, {in_sz,max_val,drop_mask} } );
  }

  cu_func_t & conv_pipe_fwd_t::gen_op_xpose( p_conv_op_t const & cop, gen_layout_info_t const & gli ) {
    u32_pt_t kern_sz = cop->kern_sz;
    assert_st( kern_sz.both_dims_non_zero() );
    rtc_func_gen_info_t rfgi{"xpose_filts", {
	{"out_chans",str(cop->out_chans)},{"in_chans",str(gli.in_chans)},{"kysz",str(kern_sz.d[1])},{"kxsz",str(kern_sz.d[0])} 
      } };
    cu_func_t & cf = rfgi.init( cu_funcs );
    vect_pair_str_str & tf_exprs = rfgi.tf_exprs;
    if( cf.finalized ) { return cf; } // already generated
    tf_exprs.push_back( make_pair( "t_tile_sz", str(t_tile_sz) ) );
    insert_nda_exprs( tf_exprs, "filts_ix", vect_string{"out_chan","in_chan","y","x"}, 
		      vect_uint32_t{cop->out_chans,gli.in_chans,kern_sz.d[1],kern_sz.d[0]} );
    insert_nda_exprs( tf_exprs, "filts_xp_ix", vect_string{"in_chan","y","x","out_chan_blk","out_chan_reg","out_chan_tile"}, 
		      vect_uint32_t{gli.in_chans,kern_sz.d[1],kern_sz.d[0],
			  gli.bix_out_chan_blk_sz,t_tile_sz,gli.tix_out_chan_tile_sz} );
    insert_nda_exprs( tf_exprs, "fioc", vect_string{"out_chan_blk","out_chan_tile","out_chan_reg"}, 
		      vect_uint32_t{ gli.bix_out_chan_blk_sz,gli.tix_out_chan_tile_sz,t_tile_sz} );
    uint32_t const filts_ix_sz = get_sz( tf_exprs, "filts_ix" );
    cf.tpb = 256;
    cf.blks = u32_ceil_div( filts_ix_sz, cf.tpb ); // handle one img,y,x per thread (across chans)
    cf.arg_sizes.push_back( filts_ix_sz );
    cf.arg_sizes.push_back( get_sz( tf_exprs, "filts_xp_ix" ) );
    rfgi.instantiate_template( cu_prog_str );
    return cf;
  }

  void conv_pipe_fwd_t::gen_xpose( cu_func_t const & cf, 
				   p_conv_op_t const & cop, string const & filts_name, gen_layout_info_t const & gli ) {
    string const filts_xposed_name = filts_name + "_xposed"; // note: doesn't exist yet
    init_calls.push_back( cu_func_call_t{ cf.name, vect_string{ filts_name, filts_xposed_name } } );
    // create cup for xposed version of filts
    p_cup_float filts_cup = must_find( *cups, filts_name );
    // note that if out_chans doesn't divide evenly by t_tile_sz, the xposed array will have internal padding/garbage
    // FIXME: partially untested! none of alexnet, googlenet, nin have any layers with out_chans not divisible by 8 (def. t_tile_sz)
    uint32_t const padded_out_chans = gli.bix_out_chan_blk_sz * gli.tix_out_chan_tile_sz * t_tile_sz;
    must_insert( *cups, filts_xposed_name, make_shared<cup_float>( filts_cup->sz / cop->out_chans * padded_out_chans ) ); 
  }


  void conv_pipe_fwd_t::gen_op( p_conv_op_t const & cop ) {
    string const tag_id_str = as_pyid( cop->tag );
    //char const * const tag_id = tag_id_str.c_str();
    assert_st( cop->tops.size() == 1 );
    p_conv_node_t node_out = cp->must_get_node( cop->tops[0] );
    conv_io_t & cio_out = node_out->cio;

    if( cop->type == Concat_str ) {      
      vect_string arg_ids;
      arg_ids.push_back( "" ); // placeholder for per-copy input
      arg_ids.push_back( as_pyid(cop->tops[0]) );
      uint32_t chans_out_done = 0;
      for( uint32_t bi = 0; bi != cop->bots.size(); ++bi ) {
	conv_io_t & cio_in = cp->must_get_node( cop->bots[bi] )->cio;
	assert_st( cio_in.sz == cio_out.sz );
	assert_st( chans_out_done+cio_in.chans <= cio_out.chans );
	cu_func_t & cf = gen_op_copy( cop, cio_in, cio_out, chans_out_done );
	arg_ids[0] = as_pyid(cop->bots[bi]);
	fwd_calls.push_back( cu_func_call_t{ cf.name, arg_ids, {}, tag_id_str } );
	chans_out_done += cio_in.chans;
      }
      assert_st( chans_out_done == cio_out.chans );
      return;
    }

    assert_st( cop->bots.size() == 1 );
    p_conv_node_t node_in = cp->must_get_node( cop->bots[0] );
    conv_io_t & cio_in = node_in->cio;
    bool const is_conv = cop->type == Convolution_str;
    if( is_conv || (cop->type == Pooling_str) ) {
      vect_string arg_ids;
      string const filts_id = tag_id_str + "_filts";
      string const biases_id = tag_id_str + "_biases";
      if( is_conv ) {
	arg_ids.push_back( filts_id + "_xposed" );
	arg_ids.push_back( biases_id );
      }
      arg_ids.push_back( as_pyid(cop->bots[0]) );
      arg_ids.push_back( as_pyid(cop->tops[0]) );
      cu_func_t & cf = gen_op_kern( cop, cio_in, node_out );
      fwd_calls.push_back( cu_func_call_t{ cf.name, arg_ids, {}, tag_id_str } );
      if( is_conv ) {
	assert_st( cio_out.chans == cop->out_chans );
	vect_uint32_t const & arg_sizes = cf.arg_sizes;
	assert_st( arg_sizes.size() == 4 );
	cu_func_t & xpose_cf = gen_op_xpose( cop, cf.gli );
	add_op_param( filts_id, xpose_cf.arg_sizes[0] );
	bool const did_ins = filts_names.insert( filts_id ).second; // track filt names
	if( did_ins ) { gen_xpose( xpose_cf, cop, filts_id, cf.gli ); } // newly-seen/used filter, so set up to transpose it
	add_op_param( biases_id, arg_sizes[1] );
      }
    } else if( cop->type == ReLU_str ) {
      // check that this is a single in-out in-place operation
      assert_st( cop->bots[0] == cop->tops[0] );
      fwd_calls.push_back( cu_func_call_t{ gen_op_relu( cio_out ).name, { as_pyid(cop->tops[0]) }, {}, tag_id_str } );
    } else if( cop->type == LRN_str ) {
      assert_st( cop->bots.size() == 1 );
      conv_io_t & cio_in = cp->must_get_node( cop->bots[0] )->cio;
      assert_st( cop->tops.size() == 1 );
      conv_io_t & cio_out = cp->must_get_node( cop->tops[0] )->cio;
      vect_string arg_ids;
      arg_ids.push_back( as_pyid(cop->bots[0]) );
      arg_ids.push_back( as_pyid(cop->tops[0]) );
      cu_func_t & cf = gen_op_lrn( cop, cio_in, cio_out );
      fwd_calls.push_back( cu_func_call_t{ cf.name, arg_ids, {}, tag_id_str } );
    } else if( cop->type == Dropout_str ) {
      // check that this is a single in-out in-place operation
      assert_st( cop->bots[0] == cop->tops[0] );
      // ignore for fwd
    } else { rt_err( "gen_op: unhandled op of type: " + cop->type ); }
  }

  void conv_pipe_fwd_t::gen_node( string const & name, p_conv_node_t const & node ) {
    conv_io_t & cio = node->cio;
    must_insert( *cups, as_pyid(name), make_shared<cup_float>( num_imgs * cio.chans * cio.sz.dims_prod() ) ); 
  }

  // quantize command line example:
  // export QOPTS="keep_bits=8,quantize=(_=(name=conv1,max_val=4096),_=(name=conv2,max_val=1024),_=(name=conv3,max_val=1024),_=(name=conv4,max_val=512),_=(name=conv5,max_val=512))

  // CUDA_VISIBLE_DEVICES=0 DISABLE_CUDNN=0 time boda test_lmdb --model-name=alexnet_ng_conv --num-to-read=1000 --run-cnet="(in_sz=227 227,in_num_imgs=20,ptt_fn=%(models_dir)/%(model_name)/train_val.prototxt,trained_fn=%(models_dir)/%(model_name)/best.caffemodel,out_layer_name=fc8-conv,compute_mode=1,conv_fwd=(mode=nvrtc,enable_stats=0,show_rtc_calls=0,${QOPTS}))"


  void conv_pipe_fwd_t::gen_ops_rec( string const & node_name ) {
    p_conv_node_t node = cp->must_get_node( node_name );
    // setup source nodes here, otherwise print with thier writing op
    bool writer_is_conv = 0;
    if( node->top_for.empty() ) { gen_node( node_name, node ); }
    else { 
      assert( node->top_for.size() == 1 ); // multiple writers not handled
      p_conv_op_t const & cop = cp->get_op( node->top_for[0] );
      writer_is_conv = ( cop->type == Convolution_str );
    }
    // in-place ops for this node
    for( vect_p_conv_op_t::const_iterator j = node->in_place_ops.begin(); j != node->in_place_ops.end(); ++j ) { 
      // skip first operation if it is a ReLU on a node written by a conv, as it will have been fused into the conv:
      if( writer_is_conv && (j == node->in_place_ops.begin()) && ((*j)->type == ReLU_str) ) { continue; } 
      gen_op( *j ); 
    }
    // generate stats gathering call
    // printf( "node_name=%s\n", str(node_name).c_str() );
    for( vect_p_quantize_ops_t::const_iterator i = quantize.begin(); i != quantize.end(); ++i ) {
      if( node_name != (*i)->name ) { continue; }
      gen_op_quantize( node->cio, as_pyid(node_name), (*i)->max_val, (*i)->keep_bits );
    }
    if( enable_stats ) {
      vect_string new_stats_names = gen_op_stats( node->cio, as_pyid(node_name) );
      stats_names.insert( stats_names.end(), new_stats_names.begin(), new_stats_names.end() );
    }

    for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) {
      p_conv_op_t const & cop = cp->get_op( *i );
      if( !cop->on_seen_bot() ) { continue; } // wait till we've seen all bottoms
      gen_op( cop );
      for( vect_string::const_iterator j = cop->tops.begin(); j != cop->tops.end(); ++j ) { 
	gen_node( *j, cp->must_get_node(*j) );
	gen_ops_rec( *j ); 
      }
    }
  }
  string cu_base_decls = R"rstr(
//typedef unsigned uint32_t;
typedef int int32_t;
typedef long long int64_t;
float const FLT_MAX = /*0x1.fffffep127f*/ 340282346638528859811704183484516925440.0f;

)rstr";

  void conv_pipe_fwd_t::init( p_conv_pipe_t const & cp_, uint32_t const & num_imgs_ ) {
    cp = cp_;
    assert_st( cp );
    assert_st( cp->finalized );
    num_imgs = num_imgs_;
    assert_st( num_imgs );
    // need to init CUDA prior to potential cuda mallocs during setup/codegen
    cu_err_chk( cuInit( 0 ), "cuInit" ); 
    cu_err_chk( cuDeviceGet( &cu_dev, 0 ), "cuDeviceGet" );
    //cu_err_chk( cuCtxCreate( &cu_context, 0, cu_dev ), "cuCtxCreate" );
    cu_err_chk( cuDevicePrimaryCtxRetain( &cu_context, cu_dev ), "cuDevicePrimaryCtxRetain" );
    cu_err_chk( cuCtxSetCurrent( cu_context ), "cuCtxSetCurrent" ); // is this always needed/okay?

    // cu_err_chk( cuCtxSetCacheConfig( CU_FUNC_CACHE_PREFER_L1 ), "cuCtxSetCacheConfig" ); // does nothing?

    cups.reset( new map_str_p_cup_float_t );
    cu_prog_str += cu_base_decls;
    for( vect_string::const_iterator i = def.begin(); i != def.end(); ++i ) { cu_prog_str += "#define "+*i+" 1\n"; }
    cp->topo_visit_setup();
    for( vect_string::const_iterator i = cp->bots.begin(); i != cp->bots.end(); ++i ) { gen_ops_rec( *i ); }

    write_whole_fn( "out.cu", cu_prog_str );
    string const prog_ptx = nvrtc_compile( cu_prog_str, show_compile_log );
    write_whole_fn( "out.ptx", prog_ptx );
    //printf( "cu_prog_str=%s\n", str(cu_prog_str).c_str() );
    //printf( "prog_ptx=%s\n", str(prog_ptx).c_str() );
    cu_err_chk( cuModuleLoadDataEx( &cu_mod, &prog_ptx[0], 0, 0, 0 ), "cuModuleLoadDataEx" );
    for( cu_funcs_t::iterator i = cu_funcs.begin(); i != cu_funcs.end(); ++i ) {
      cu_err_chk( cuModuleGetFunction( &i->second.cu_func, cu_mod, i->first.c_str() ), "cuModuleGetFunction" );
      // FIXME: i'd like to play with enabling L1 caching for these kernels, but it's not clear how to do that
      // cu_err_chk( cuFuncSetCacheConfig( i->second.cu_func, CU_FUNC_CACHE_PREFER_L1 ), "cuFuncSetCacheConfig" ); // does nothing?
      if( show_func_attrs ) {
	string cfas = cu_get_all_func_attrs( i->second.cu_func );
	printf( "%s: \n%s", i->first.c_str(), str(cfas).c_str() );
      }
    }
    copy_named_ndas_to_cups( op_param_names, *cp->op_params, *cups ); // copy op_params in

    // transpose filters ... and do any other init-time work added after this comment was written ;)
    for( vect_cu_func_call_t::iterator i = init_calls.begin(); i != init_calls.end(); ++i ) { run_cfc( *i ); }
    cu_err_chk( cuCtxSynchronize(), "cuCtxSynchronize" );
  }

  void conv_pipe_fwd_t::run_cfc( cu_func_call_t & cfc ) {
    cu_func_t const & cf = must_find( cu_funcs, cfc.cu_func_name );
    vect_rp_void cu_func_args;
    uint32_t blks = cf.blks; // if non-zero, blks is static, and we can check arg sizes
    //printf( "cf.name=%s cf.arg_sizes=%s cfc.args.size()=%s\n", str(cf.name).c_str(), str(cf.arg_sizes).c_str(), str(cfc.args.size()).c_str() );
    if( blks ) { assert( cf.arg_sizes.size() == cfc.args.size() ); }
    for( uint32_t i = 0; i != cfc.args.size(); ++i ) {
      p_cup_float arg = must_find( *cups, cfc.args[i] );
      // printf( "  cfc.args[i]=%s arg->sz=%s\n", str(cfc.args[i]).c_str(), str(arg->sz).c_str() );
      if( blks ) { assert_st( arg->sz == cf.arg_sizes[i] ); }
      cu_func_args.push_back( &arg->p );
    }
    // add u32 args
    for( uint32_t i = 0; i != cfc.u32_args.size(); ++i ) { cu_func_args.push_back( (void *)&cfc.u32_args[i] ); }
    // FIXME: check that we're passing the correct # of args here somehow.
    if( !blks ) { // handle dynamic # of blks case
      // FIXME: pretty limited / special cased here
      assert_st( cfc.u32_args.size() > 0 );
      blks = u32_ceil_div( cfc.u32_args[0], cf.tpb );
    }
    if( show_rtc_calls ) { 
      printf( "%s( %s -- %s ) tpb=%s blks=%s\n", str(cfc.cu_func_name).c_str(), str(cfc.args).c_str(), str(cfc.u32_args).c_str(),
	      str(cf.tpb).c_str(), str(blks).c_str() );
    }
    cfc.ensure_evs();
    cu_err_chk( cuEventRecord( *cfc.b_ev, 0 ), "cuEventRecord" );
    cu_err_chk( cuLaunchKernel( cf.cu_func,
				blks, 1, 1, // grid x,y,z dims
				cf.tpb, 1, 1, // block x,y,z dims
				0, 0, // smem_bytes, stream_ix
				&cu_func_args[0], // cu_func's args
				0 ), "cuLaunchKernel" ); // unused 'extra' arg-passing arg      
    cu_err_chk( cuEventRecord( *cfc.e_ev, 0 ), "cuEventRecord" );
  }


  void conv_pipe_fwd_t::run_fwd( p_map_str_p_nda_float_t const & fwd ) {
    timer_t t("conv_pipe_fwd_t::run_fwd");
    if( enable_prof ) { cuProfilerStart(); }
    //printf("run_fwd() begin\n");
    copy_named_ndas_to_cups( cp->bots, *fwd, *cups ); // copy sources in
    //printf("run_fwd() exec\n");
    for( vect_cu_func_call_t::iterator i = fwd_calls.begin(); i != fwd_calls.end(); ++i ) { run_cfc( *i ); }
    cu_err_chk( cuCtxSynchronize(), "cuCtxSynchronize" );
    if( enable_prof ) { cuProfilerStop(); }
    if( !per_call_fn.empty() ) {
      string per_call_str;
      for( vect_cu_func_call_t::iterator i = fwd_calls.begin(); i != fwd_calls.end(); ++i ) {
	cu_func_call_t & cfc = *i;
	if( cfc.call_tag.empty() ) { continue; }
	float cfc_dur = 0.0f;
	cu_err_chk( cuEventElapsedTime( &cfc_dur, *cfc.b_ev, *cfc.e_ev ), "cuEventElapsedTime" );
	per_call_str += strprintf( "per_layer_time['%s']=%s\n", str(cfc.call_tag).c_str(), str(cfc_dur/1000.0).c_str() );
      }
      write_whole_fn( per_call_fn, per_call_str );
    }

    //printf("run_fwd() copy out\n");
    cp->fwd_alloc_ndas( fwd, num_imgs, 1 ); // sinks_only=1
    copy_named_cups_to_ndas( cp->tops, *cups, *fwd ); // copy sinks out
    update_stats();
    //printf("run_fwd() done\n");
  }
  
#include"gen/nvrtc_util.cc.nesi_gen.cc"
}
