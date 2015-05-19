// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"nvrtc_util.H"
#include"str_util.H"
#include"rand_util.H"
#include"has_main.H"
#include"timers.H"
#include<nvrtc.h>
#include<cuda.h>

// for conv_pipe_fwd_t
#include"conv_util.H"

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
  string nvrtc_compile( string const & cuda_prog_str ) {
    timer_t t("nvrtc_compile");
    p_nvrtcProgram cuda_prog = make_p_nvrtcProgram( cuda_prog_str );
    vect_string cc_opts = {};
    auto const comp_ret = nvrtcCompileProgram( cuda_prog.get(), cc_opts.size(), &get_vect_rp_const_char( cc_opts )[0] );
    string const log = nvrtc_get_compile_log( cuda_prog );
    //printf( "log=%s\n", str(log).c_str() );
    nvrtc_err_chk( comp_ret, ("nvrtcCompileProgram\n"+log).c_str() ); // delay error check until after getting log
    return nvrtc_get_ptx( cuda_prog );
  }
  
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
      string const prog_ptx = nvrtc_compile( *prog_str );

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

  typedef map< string, p_cup_float > map_str_p_cup_float_t;
  typedef shared_ptr< map_str_p_cup_float_t > p_map_str_p_cup_float_t;

  void copy_named_ndas_to_cups( vect_string const & names, map_str_p_nda_float_t const & ndas, map_str_p_cup_float_t const & cups ) {
    for( vect_string::const_iterator i = names.begin(); i != names.end(); ++i ) {
      string const pyid = as_pyid( *i );
      //printf( "(*i)=%s pyid=%s\n", str((*i)).c_str(), str(pyid).c_str() );
      cu_copy_nda_to_cup( must_find( cups, pyid ), must_find( ndas, pyid ) );
    }
  }
  void copy_named_cups_to_ndas( vect_string const & names, map_str_p_cup_float_t const & cups, map_str_p_nda_float_t & ndas ) {
    for( vect_string::const_iterator i = names.begin(); i != names.end(); ++i ) {
      string const pyid = as_pyid( *i );
      printf( "(*i)=%s pyid=%s\n", str((*i)).c_str(), str(pyid).c_str() );
      cu_copy_cup_to_nda( must_find( ndas, pyid ), must_find( cups, pyid ) );
    }
  }

  struct cu_func_call_t { 
    string cu_func_name; 
    vect_string args; 
  };
  typedef vector< cu_func_call_t > vect_cu_func_call_t; 
  struct cu_func_t { 
    vect_uint32_t arg_sizes;
    uint32_t tpb;
    uint32_t blks;
    CUfunction cu_func; 
  };
  typedef map< string, cu_func_t > cu_funcs_t;

  struct conv_pipe_fwd_t {
    p_conv_pipe_t cp;
    uint32_t num_imgs;
    p_map_str_p_cup_float_t cups;
    vect_string op_param_names;

    //nvrtc/cuda state
    CUdevice cu_dev;
    CUcontext cu_context;
    CUmodule cu_mod;
    CUfunction cu_func;

    string cu_prog_str;
    vect_cu_func_call_t fwd_calls;
    cu_funcs_t cu_funcs;

    void init( p_conv_pipe_t const & cp_, uint32_t const & num_imgs_ );
    void run_fwd( p_map_str_p_nda_float_t const & fwd );

  protected:
    string gen_op_conv( uint32_t const & in_pad, uint32_t const & kern_sz, uint32_t const & stride,
			conv_io_t const & cio_in, conv_io_t const & cio_out );
    string gen_op_relu( conv_io_t const & cio_out );
    void gen_node( string const & name, p_conv_node_t const & node );
    void add_op_param( string const & name, uint32_t const & sz );
    void gen_op( p_conv_op_t const & cop );
    void gen_ops_rec( string const & node_name );

  };
  p_conv_pipe_fwd_t make_conv_pipe_fwd_t( p_conv_pipe_t const & cp, uint32_t const & num_imgs ) { 
    p_conv_pipe_fwd_t ret = make_shared<conv_pipe_fwd_t>(); ret->init(cp,num_imgs); return ret; 
  }
  void conv_pipe_fwd_t_run( p_conv_pipe_fwd_t const & cpf, p_map_str_p_nda_float_t const & fwd ) { cpf->run_fwd( fwd ); }

  void conv_pipe_fwd_t::add_op_param( string const & name, uint32_t const & sz ) {
    string const & name_id = as_pyid( name );
    must_insert( *cups, name_id, make_shared<cup_float>( sz ) ); 
    op_param_names.push_back( name );
  }

  string conv_pipe_fwd_t::gen_op_conv( uint32_t const & in_pad, uint32_t const & kern_sz, uint32_t const & stride,
				       conv_io_t const & cio_in, conv_io_t const & cio_out ) {
    // for now, we'll only handle square inputs. however, this is probably too limiting for more than initial tests.
    assert_st( cio_in.sz.dims_are_same() );
    uint32_t const in_dim = cio_in.sz.d[0];
    string const cu_func_name = strprintf( "conv__num_imgs_%s__in_pad_%s__in_dim_%s__in_chans_%s__kern_sz_%s__stride_%s", 
					   str(num_imgs).c_str(), str(in_pad).c_str(), str(in_dim).c_str(), str(cio_in.chans).c_str(),
					   str(kern_sz).c_str(), str(stride).c_str() );
    std::pair< cu_funcs_t::iterator, bool > ins_ret = cu_funcs.insert( make_pair( cu_func_name, cu_func_t{} ) );
    if( !ins_ret.second ) { return cu_func_name; } // already generated
    cu_func_t & cf = ins_ret.first->second;
    
    // for error checking, (re-) calculate the sizes of the arguments (note: in elements, not bytes)
    uint32_t const filts_sz = cio_out.chans * cio_in.chans * kern_sz * kern_sz;
    uint32_t const biases_sz = cio_out.chans;
    uint32_t const in_sz = cio_in.sz.dims_prod() * cio_in.chans * num_imgs;
    uint32_t const out_sz = cio_out.sz.dims_prod() * cio_out.chans * num_imgs;
    cf.arg_sizes = vect_uint32_t{ filts_sz, biases_sz, in_sz, out_sz }; 
    cf.tpb = 256;
    cf.blks = u32_ceil_div( out_sz, cf.tpb );
    cu_prog_str += strprintf( R"rstr(
extern "C"  __global__ void %s( float const * const filts, float const * const biases, float const * const in, float * const out ) {
    uint32_t const ix = blockDim.x * blockIdx.x + threadIdx.x;
    if( ix < %s ) { out[ix] = -4.0f; }
}
)rstr", cu_func_name.c_str(), str(out_sz).c_str() );
    
    //printf( "cu_func_name=%s\n", str(cu_func_name).c_str() );
    return cu_func_name;
  }

  string conv_pipe_fwd_t::gen_op_relu( conv_io_t const & cio_out ) {
    uint32_t const out_sz = cio_out.sz.dims_prod() * cio_out.chans * num_imgs;
    string const cu_func_name = strprintf( "relu__out_sz_%s", str(out_sz).c_str() );
    std::pair< cu_funcs_t::iterator, bool > ins_ret = cu_funcs.insert( make_pair( cu_func_name, cu_func_t{} ) );
    if( !ins_ret.second ) { return cu_func_name; } // already generated
    cu_func_t & cf = ins_ret.first->second;
    cf.arg_sizes = vect_uint32_t{ out_sz }; 
    cf.tpb = 256;
    cf.blks = u32_ceil_div( out_sz, cf.tpb );
    cu_prog_str += strprintf( R"rstr(
extern "C"  __global__ void %s( float * const out ) {
    uint32_t const ix = blockDim.x * blockIdx.x + threadIdx.x;
    if( ix < %s ) { out[ix] = (out[ix] <= 0) ? 0.0f : out[ix]; }
}
)rstr", cu_func_name.c_str(), str(out_sz).c_str() );
    
    //printf( "cu_func_name=%s\n", str(cu_func_name).c_str() );
    return cu_func_name;
  }

  void conv_pipe_fwd_t::gen_op( p_conv_op_t const & cop ) {
    string const tag_id_str = as_pyid( cop->tag );
    //char const * const tag_id = tag_id_str.c_str();

    if( cop->type == Convolution_str ) {
      assert_st( cop->bots.size() == 1 );
      conv_io_t & cio_in = cp->must_get_node( cop->bots[0] )->cio;
      assert_st( cop->tops.size() == 1 );
      conv_io_t & cio_out = cp->must_get_node( cop->tops[0] )->cio;
      u32_pt_t kern_sz = cop->kern_sz;
      if( kern_sz.is_zeros() ) { kern_sz = cio_in.sz; } // 'global' input special case

      assert_st( cio_out.chans == cop->out_chans );

      string const filts_id = tag_id_str + "_filts";
      string const biases_id = tag_id_str + "_biases";
      add_op_param( filts_id, cop->out_chans * cio_in.chans * kern_sz.dims_prod() );
      add_op_param( biases_id, cop->out_chans );

      // for now, we only attempt to handle the (common) case of uniform padding, kernel size, and stride
      assert_st( cop->in_pad.bnds_are_same() );
      assert_st( cop->in_pad.p[0].dims_are_same() );
      assert_st( cop->stride.dims_are_same() );
      assert_st( cop->kern_sz.dims_are_same() );
      fwd_calls.push_back( cu_func_call_t{ gen_op_conv( cop->in_pad.p[0].d[0], cop->kern_sz.d[0], cop->stride.d[0], cio_in, cio_out ), 
	  { filts_id, biases_id, as_pyid(cop->bots[0]), as_pyid(cop->tops[0]) } } );
    }
    else if( cop->type == ReLU_str ) {
      // check that this is a single in-out in-place operation
      assert_st( cop->tops.size() == 1 );
      conv_io_t & cio_out = cp->must_get_node( cop->tops[0] )->cio;
      assert_st( cop->bots.size() == 1 );
      assert_st( cop->bots[0] == cop->tops[0] );
      fwd_calls.push_back( cu_func_call_t{ gen_op_relu( cio_out ), { as_pyid(cop->tops[0]) } } );
    } else { rt_err( "gen_op: unhandled op of type" + cop->type ); }
  }

  void conv_pipe_fwd_t::gen_node( string const & name, p_conv_node_t const & node ) {
    conv_io_t & cio = node->cio;
    must_insert( *cups, as_pyid(name), make_shared<cup_float>( num_imgs * cio.chans * cio.sz.dims_prod() ) ); 
  }

  void conv_pipe_fwd_t::gen_ops_rec( string const & node_name ) {
    p_conv_node_t node = cp->must_get_node( node_name );
    // setup source nodes here, otherwise print with thier writing op
    if( node->top_for.empty() ) { gen_node( node_name, node ); }
    else { assert( node->top_for.size() == 1 ); } // multiple writers not handled
    // in-place ops for this node
    for( vect_p_conv_op_t::const_iterator j = node->in_place_ops.begin(); j != node->in_place_ops.end(); ++j ) { gen_op( *j ); }
    for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) {
      p_conv_op_t const & cop = cp->get_op( *i );
      if( !cop->on_seen_bot() ) { continue; } // wait till we've seen all bottoms
      gen_op( cop );
      for( vect_string::const_iterator j = cop->tops.begin(); j != cop->tops.end(); ++j ) { 
	gen_node( *i, cp->must_get_node(*i) );
	gen_ops_rec( *i ); 
      }
    }
  }
  string cu_base_decls = R"rstr(
typedef unsigned uint32_t;
)rstr";

  void conv_pipe_fwd_t::init( p_conv_pipe_t const & cp_, uint32_t const & num_imgs_ ) {
    cp = cp_;
    assert_st( cp );
    assert_st( cp->finalized );
    num_imgs = num_imgs_;
    assert_st( num_imgs );
    cups.reset( new map_str_p_cup_float_t );

    cu_prog_str += cu_base_decls;
    
    cp->topo_visit_setup();
    for( vect_string::const_iterator i = cp->bots.begin(); i != cp->bots.end(); ++i ) { gen_ops_rec( *i ); }

    string const prog_ptx = nvrtc_compile( cu_prog_str );

    //printf( "cu_prog_str=%s\n", str(cu_prog_str).c_str() );
    //printf( "prog_ptx=%s\n", str(prog_ptx).c_str() );
    cu_err_chk( cuInit( 0 ), "cuInit" );
    cu_err_chk( cuDeviceGet( &cu_dev, 0 ), "cuDeviceGet" );
    cu_err_chk( cuCtxCreate( &cu_context, 0, cu_dev ), "cuCtxCreate" );
    cu_err_chk( cuModuleLoadDataEx( &cu_mod, &prog_ptx[0], 0, 0, 0 ), "cuModuleLoadDataEx" );
    for( cu_funcs_t::iterator i = cu_funcs.begin(); i != cu_funcs.end(); ++i ) {
      cu_err_chk( cuModuleGetFunction( &i->second.cu_func, cu_mod, i->first.c_str() ), "cuModuleGetFunction" );
    }

    copy_named_ndas_to_cups( op_param_names, *cp->op_params, *cups ); // copy op_params in  
  }

  void conv_pipe_fwd_t::run_fwd( p_map_str_p_nda_float_t const & fwd ) {
    timer_t t("conv_pipe_fwd_t::run_fwd");
    printf("run_fwd() begin\n");
    copy_named_ndas_to_cups( cp->bots, *fwd, *cups ); // copy sources in
    printf("run_fwd() exec\n");
    for( vect_cu_func_call_t::const_iterator i = fwd_calls.begin(); i != fwd_calls.end(); ++i ) {
      cu_func_call_t const & cfc = *i;
      cu_func_t const & cf = must_find( cu_funcs, cfc.cu_func_name );
      assert( cf.arg_sizes.size() == cfc.args.size() );
      vect_rp_void cu_func_args;
      //printf( "cfc.cu_func_name=%s cfc.args=%s\n", str(cfc.cu_func_name).c_str(), str(cfc.args).c_str() );
      for( uint32_t i = 0; i != cfc.args.size(); ++i ) {
	p_cup_float arg = must_find( *cups, cfc.args[i] );
	//printf( "  cfc.args[i]=%s arg->sz=%s\n", str(cfc.args[i]).c_str(), str(arg->sz).c_str() );
	assert_st( arg->sz == cf.arg_sizes[i] );
	cu_func_args.push_back( &arg->p );
      }
      cu_err_chk( cuLaunchKernel( cf.cu_func,
				  cf.blks, 1, 1, // grid x,y,z dims
				  cf.tpb, 1, 1, // block x,y,z dims
				  0, 0, // smem_bytes, stream_ix
				  &cu_func_args[0], // cu_func's args
				  0 ), "cuLaunchKernel" ); // unused 'extra' arg-passing arg
      
    }
    cu_err_chk( cuCtxSynchronize(), "cuCtxSynchronize" );
    printf("run_fwd() copy out\n");
    cp->fwd_alloc_ndas( fwd, num_imgs, 1 ); // sinks_only=1
    copy_named_cups_to_ndas( cp->tops, *cups, *fwd ); // copy sinks out
    printf("run_fwd() done\n");
  }


  string gen_conv_op_one_img_conv( p_conv_op_t const & cop, p_map_str_p_nda_float_t const & fwd, uint32_t const img_ix, 
				 p_nda_float_t const & bot, p_nda_float_t const & top ) {
    
    string ret;
    ret += cu_base_decls;
    ret += R"rstr(
extern "C" {
  __global__ void conv_mumble( float const * const bot, float const * const filts, float const * const biases, float * const top ) {
    uint32_t const ix = blockDim.x * blockIdx.x + threadIdx.x;

}
)rstr";
    return ret;
#if 0
    
    u32_pt_t kern_sz = cop->kern_sz;
    if( kern_sz.is_zeros() ) { kern_sz = {bot->dims.dims(3), bot->dims.dims(2)}; } // 'global' input special case
    string const tag_id_str = as_pyid( cop->tag );    
    p_nda_float_t const & filts = must_find( *fwd, tag_id_str + "_filts" );
    p_nda_float_t const & biases = must_find( *fwd, tag_id_str + "_biases" );
    assert_st( filts->dims == dims_t(vect_uint32_t{top->dims.dims(1),bot->dims.dims(1),kern_sz.d[1],kern_sz.d[0] },1) );
    assert_st( biases->dims == dims_t(vect_uint32_t{top->dims.dims(1)},1) );
    assert_st( top->dims.dims(1) == cop->out_chans );

    for( uint32_t fix = 0; fix != filts->dims.dims(0); ++fix ) {
      for( uint32_t y = 0; y != top->dims.dims(2); ++y ) {
	for( uint32_t x = 0; x != top->dims.dims(3); ++x ) {
	  float out_pel = 0;
	  i32_pt_t in_ix = u32_to_i32( u32_pt_t{x,y}*cop->stride) - u32_to_i32(cop->in_pad.p[0]);
	  for( uint32_t in_chan = 0; in_chan != bot->dims.dims(1); ++in_chan ) {
	    for( uint32_t ky = 0; ky < kern_sz.d[1]; ++ky ) {
	      int32_t in_ky = in_ix.d[1] + ky;
	      if( (in_ky < 0) || (uint32_t(in_ky) >= bot->dims.dims(2)) ) { continue; }
	      for( uint32_t kx = 0; kx < kern_sz.d[0]; ++kx ) {
		int32_t in_kx = in_ix.d[0] + kx;
		if( (in_kx < 0) || (uint32_t(in_kx) >= bot->dims.dims(3)) ) { continue; }
		out_pel += bot->at4( img_ix, in_chan, in_ky, in_kx ) * filts->at4( fix, in_chan, ky, kx );
	      }
	    }
	  }
	  out_pel += biases->at1( fix );
	  top->at4( img_ix, fix, y, x ) = out_pel; // > 0 ? out_pel : 0;
	}
      }
    }
#endif
  }

  
#include"gen/nvrtc_util.cc.nesi_gen.cc"
}
