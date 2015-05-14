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

  p_cup_float get_cup_copy( vect_float const & v ) { 
    p_cup_float ret = make_shared<cup_float>( v.size() ); 
    cu_err_chk( cuMemcpyHtoD( ret->p, &v[0], v.size()*sizeof(vect_float::value_type) ), "cuMemcpyHtoD" );
    return ret;
  }
  void set_from_cup( vect_float & v, p_cup_float const & cup ) {
    assert_st( cup->sz == v.size() );
    cu_err_chk( cuMemcpyDtoH( &v[0], cup->p, v.size()*sizeof(vect_float::value_type) ), "cuMemcpyHtoD" );
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

  struct conv_pipe_fwd_t {
    p_conv_pipe_t cp;
    void gen_ops_rec( string const & node_name );
    void init( p_conv_pipe_t const & cp_ );
    void run_fwd( p_map_str_p_nda_float_t const & fwd );
  };
  p_conv_pipe_fwd_t make_conv_pipe_fwd_t( p_conv_pipe_t const & cp ) { 
    p_conv_pipe_fwd_t ret = make_shared<conv_pipe_fwd_t>(); ret->init(cp); return ret; 
  }
  void conv_pipe_fwd_t_run( p_conv_pipe_fwd_t const & cpf, p_map_str_p_nda_float_t const & fwd ) { cpf->run_fwd( fwd ); }

  void conv_pipe_fwd_t::gen_ops_rec( string const & node_name ) {
    p_conv_node_t node = cp->must_get_node( node_name );
#if 0
    // print source nodes here, otherwise print with thier writing op
    if( node->top_for.empty() ) { print_blob_decl( out, node_name, node ); }
    else { assert( node->top_for.size() == 1 ); } // multiple writers not handled
    // print in-place ops for this node
    for( vect_p_conv_op_t::const_iterator j = node->in_place_ops.begin(); j != node->in_place_ops.end(); ++j ) {
      p_conv_op_t const & ip_cop = *j;
      out << strprintf( "%s(name=\"%s\",in_place=[%s])\n", ip_cop->type.c_str(), as_pyid(ip_cop->tag).c_str(), as_pyid(node->name).c_str() );
    }
#endif
    for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) {
      p_conv_op_t const & cop = cp->get_op( *i );
      if( !cop->on_seen_bot() ) { continue; } // wait till we've seen all bottoms
      //print_op_decl( out, this, cop, expand_ops );
      for( vect_string::const_iterator j = cop->tops.begin(); j != cop->tops.end(); ++j ) { gen_ops_rec( *i ); }
    }
  }
  void conv_pipe_fwd_t::init( p_conv_pipe_t const & cp_ ) {
    cp = cp_;
    assert_st( cp );
    assert_st( cp->finalized );
    cp->topo_visit_setup();
    for( vect_string::const_iterator i = cp->bots.begin(); i != cp->bots.end(); ++i ) { gen_ops_rec( *i ); }
  }
  void conv_pipe_fwd_t::run_fwd( p_map_str_p_nda_float_t const & fwd ) {
  }


  string cu_base_decls = R"rstr(
typedef unsigned uint32_t;
)rstr";

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
