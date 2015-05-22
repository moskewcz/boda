// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"nvrtc_util.H"
#include"str_util.H"
#include"rand_util.H"
#include"has_main.H"
#include"timers.H"
#include<nvrtc.h>
#include<cuda.h>
#include<boost/filesystem.hpp>
#include<boost/lexical_cast.hpp>
#include"lexp.H"

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
  string nvrtc_compile( string const & cuda_prog_str ) {
    timer_t t("nvrtc_compile");
    p_nvrtcProgram cuda_prog = make_p_nvrtcProgram( cuda_prog_str );
    vect_string cc_opts = {"--use_fast_math","--gpu-architecture=compute_52","--restrict"};
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
      cu_copy_nda_to_cup( must_find( cups, pyid ), must_find( ndas, pyid ) );
    }
  }
  void copy_named_cups_to_ndas( vect_string const & names, map_str_p_cup_float_t const & cups, map_str_p_nda_float_t & ndas ) {
    for( vect_string::const_iterator i = names.begin(); i != names.end(); ++i ) {
      string const pyid = as_pyid( *i );
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
    cu_funcs_t::iterator gen_op_kern( p_conv_op_t const & cop, conv_io_t const & cio_in, conv_io_t const & cio_out );
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
  
  void insert_nda_exprs( vect_pair_str_str & mss, string const & ix, vect_string const & dns, vect_uint32_t const & dss ) {
    assert_st( dns.size() );
    assert_st( dns.size() == dss.size() );
    uint32_t stride = 1;
    for( int32_t i = dns.size()-1; i >= 0; --i ) {
      mss.push_back( make_pair( ix+"_"+dns[i]+"_dim", str(dss[i]) ) );
      assert_st( stride );
      mss.push_back( make_pair( ix+"_"+dns[i]+"_sz", str(stride) ) );
      string v = (stride > 1) ? "("+ix+"/"+str(stride)+")" : ix;
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

  cu_funcs_t::iterator conv_pipe_fwd_t::gen_op_kern( p_conv_op_t const & cop, conv_io_t const & cio_in, conv_io_t const & cio_out ) {
    // note: cio_in and cio_out are derived from cop->bots[0] and cop->tops[0]
    // for now, we only attempt to handle the (common) case of uniform padding, kernel size, and stride
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
    uint32_t const in_dim = cio_in.sz.d[0];

    bool const is_conv = cop->type == Convolution_str;
    bool const is_pool = cop->type == Pooling_str;
    string cu_func_name;
    p_string cu_func_template;
    if( is_conv ) {
      cu_func_name = strprintf( "conv__num_imgs_%s__in_pad_%s__in_dim_%s__in_chans_%s__kern_sz_%s__stride_%s__out_chans_%s", 
				str(num_imgs).c_str(), str(in_pad).c_str(), str(in_dim).c_str(), str(cio_in.chans).c_str(),
				str(kern_sz).c_str(), str(stride).c_str(), str(cio_out.chans).c_str() );
      cu_func_template = read_whole_fn( (path(py_boda_test_dir()) / "rtc" / "conv.cu").string() );
    } else if( is_pool ) {
      assert_st( cio_out.chans == cio_in.chans );
      cu_func_name = strprintf( "pool__num_imgs_%s__in_pad_%s__in_dim_%s__avg_pool_%s__kern_sz_%s__stride_%s__out_chans_%s", 
				str(num_imgs).c_str(), str(in_pad).c_str(), str(in_dim).c_str(), str(uint32_t(cop->avg_pool)).c_str(),
				str(kern_sz).c_str(), str(stride).c_str(), str(cio_out.chans).c_str() );
      cu_func_template = read_whole_fn( (path(py_boda_test_dir()) / "rtc" / "pool.cu").string() );
    } else { rt_err( "unhanded kern op: " + cop->type ); }
    
    std::pair< cu_funcs_t::iterator, bool > ins_ret = cu_funcs.insert( make_pair( cu_func_name, cu_func_t{} ) );
    if( !ins_ret.second ) { return ins_ret.first; } // already generated
    cu_func_t & cf = ins_ret.first->second;

    vect_pair_str_str tf_exprs;
    tf_exprs.push_back( make_pair( "cu_func_name", cu_func_name ) );
    tf_exprs.push_back( make_pair( "kern_sz", str(kern_sz) ) );
    tf_exprs.push_back( make_pair( "stride", str(stride) ) );

    uint32_t const t_tile_sz = 8;
    tf_exprs.push_back( make_pair( "t_tile_sz", str(t_tile_sz) ) );


    vect_string const cio_dims{"img","chan","y","x"};
    insert_nda_exprs( tf_exprs, "out_ix", cio_dims, vect_uint32_t{num_imgs,cio_out.chans,cio_out.sz.d[1],cio_out.sz.d[0]} );
    uint32_t const out_ix_sz = get_sz( tf_exprs, "out_ix" );
    insert_nda_exprs( tf_exprs, "in_ix", cio_dims, vect_uint32_t{num_imgs,cio_in.chans,cio_in.sz.d[1],cio_in.sz.d[0]} );
    if( is_conv ) {
      insert_nda_exprs( tf_exprs, "filts_ix", vect_string{"out_chan","in_chan","y","x"}, 
			vect_uint32_t{cio_out.chans,cio_in.chans,kern_sz,kern_sz} );
      // for reg blocking
      uint32_t const out_chan_tile_sz = u32_ceil_div( cio_out.chans, t_tile_sz );
      assert_st( out_chan_tile_sz * t_tile_sz == cio_out.chans ); // FIXME: too strong (need to handle partial tiles)
      uint32_t const patch_sz = u32_ceil_div( out_ix_sz, cio_out.chans );
      assert_st( patch_sz * cio_out.chans == out_ix_sz ); // by construction
      uint32_t const patch_tile_sz = u32_ceil_div( patch_sz, t_tile_sz );
      //assert_st( patch_tile_sz * t_tile_sz == patch_sz ); // FIXME: too strong (need to handle partial tiles)
      insert_nda_exprs( tf_exprs, "tile_ix", vect_string{"patch_tile","out_chan_tile"}, 
			vect_uint32_t{patch_tile_sz,out_chan_tile_sz} );
      insert_nda_exprs( tf_exprs, "patch_ix", vect_string{"img","y","x"}, vect_uint32_t{num_imgs,cio_out.sz.d[1],cio_out.sz.d[0]} );
      insert_nda_exprs( tf_exprs, "filts_ix_out_chan_elem", vect_string{"in_chan","y","x"}, 
			vect_uint32_t{cio_in.chans,kern_sz,kern_sz} );
    }
    string ops("// begin ops\n");
    if( is_conv ) {
#if 0      
      uint32_t filts_off = 0;
      for( uint32_t kc = 0; kc != cio_in.chans; ++kc ) {
	for( uint32_t ky = 0; ky != kern_sz; ++ky ) {
	  for( uint32_t kx = 0; kx != kern_sz; ++kx ) {
	    uint32_t const in_off = kc * cio_in.sz.dims_prod() + ky * cio_in.sz.d[0] + kx; // FIXME: get dims from tf_exprs?
	    ops += "  out_v += in[in_ix+"+str(in_off)+"] * filts[filts_ix+"+str(filts_off)+"];\n";
	    ++filts_off;
	  }
	}
      }
#else
      ops += "#error disabled\n";
#endif
    }
    if( is_pool ) {
      for( uint32_t ky = 0; ky != kern_sz; ++ky ) {
	for( uint32_t kx = 0; kx != kern_sz; ++kx ) {
	  uint32_t const in_off = ky * cio_in.sz.d[0] + kx; // FIXME: get dims from tf_exprs?
	  if( cop->avg_pool ) { ops += "  out_v += in[in_ix+"+str(in_off)+"];\n"; }
	  else { ops += "  out_v = max( out_v, in[in_ix+"+str(in_off)+"]);\n"; }
	}
      }
      if( cop->avg_pool ) { ops += "  out_v /= float("+str(kern_sz*kern_sz)+");\n"; }
    }
    ops += "  // end ops"; // note: newline (and semi-unwanted semi-colon) will go here from src
    tf_exprs.push_back( std::make_pair( "ops", ops ) );
    string t_tile_fmas("// begin t_tile_fmas\n");
    string t_tile_loads("// begin t_tile_loads\n");
    string t_tile_stores("// begin t_tile_stores\n");
    if( is_conv ) {
      t_tile_loads += "    uint32_t const filt_ix_base = "
	"((%(tile_ix_out_chan_tile)*%(t_tile_sz)))*%(filts_ix_out_chan_sz)+filts_ix_out_chan_elem;\n"; 
      for( uint32_t tx = 0; tx != t_tile_sz; ++tx ) {
	t_tile_loads += strprintf( "    filts_strip[%s] = filts[%s*%%(filts_ix_out_chan_sz) + filt_ix_base];\n",
				   str(tx).c_str(), str(tx).c_str() );
      }
      for( uint32_t ty = 0; ty != t_tile_sz; ++ty ) { // note: could merge with above loop, but we want to use ty for consistency
	t_tile_loads += strprintf("  { uint32_t const patch_ix = %%(tile_ix_patch_tile)*%%(t_tile_sz)+%s;\n", str(ty).c_str() );
	t_tile_loads += strprintf( "    in_strip[%s] = in[%%(patch_ix_img)*%%(in_ix_img_sz) + \n"
				   "    %%(filts_ix_out_chan_elem_in_chan)*%%(in_ix_chan_sz) + \n"
				   "    (%%(patch_ix_y)*%%(stride)+%%(filts_ix_out_chan_elem_y))*%%(in_ix_y_sz) + \n"
				   "    (%%(patch_ix_x)*%%(stride)+%%(filts_ix_out_chan_elem_x))*%%(in_ix_x_sz)]; }\n", 
				   str(ty).c_str() );
      }

      t_tile_stores += "  uint32_t const chan_ix = %(tile_ix_out_chan_tile)*%(t_tile_sz);\n";
      for( uint32_t ty = 0; ty != t_tile_sz; ++ty ) {
	t_tile_stores += strprintf("  {\n    uint32_t const patch_ix = %%(tile_ix_patch_tile)*%%(t_tile_sz)+%s;\n", str(ty).c_str() );
	t_tile_stores += "    if( patch_ix >= %(patch_ix_sz) ) { return; } // this and the following off-the-end patches\n";
	for( uint32_t tx = 0; tx != t_tile_sz; ++tx ) {
	  t_tile_fmas += strprintf( "    out_tile[%s] += filts_strip[%s]*in_strip[%s];\n", 
				    str((ty*t_tile_sz+tx)).c_str(), str(tx).c_str(), str(ty).c_str() );
	  //t_tile_stores += strprintf( "    { float const v = (tile_ix+1)*1000 + %s*10 + %s;\n", str(tx).c_str(), str(ty).c_str() );
	  //t_tile_stores += strprintf( "    { float const v = tile_ix;\n" );
	  t_tile_stores += strprintf( "    { float const v = out_tile[%s] + biases[chan_ix+%s];\n",  
				      str((ty*t_tile_sz+tx)).c_str(), str(tx).c_str() );
	  t_tile_stores += strprintf( "    out[ %%(patch_ix_img)*%%(out_ix_img_sz) + %%(patch_ix_y)*%%(out_ix_y_sz) + "
				      " %%(patch_ix_x)*%%(out_ix_x_sz) + (chan_ix+%s)*%%(out_ix_chan_sz)] = v; };\n ",
				      str(tx).c_str() );
	}
	t_tile_stores += "  }\n";
      }
    } 
    if( is_pool ) { } // unused for pooling
    // note: newline (and semi-unwanted semi-colon) from src will go after blocks, hence no newline on these lines
    t_tile_fmas += "    // end t_tile_fmas"; 
    t_tile_loads += "    // end t_tile_loads";
    t_tile_stores += "  // end t_tile_stores";
    tf_exprs.push_back( std::make_pair( "t_tile_fmas", t_tile_fmas ) );
    tf_exprs.push_back( std::make_pair( "t_tile_loads", t_tile_loads ) );
    tf_exprs.push_back( std::make_pair( "t_tile_stores", t_tile_stores ) );

    lexp_name_val_map_t tf_nvm{ p_lexp_t() };
    tf_nvm.insert_leafs_from( tf_exprs );

    string cu_func_str;
    str_format_from_nvm( cu_func_str, *cu_func_template, tf_nvm );
    cu_prog_str += cu_func_str;

    cu_prog_str += "// -- template substituion table used: --\n";
    for( vect_pair_str_str::const_iterator i = tf_exprs.begin(); i != tf_exprs.end(); ++i ) {
      cu_prog_str += strprintf( "/* %s = %s */\n", str(i->first).c_str(), str(i->second).c_str() );
    } 

    // for error checking, (re-) calculate the sizes of the arguments (note: in elements, not bytes)
    if( is_conv ) { 
      cf.arg_sizes.push_back( get_sz( tf_exprs, "filts_ix" ) );
      cf.arg_sizes.push_back( cio_out.chans ); // biases_sz
    }
    cf.arg_sizes.push_back( get_sz( tf_exprs, "in_ix" ) );
    cf.arg_sizes.push_back( out_ix_sz );

    cf.tpb = 256;
    if( is_conv ) { cf.blks = u32_ceil_div( u32_ceil_div( out_ix_sz, t_tile_sz*t_tile_sz ), cf.tpb ); }
    else if( is_pool ) { cf.blks = u32_ceil_div( out_ix_sz, cf.tpb ); }
    else { assert_st( 0 ); }
    printf( "cu_func_name=%s cf.tpb=%s cf.blks=%s\n", str(cu_func_name).c_str(), str(cf.tpb).c_str(), str(cf.blks).c_str() );
    return ins_ret.first;
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
    assert_st( cop->bots.size() == 1 );
    conv_io_t & cio_in = cp->must_get_node( cop->bots[0] )->cio;
    assert_st( cop->tops.size() == 1 );
    conv_io_t & cio_out = cp->must_get_node( cop->tops[0] )->cio;
    bool const is_conv = cop->type == Convolution_str;
    if( is_conv || (cop->type == Pooling_str) ) {
      vect_string arg_ids;
      string const filts_id = tag_id_str + "_filts";
      string const biases_id = tag_id_str + "_biases";
      if( is_conv ) {
	arg_ids.push_back( filts_id );
	arg_ids.push_back( biases_id );
      }
      arg_ids.push_back( as_pyid(cop->bots[0]) );
      arg_ids.push_back( as_pyid(cop->tops[0]) );
      cu_funcs_t::iterator cfi = gen_op_kern( cop, cio_in, cio_out );
      fwd_calls.push_back( cu_func_call_t{ cfi->first, arg_ids } );
      if( is_conv ) {
	assert_st( cio_out.chans == cop->out_chans );
	vect_uint32_t const & arg_sizes = cfi->second.arg_sizes;
	assert_st( arg_sizes.size() == 4 );
	add_op_param( filts_id, arg_sizes[0] );
	add_op_param( biases_id, arg_sizes[1] );
      }

    } else if( cop->type == ReLU_str ) {
      // check that this is a single in-out in-place operation
      assert_st( cop->bots[0] == cop->tops[0] );
      fwd_calls.push_back( cu_func_call_t{ gen_op_relu( cio_out ), { as_pyid(cop->tops[0]) } } );
    } else if( cop->type == Dropout_str ) {
      // check that this is a single in-out in-place operation
      assert_st( cop->bots[0] == cop->tops[0] );
      // ignore for fwd
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
	gen_node( *j, cp->must_get_node(*j) );
	gen_ops_rec( *j ); 
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

    write_whole_fn( "out.cu", cu_prog_str );
    string const prog_ptx = nvrtc_compile( cu_prog_str );
    write_whole_fn( "out.ptx", prog_ptx );
    //printf( "cu_prog_str=%s\n", str(cu_prog_str).c_str() );
    //printf( "prog_ptx=%s\n", str(prog_ptx).c_str() );
    cu_err_chk( cuInit( 0 ), "cuInit" );
    cu_err_chk( cuDeviceGet( &cu_dev, 0 ), "cuDeviceGet" );
    //cu_err_chk( cuCtxCreate( &cu_context, 0, cu_dev ), "cuCtxCreate" );
    cu_err_chk( cuDevicePrimaryCtxRetain( &cu_context, cu_dev ), "cuDevicePrimaryCtxRetain" );
    cu_err_chk( cuModuleLoadDataEx( &cu_mod, &prog_ptx[0], 0, 0, 0 ), "cuModuleLoadDataEx" );
    for( cu_funcs_t::iterator i = cu_funcs.begin(); i != cu_funcs.end(); ++i ) {
      cu_err_chk( cuModuleGetFunction( &i->second.cu_func, cu_mod, i->first.c_str() ), "cuModuleGetFunction" );
    }

    copy_named_ndas_to_cups( op_param_names, *cp->op_params, *cups ); // copy op_params in  
  }

  void conv_pipe_fwd_t::run_fwd( p_map_str_p_nda_float_t const & fwd ) {
    timer_t t("conv_pipe_fwd_t::run_fwd");
    //printf("run_fwd() begin\n");
    copy_named_ndas_to_cups( cp->bots, *fwd, *cups ); // copy sources in
    //printf("run_fwd() exec\n");
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
    //printf("run_fwd() copy out\n");
    cp->fwd_alloc_ndas( fwd, num_imgs, 1 ); // sinks_only=1
    copy_named_cups_to_ndas( cp->tops, *cups, *fwd ); // copy sinks out
    //printf("run_fwd() done\n");
  }
  
#include"gen/nvrtc_util.cc.nesi_gen.cc"
}
