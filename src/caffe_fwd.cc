// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"has_conv_fwd.H"
#include"timers.H"
#include"conv_util.H"
#include"caffe/caffe.hpp"
#include<cudaProfiler.h>
#include<dlfcn.h>

namespace caffe { 
  template< typename T > struct Net; 
}

namespace boda 
{
  using caffe::Caffe;
  using caffe::Blob;

  typedef caffe::Net< float > Net_float;
  typedef shared_ptr< Net_float > p_Net_float;

  void dims_t_to_shape( dims_t const & dims, caffe::BlobShape & bs ); // from caffepb.cc

  typedef void caffe_set_det_drop_seed_t( uint32_t const det_drop_seed_ );

  void boda_stub_caffe_set_det_drop_seed( uint32_t const det_drop_seed_ ) {
    static bool caffe_set_det_drop_seed_set = 0;
    static caffe_set_det_drop_seed_t * caffe_set_det_drop_seed = 0;
    if( !caffe_set_det_drop_seed_set ) {
      caffe_set_det_drop_seed = (caffe_set_det_drop_seed_t *)dlsym( RTLD_DEFAULT, "caffe_set_det_drop_seed" );
      caffe_set_det_drop_seed_set = 1;
    }
    if( caffe_set_det_drop_seed ) { caffe_set_det_drop_seed(det_drop_seed_); }
    else {
      printf("Warning: caffe not compiled with caffe_set_det_drop_seed() support; determinisic Dropout not availible. Expect related test failures.\n");
    }
  }

  struct caffe_fwd_t : virtual public nesi, public has_conv_fwd_t // NESI(help="compute conv pipe forward using caffe",
			   // bases=["has_conv_fwd_t"], type_id="caffe" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t gpu_id; //NESI(default=0,help="id of gpu to use, passed to caffe::SetDevice() and in turn to CUDA.")
    uint32_t enable_prof; //NESI(default=0,help="if 1, enable profiling of caffe forward calls")

    p_conv_pipe_t cp;

    p_Net_float net;

    virtual void init( p_conv_pipe_t const & cp_ );
    virtual void run_fwd( vect_string const & to_set_vns, p_map_str_p_nda_float_t const & fwd, vect_string const & to_get_vns );
    virtual string get_info_log( void ) { return string(); }
    virtual void set_det_drop_seed( uint32_t const & det_drop_seed_ ) { boda_stub_caffe_set_det_drop_seed( det_drop_seed_ ); }
  };

  void init_caffe( uint32_t const gpu_id ) {
    static bool caffe_is_init = 0;
    if( !caffe_is_init ) {
      caffe_is_init = 1;
      timer_t t("caffe_init");
      google::InitGoogleLogging("boda_caffe");
      Caffe::set_mode(Caffe::GPU);
      Caffe::SetDevice(gpu_id);
    }
  }

  void copy_nda_to_caffe_blob( p_nda_float_t const & nda, Blob<float> * const blob, bool const set_diff ) {
    dims_t const & nda_dims = nda->dims;
    assert_st( nda_dims.sz() == uint32_t(blob->num_axes()) );
    for( uint32_t i = 0; i != nda_dims.sz(); ++i ) { assert_st( nda_dims.dims(i) == uint32_t(blob->shape(i)) ); }
    assert_st( nda->elems.sz == uint32_t(blob->count()) );
    float * const src = &nda->elems[0];
    switch (Caffe::mode()) {
    case Caffe::CPU: memcpy( set_diff ? blob->mutable_cpu_diff() : blob->mutable_cpu_data(), src, sizeof(float) * blob->count()); break;
    case Caffe::GPU: cudaMemcpy( set_diff ? blob->mutable_gpu_diff() : blob->mutable_gpu_data(), src, sizeof(float) * blob->count(), cudaMemcpyHostToDevice); break;
    default: LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  void set_layer_blobs( p_Net_float net_, string const & layer_name, vect_p_nda_float_t & blobs ) {
    timer_t t("caffe_set_layer_blob_data");
    shared_ptr< caffe::Layer<float> > layer = net_->layer_by_name( layer_name );
    if( !layer ) { rt_err( strprintf("setting parameters: layer '%s' not found in network\n",str(layer_name).c_str() )); }    
    const vector< shared_ptr< caffe::Blob<float> > >& layer_blobs = layer->blobs();
    assert_st( blobs.size() == layer_blobs.size() );
    for( uint32_t bix = 0; bix < layer_blobs.size(); ++bix ) {
      p_nda_float_t const & nda = blobs[bix];
      Blob<float> * const layer_blob = layer_blobs[bix].get();
      copy_nda_to_caffe_blob( nda, layer_blob, 0 ); 
    }  // switch (Caffe::mode())
  }

  void copy_caffe_blob_to_nda( Blob<float> * const blob, bool const get_diff, p_nda_float_t const & nda ) {
    dims_t const & nda_dims = nda->dims;
    //printf( "nda_dims=%s blob->shape->str()=%s\n", str(nda_dims).c_str(), str(blob->shape_string()).c_str() );
    assert_st( nda_dims.sz() == uint32_t(blob->num_axes()) );
    for( uint32_t i = 0; i != nda_dims.sz(); ++i ) { assert_st( nda_dims.dims(i) == uint32_t(blob->shape(i)) ); }
    assert_st( nda->elems.sz == uint32_t(blob->count()) );
    float * const dest = &nda->elems[0];
    switch (Caffe::mode()) {
    case Caffe::CPU: memcpy(dest, get_diff ? blob->cpu_diff() : blob->cpu_data(), sizeof(float) * blob->count() ); break;
    case Caffe::GPU: cudaMemcpy(dest, get_diff ? blob->gpu_diff() : blob->gpu_data(), sizeof(float) * blob->count(), 
				cudaMemcpyDeviceToHost); break;
    default: LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  void get_layer_blob( p_Net_float net_, string const & layer_name, uint32_t const bix, bool const get_diff, p_nda_float_t const & blob ) {
    timer_t t("caffe_get_layer_blob_data");
    shared_ptr< caffe::Layer<float> > layer = net_->layer_by_name( layer_name );
    if( !layer ) { rt_err( strprintf("gettting parameters: layer '%s' not found in network\n",str(layer_name).c_str() )); }    
    const vector< shared_ptr< caffe::Blob<float> > >& layer_blobs = layer->blobs();
    assert_st( bix < layer_blobs.size() );
    Blob<float> * const layer_blob = layer_blobs[bix].get();
    copy_caffe_blob_to_nda( layer_blob, get_diff, blob );
  }

  p_Net_float caffe_create_net( p_conv_pipe_t const & cp ) {
    timer_t t("caffe_create_net");

    p_net_param_t net_param( new net_param_t ( *cp->as_net_param() ) ); // start with copy of original (unmodified by boda) input net_param
    *net_param->mutable_state() = *cp->net_state; // set state as set/used by boda's param->pipe conversion
    // remove all layers that don't have matching ops in the pipe. in particular, this will remove data and accuracy layers, as well
    // as some unhandled layers (i.e. currently dropout). in general, this isn't particulaly sensible/correct, but it does what we want for now.
    int o = 0;
    for( int i = 0; i < net_param->layer_size(); i++ ) {
      caffe::LayerParameter const * const lp = &net_param->layer(i);
      if( !has( *cp->convs, lp->name() ) ) { continue; }
      caffe::LayerParameter * const olp = net_param->mutable_layer(o);
      if( i != o ) { *olp = net_param->layer(i); } ++o; // keep layer
    }
    while( net_param->layer_size() > o ) { net_param->mutable_layer()->RemoveLast(); }

    // FIXME: perhaps we should process all of cp->bots here, and detect which nodes are filts/biases, and special-case set them
    // using something similar to set_layer_blobs()?
    // add input blobs for conv_pipe inputs (which in turn were derived from original
    // param data layers)
    assert_st( net_param->input_dim_size() == 0 ); // should be no input blobs to start (only train_val format is supported as input)
    vect_string caffe_bots = cp->data_img_node_names;
    caffe_bots.insert( caffe_bots.end(), cp->data_label_node_names.begin(), cp->data_label_node_names.end() );
    for( vect_string::const_iterator i = caffe_bots.begin(); i != caffe_bots.end(); ++i ) { 
      dims_t & dims = cp->must_get_node( *i )->dims;
      assert_st( !dims.empty() ); // all bot sizes (and further-but-unchecked-here, all nodes) should be set
      net_param->add_input(*i);
      dims_t_to_shape( dims, *net_param->add_input_shape() );      
    }
    p_Net_float net( new Net_float( *net_param ) );
    //net->CopyTrainedLayersFrom( trained_fn );
    for( map_str_p_vect_p_nda_float_t::const_iterator i = cp->layer_blobs->begin(); i != cp->layer_blobs->end(); ++i ) { 
      //printf( "i->first=%s i->second->size()=%s\n", str(i->first).c_str(), str(i->second->size()).c_str() );
      set_layer_blobs( net, i->first, *i->second );  
    }

    // for timing, do one (garbage) forward. presumably forces memory allocs and/or some other lazy
    // setup to happen here
    net->ForwardPrefilled();
    if( cp->has_bck_ops.v ) { net->Backward(); }
    cudaDeviceSynchronize();
    return net;
  }

  void caffe_fwd_t::init( p_conv_pipe_t const & cp_ ) {
    cp = cp_;
    assert_st( cp );
    init_caffe( gpu_id ); // FIXME/note: only does something on first call
    net = caffe_create_net( cp );      
  }

  void raw_do_forward( p_Net_float net, bool const enable_prof, bool const do_bck ) {
    if( enable_prof ) { cuProfilerStart(); }
    if( do_bck ) { net->ClearParamDiffs(); }
    net->ForwardPrefilled();
    if( do_bck ) { net->Backward(); }
    if( enable_prof ) { cuProfilerStop(); }
  }

  void copy_output_blob_data( p_Net_float net, string const & out_node_name, bool const & get_diff, p_nda_float_t const & out_nda ) {
    timer_t t("caffe_copy_output_blob_data");
    shared_ptr< Blob<float> > output_blob;
    string layer_name = out_node_name;
    if( maybe_strip_suffix( layer_name, "_filts" ) ) { get_layer_blob( net, layer_name, 0, get_diff, out_nda ); }
    else if( maybe_strip_suffix( layer_name, "_biases" ) ) { get_layer_blob( net, layer_name, 1, get_diff, out_nda ); }
    else {
      output_blob = net->blob_by_name( out_node_name );
      if( !output_blob ) { rt_err( strprintf("gettting output: node '%s' not found in network as regular blob (didn't end in _filts or _biases) "
					     "(note: get_diff=%s).\n", str(out_node_name).c_str(), str(get_diff).c_str() )); }
      copy_caffe_blob_to_nda( output_blob.get(), get_diff, out_nda );
    }
  }

  void caffe_fwd_t::run_fwd( vect_string const & to_set_vns, p_map_str_p_nda_float_t const & fwd, vect_string const & to_get_vns ) {
    assert_st( net );
    cudaDeviceSynchronize();
    {
      timer_t t("caffe_fwd_t::set_vars");
      for( vect_string::const_iterator i = to_set_vns.begin(); i != to_set_vns.end(); ++i ) {
	shared_ptr< caffe::Blob<float> > const & ib = net->blob_by_name( *i );
	if( !ib ) { rt_err( strprintf("gettting caffe blob for setting inputs: node '%s' from to_set_vns not found in network (note: do_bck=%s).\n",
				      (*i).c_str(), str(cp->has_bck_ops.v).c_str() )); }
	p_nda_float_t const & ib_nda = must_find( *fwd, *i );
	copy_nda_to_caffe_blob( ib_nda, ib.get(), 0 );
      }
      cudaDeviceSynchronize();
    }
    {
      timer_t t("caffe_fwd_t::run_fwd");
      raw_do_forward( net, enable_prof, cp->has_bck_ops.v );
      cudaDeviceSynchronize();
    }
    {
      timer_t t("caffe_fwd_t::get_vars");
      for( vect_string::const_iterator i = to_get_vns.begin(); i != to_get_vns.end(); ++i ) {
	string const out_node_name = *i;
	dims_t const & out_node_dims = cp->must_get_node( out_node_name )->dims;
	p_nda_float_t & out_nda = (*fwd)[*i];
	if( out_nda ) { assert_st( out_nda->dims == out_node_dims ); }
	else{ out_nda.reset( new nda_float_t( out_node_dims ) ); }
	string caffe_node_name = out_node_name;
	bool get_diff = maybe_strip_suffix( caffe_node_name, "_grad_loss" );
	copy_output_blob_data( net, caffe_node_name, get_diff, out_nda );
	//must_insert( *fwd, out_node_name, out_nda );
      }
      cudaDeviceSynchronize();
    }
  }  


#include"gen/caffe_fwd.cc.nesi_gen.cc"
}
