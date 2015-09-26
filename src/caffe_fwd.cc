// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"str_util.H"
#include"has_conv_fwd.H"
#include"timers.H"
#include"conv_util.H"
#include"caffe/caffe.hpp"
#include<cudaProfiler.h>

namespace caffe { template< typename T > struct Net; }

namespace boda 
{
  using caffe::Caffe;
  using caffe::Blob;

  typedef caffe::Net< float > Net_float;
  typedef shared_ptr< Net_float > p_Net_float;

  struct caffe_fwd_t : virtual public nesi, public has_conv_fwd_t // NESI(help="compute conv pipe forward using caffe",
			   // bases=["has_conv_fwd_t"], type_id="caffe" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t gpu_id; //NESI(default=0,help="id of gpu to use, passed to caffe::SetDevice() and in turn to CUDA.")
    uint32_t enable_prof; //NESI(default=0,help="if 1, enable profiling of caffe forward calls")

    p_conv_pipe_t cp;
    uint32_t num_imgs;

    p_Net_float net;

    virtual void init( p_conv_pipe_t const & cp_, uint32_t const & num_imgs_ );
    virtual void run_fwd( p_map_str_p_nda_float_t const & fwd );
  };


  void init_caffe( uint32_t const gpu_id ) {
    static bool caffe_is_init = 0;
    if( !caffe_is_init ) {
      caffe_is_init = 1;
      timer_t t("caffe_init");
      google::InitGoogleLogging("boda_caffe");
      Caffe::set_mode(Caffe::GPU);
      Caffe::SetDevice(gpu_id);
      //Caffe::set_mode(Caffe::CPU);
    }
  }

  void set_layer_blobs( p_Net_float net_, string const & layer_name, vect_p_nda_float_t & blobs ) {
    timer_t t("caffe_set_layer_blob_data");
    shared_ptr< caffe::Layer<float> > layer = net_->layer_by_name( layer_name );
    if( !layer ) { rt_err( strprintf("setting parameters: layer '%s' not found in network\n",str(layer_name).c_str() )); }    
    const vector< shared_ptr< caffe::Blob<float> > >& layer_blobs = layer->blobs();
    assert_st( blobs.size() == layer_blobs.size() );
    for( uint32_t bix = 0; bix < layer_blobs.size(); ++bix ) {
      p_nda_float_t const & blob = blobs[bix];
      Blob<float> * const layer_blob = layer_blobs[bix].get();
      dims_t const & blob_dims = blob->dims;
      assert_st( blob_dims.sz() == uint32_t(layer_blob->num_axes()) );
      for( uint32_t i = 0; i != blob_dims.sz(); ++i ) { assert_st( blob_dims.dims(i) == uint32_t(layer_blob->shape(i)) ); }
      assert_st( blob->elems.sz == uint32_t(layer_blob->count()) );

      const float * const data_ptr = &blob->elems[0];
      switch (Caffe::mode()) {
      case Caffe::CPU: memcpy( layer_blob->mutable_cpu_data(), data_ptr, sizeof(float) * layer_blob->count()); break;
      case Caffe::GPU: cudaMemcpy( layer_blob->mutable_gpu_data(), data_ptr,
				   sizeof(float) * layer_blob->count(), cudaMemcpyHostToDevice); break;
      default: rt_err( "Unknown Caffe mode." );
      }  // switch (Caffe::mode())
    }
  }

  p_Net_float caffe_create_net( p_conv_pipe_t const & cp ) {
    timer_t t("caffe_create_net");

    p_net_param_t net_param = cp->as_net_param();
    // for now, we mimic the behavior of the caffe Net ctor that takes
    // a phase and 'force' the phase in the passed net. hey, it is
    // passed by non-const ref, right?
    net_param->mutable_state()->set_phase(caffe::TEST);
    p_Net_float net( new Net_float( *net_param ) );

    //net->CopyTrainedLayersFrom( trained_fn );
    for( map_str_p_vect_p_nda_float_t::const_iterator i = cp->layer_blobs->begin(); i != cp->layer_blobs->end(); ++i ) { 
      //printf( "i->first=%s i->second->size()=%s\n", str(i->first).c_str(), str(i->second->size()).c_str() );
      set_layer_blobs( net, i->first, *i->second );  
    }

    // for timing, do one (garbage) forward. presumably forces memory allocs and/or some other lazy
    // setup to happen here
    net->ForwardPrefilled();
    cudaDeviceSynchronize();
    return net;
  }

  void caffe_fwd_t::init( p_conv_pipe_t const & cp_, uint32_t const & num_imgs_ ) {
    num_imgs = num_imgs_;
    assert_st( num_imgs );
    cp = cp_;
    assert_st( cp );
    init_caffe( gpu_id ); // FIXME/note: only does something on first call
    net = caffe_create_net( cp );      
  }

  void raw_do_forward( p_Net_float net, p_map_str_p_nda_float_t const & fwd, bool const enable_prof ) {
    vector<int> const & ibixs = net->input_blob_indices();
    //vector<caffe::Blob<float>*> const & input_blobs = net->input_blobs();
    //assert_st( bottom.size() == input_blobs.size() );
    for (unsigned int i = 0; i < ibixs.size(); ++i) {
      string const & ib_name = net->blob_names()[ibixs[i]];
      shared_ptr< caffe::Blob<float> > const & ib = net->blob_by_name( ib_name );
      p_nda_float_t const & ib_nda = must_find( *fwd, ib_name );
      assert_st( ib_nda->elems.sz == uint32_t(ib->count()) );
      const float* const data_ptr = &ib_nda->elems[0];
      switch ( Caffe::mode() ) {
        case Caffe::CPU:	memcpy(ib->mutable_cpu_data(), data_ptr, sizeof(float) * ib->count()); break;
        case Caffe::GPU:	cudaMemcpy(ib->mutable_gpu_data(), data_ptr, sizeof(float) * ib->count(), cudaMemcpyHostToDevice); break;
        default:	rt_err( "Unknown Caffe mode." );
      }  // switch (Caffe::mode())
    }
    //const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
    if( enable_prof ) { cuProfilerStart(); }
    net->ForwardPrefilled();
    if( enable_prof ) { cuProfilerStop(); }
  }


  p_nda_float_t copy_output_blob_data( p_Net_float net, string const & out_node_name ) {
    timer_t t("caffe_copy_output_blob_data");
    shared_ptr< Blob<float> > output_blob = net->blob_by_name( out_node_name );
    assert_st( output_blob );

    dims_t out_batch_dims( 4 );
    out_batch_dims.dims(3) = output_blob->width();
    out_batch_dims.dims(2) = output_blob->height();
    out_batch_dims.dims(1) = output_blob->channels();
    out_batch_dims.dims(0) = output_blob->num();
    p_nda_float_t out_batch( new nda_float_t( out_batch_dims ) );
    assert_st( out_batch->elems.sz == uint32_t(output_blob->count()) );
      
    float * const dest = &out_batch->elems[0];
    switch (Caffe::mode()) {
    case Caffe::CPU: memcpy(dest, output_blob->cpu_data(), sizeof(float) * output_blob->count() ); break;
    case Caffe::GPU: cudaMemcpy(dest, output_blob->gpu_data(), sizeof(float) * output_blob->count(), 
				cudaMemcpyDeviceToHost); break;
    default: LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
    return out_batch;
  }

  void caffe_fwd_t::run_fwd( p_map_str_p_nda_float_t const & fwd ) {
    timer_t t("caffe_fwd_t::run_fwd");
    assert_st( net );
    raw_do_forward( net, fwd, enable_prof );
    string const & out_node_name = cp->get_single_top_node()->name;
    p_nda_float_t out = copy_output_blob_data( net, out_node_name );
    must_insert( *fwd, out_node_name, out );
  }  


#include"gen/caffe_fwd.cc.nesi_gen.cc"
}
