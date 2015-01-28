// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"timers.H"
#include"str_util.H"
#include"img_io.H"
#include"lexp.H"
#include"conv_util.H"
#include"caffeif.H"
#include<glog/logging.h>
#include<google/protobuf/text_format.h>
#include"caffe/caffe.hpp"
#include"anno_util.H"

namespace boda 
{
  using caffe::Caffe;
  using caffe::Blob;

  std::ostream & operator <<(std::ostream & os, scale_info_t const & v ) { 
    return os << strprintf( "from_upsamp_net=%s bix=%s feat_box=%s feat_img_box=%s\n", 
			    str(v.from_upsamp_net).c_str(), str(v.bix).c_str(), str(v.feat_box).c_str(), 
			    str(v.feat_img_box).c_str() ); 
  }
  
  void subtract_mean_and_copy_img_to_batch( p_nda_float_t const & in_batch, uint32_t img_ix, p_img_t const & img ) {
    timer_t t("subtract_mean_and_copy_img_to_batch");
    dims_t const & ibd = in_batch->dims;
    assert_st( img_ix < ibd.dims(0) );
    assert_st( 3 == ibd.dims(1) );
    assert_st( img->sz.d[0] == ibd.dims(3) );
    assert_st( img->sz.d[1] == ibd.dims(2) );
    uint32_t const inmc = 123U+(117U<<8)+(104U<<16)+(255U<<24); // RGBA
#pragma omp parallel for	  
    for( uint32_t y = 0; y < ibd.dims(2); ++y ) {
      for( uint32_t x = 0; x < ibd.dims(3); ++x ) {
	uint32_t const pel = img->get_pel({x,y});
	for( uint32_t c = 0; c < 3; ++c ) {
	  // note: RGB -> BGR swap via the '2-c' below
	  in_batch->at4( img_ix, 2-c, y, x ) = get_chan(c,pel) - float(uint8_t(inmc >> (c*8)));
	}
      }
    }
  }

  void chans_to_area( uint32_t & out_s, u32_pt_t & out_sz, u32_pt_t const & in_sz, uint32_t in_chan ) {
    out_s = u32_ceil_sqrt( in_chan );
    out_sz = in_sz.scale( out_s );
  }

  void copy_batch_to_img( p_nda_float_t const & out_batch, uint32_t img_ix, p_img_t const & img, u32_box_t region ) {
    if( region == u32_box_t{} ) { region.p[1] = u32_pt_t{out_batch->dims.dims(3),out_batch->dims.dims(2)}; }    
    // set up dim iterators that span only the image we want to process
    dims_t img_e( out_batch->dims.sz() );
    dims_t img_b( img_e.sz() );
    img_b.dims(0) = img_ix;
    img_e.dims(0) = img_ix + 1;
    img_b.dims(1) = 0;
    img_e.dims(1) = out_batch->dims.dims(1);
    img_b.dims(2) = region.p[0].d[1];
    img_e.dims(2) = region.p[1].d[1];
    img_b.dims(3) = region.p[0].d[0];
    img_e.dims(3) = region.p[1].d[0];
    float const out_max = nda_reduce( *out_batch, max_functor<float>(), 0.0f, img_b, img_e ); // note clamp to 0
    //float const out_min = nda_reduce( *out_batch, min_functor<float>(), 0.0f, img_b, img_e ); // note clamp to 0
    //assert_st( out_min == 0.0f ); // shouldn't be any negative values
    //float const out_rng = out_max - out_min;
    copy_batch_to_img( out_batch, img_ix, img, region, out_max );
  }
  void copy_batch_to_img( p_nda_float_t const & out_batch, uint32_t img_ix, p_img_t const & img, u32_box_t region, 
			  float const & out_max ) {
    if( region == u32_box_t{} ) { region.p[1] = u32_pt_t{out_batch->dims.dims(3),out_batch->dims.dims(2)}; }    
    dims_t const & obd = out_batch->dims;
    assert( obd.sz() == 4 );
    assert_st( img_ix < obd.dims(0) );

    u32_pt_t const out_sz( obd.dims(3), obd.dims(2) );
    uint32_t sqrt_out_chan;
    u32_pt_t img_sz;
    chans_to_area( sqrt_out_chan, img_sz, out_sz, obd.dims(1) );
    assert( sqrt_out_chan );
    assert( (sqrt_out_chan*sqrt_out_chan) >= obd.dims(1) );

    assert_st( img->sz == img_sz );

    for( uint32_t by = region.p[0].d[1]; by < region.p[1].d[1]; ++by ) {
      for( uint32_t bx = region.p[0].d[0]; bx < region.p[1].d[0]; ++bx ) {
	for( uint32_t bc = 0; bc < obd.dims(1); ++bc ) {
	  uint32_t const x = bx*sqrt_out_chan + (bc%sqrt_out_chan);
	  uint32_t const y = by*sqrt_out_chan + (bc/sqrt_out_chan);
	  //float const norm_val = ((out_batch->at4(img_ix,bc,by,bx)-out_min) / out_rng );
	  float const norm_val = ((out_batch->at4(img_ix,bc,by,bx)) / out_max );
	  uint32_t const gv = grey_to_pel( uint8_t( std::min( 255.0, 255.0 * norm_val ) ) );
	  //gv =grey_to_pel( uint8_t( std::min( 255.0, 255.0 * (log(.01) - log(std::max(.01f,norm_val))) / (-log(.01)) )));
	  img->set_pel( {x, y}, gv );
	}
      }
    }
  }

  void init_caffe( uint32_t const gpu_id ) {
    static bool caffe_is_init = 0;
    if( !caffe_is_init ) {
      caffe_is_init = 1;
      timer_t t("caffe_init");
      google::InitGoogleLogging("boda_caffe");
      Caffe::set_phase(Caffe::TEST);
      Caffe::set_mode(Caffe::GPU);
      Caffe::SetDevice(gpu_id);
      //Caffe::set_mode(Caffe::CPU);
    }
  }

  p_Net_float caffe_create_net( caffe::NetParameter & net_param, string const & trained_fn ) {
    p_Net_float net( new Net_float( net_param ) );
    net->CopyTrainedLayersFrom( trained_fn );
    return net;
  }

  void raw_do_forward( p_Net_float net_, vect_p_nda_float_t const & bottom ) {
    timer_t t("caffe_forward");
    vector<caffe::Blob<float>*>& input_blobs = net_->input_blobs();
    assert_st( bottom.size() == input_blobs.size() );
    for (unsigned int i = 0; i < input_blobs.size(); ++i) {
      assert_st( bottom[i]->elems.sz == uint32_t(input_blobs[i]->count()) );
      const float* const data_ptr = &bottom[i]->elems[0];
      switch ( Caffe::mode() ) {
      case Caffe::CPU:
	memcpy(input_blobs[i]->mutable_cpu_data(), data_ptr,
	       sizeof(float) * input_blobs[i]->count());
	break;
      case Caffe::GPU:
	cudaMemcpy(input_blobs[i]->mutable_gpu_data(), data_ptr,
		   sizeof(float) * input_blobs[i]->count(), cudaMemcpyHostToDevice);
	break;
      default:
	rt_err( "Unknown Caffe mode." );
      }  // switch (Caffe::mode())
    }
    //const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
    net_->ForwardPrefilled();
  }

  uint32_t get_layer_ix( p_Net_float net_, string const & out_layer_name ) {
    vect_string const & layer_names = net_->layer_names();
    for( uint32_t i = 0; i != layer_names.size(); ++i ) { if( out_layer_name == layer_names[i] ) { return i; } }
    rt_err( strprintf("layer out_layer_name=%s not found in network\n",str(out_layer_name).c_str() )); 
  }
  
  p_conv_op_t make_p_conv_op_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );

  template< typename CP > void set_param_from_conv_op( CP & cp, p_conv_op_t conv_op ) {
    // TODO/NOTE: non-square (_w/_h) handling is untested
    // SIGH: three cases are not quite consistent enough to be worth folding/sharing things more?
    cp.clear_pad_w(); cp.clear_pad_h(); cp.clear_pad();
    assert_st( conv_op->in_pad.bnds_are_same() ); // caffe can't handle different padding on +- edges
    u32_pt_t const & pad = conv_op->in_pad.p[0];
    if( pad.dims_are_same() ) { cp.set_pad( pad.d[0] ); }
    else { cp.set_pad_w( pad.d[0] ); cp.set_pad_h( pad.d[1] ); }

    cp.clear_kernel_w(); cp.clear_kernel_h(); cp.clear_kernel_size();
    if( conv_op->kern_sz.dims_are_same() ) { cp.set_kernel_size( conv_op->kern_sz.d[0] ); }
    else { cp.set_kernel_w( conv_op->kern_sz.d[0] ); cp.set_kernel_h( conv_op->kern_sz.d[1] ); }

    cp.clear_stride_w(); cp.clear_stride_h(); cp.clear_stride();
    if( conv_op->stride.dims_are_same() ) { cp.set_stride( conv_op->stride.d[0] ); }
    else { cp.set_stride_w( conv_op->stride.d[0] ); cp.set_stride_h( conv_op->stride.d[1] ); }
  }

  template< typename CP > p_conv_op_t get_conv_op_from_param( CP const & cp ) {
    p_conv_op_t conv_op( new conv_op_t );
    // TODO/NOTE: non-square (_w/_h) handling is untested
    // SIGH: three cases are not quite consistent enough to be worth folding/sharing things more?
    if( !(cp.has_pad_w() || cp.has_pad_h()) ){ 
      u32_pt_t const p( cp.pad(), cp.pad() ); conv_op->in_pad = u32_box_t(p,p);
    } else { assert_st( cp.has_pad_w() && cp.has_pad_h() && (!cp.has_pad()) );
      u32_pt_t const p( cp.pad_w(), cp.pad_h() ); conv_op->in_pad = u32_box_t(p,p); 
    }
    if( !(cp.has_stride_w() || cp.has_stride_h()) ){ 
      conv_op->stride = u32_pt_t( cp.stride(), cp.stride() );
    } else { assert_st( cp.has_stride_w() && cp.has_stride_h() && (!cp.has_stride()) );
      conv_op->stride = u32_pt_t( cp.stride_w(), cp.stride_h() );
    }
    if( !(cp.has_kernel_w() || cp.has_kernel_h()) ){ 
      conv_op->kern_sz = u32_pt_t( cp.kernel_size(), cp.kernel_size() );
    } else { assert_st( cp.has_kernel_w() && cp.has_kernel_h() && (!cp.has_kernel_size()) );
      conv_op->kern_sz = u32_pt_t( cp.kernel_w(), cp.kernel_h() );
    }
    return conv_op;
  }

  p_conv_pipe_t make_p_conv_pipe_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );

  void copy_output_blob_data( p_Net_float net_, string const & out_layer_name, vect_p_nda_float_t & top ) {
    timer_t t("caffe_copy_output_blob_data");
    uint32_t const out_layer_ix = get_layer_ix( net_, out_layer_name );
    const vector<Blob<float>*>& output_blobs = net_->top_vecs()[ out_layer_ix ];
    top.clear();
    for( uint32_t bix = 0; bix < output_blobs.size(); ++bix ) {
      Blob<float> * const output_blob = output_blobs[bix];
      dims_t out_batch_dims( 4 );
      out_batch_dims.dims(3) = output_blob->width();
      out_batch_dims.dims(2) = output_blob->height();
      out_batch_dims.dims(1) = output_blob->channels();
      out_batch_dims.dims(0) = output_blob->num();
      p_nda_float_t out_batch( new nda_float_t );
      out_batch->set_dims( out_batch_dims );
      assert_st( out_batch->elems.sz == uint32_t(output_blob->count()) );
      top.push_back( out_batch );
      
      float * const dest = &out_batch->elems[0];
      switch (Caffe::mode()) {
      case Caffe::CPU: memcpy(dest, output_blob->cpu_data(), sizeof(float) * output_blob->count() ); break;
      case Caffe::GPU: cudaMemcpy(dest, output_blob->gpu_data(), sizeof(float) * output_blob->count(), 
				  cudaMemcpyDeviceToHost); break;
      default: LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
  }


  void copy_layer_blobs( p_Net_float net_, uint32_t const & layer_ix, vect_p_nda_float_t & blobs ) {
    timer_t t("caffe_copy_layer_blob_data");
    caffe::Layer<float>* layer = net_->layers()[ layer_ix ].get();
    const vector< shared_ptr< caffe::Blob<float> > >& layer_blobs = layer->blobs();
    blobs.clear();
    for( uint32_t bix = 0; bix < layer_blobs.size(); ++bix ) {
      Blob<float> * const layer_blob = layer_blobs[bix].get();
      dims_t blob_dims( 4 );
      blob_dims.dims(3) = layer_blob->width();
      blob_dims.dims(2) = layer_blob->height();
      blob_dims.dims(1) = layer_blob->channels();
      blob_dims.dims(0) = layer_blob->num();
      p_nda_float_t blob( new nda_float_t );
      blob->set_dims( blob_dims );
      assert_st( blob->elems.sz == uint32_t(layer_blob->count()) );
      blobs.push_back( blob );
      float * const dest = &blob->elems[0];
      switch (Caffe::mode()) {
      case Caffe::CPU: memcpy(dest, layer_blob->cpu_data(), sizeof(float) * layer_blob->count() ); break;
      case Caffe::GPU: cudaMemcpy(dest, layer_blob->gpu_data(), sizeof(float) * layer_blob->count(), 
				  cudaMemcpyDeviceToHost); break;
      default: LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
  }
  void copy_layer_blobs( p_Net_float net_, string const & layer_name, vect_p_nda_float_t & blobs ) {
    uint32_t const layer_ix = get_layer_ix( net_, layer_name );
    copy_layer_blobs( net_, layer_ix, blobs );
  }

  void set_layer_blobs( p_Net_float net_, uint32_t const & layer_ix, vect_p_nda_float_t & blobs ) {
    timer_t t("caffe_set_layer_blob_data");
    caffe::Layer<float>* layer = net_->layers()[ layer_ix ].get();
    const vector< shared_ptr< caffe::Blob<float> > >& layer_blobs = layer->blobs();
    assert( blobs.size() == layer_blobs.size() );
    for( uint32_t bix = 0; bix < layer_blobs.size(); ++bix ) {
      p_nda_float_t const & blob = blobs[bix];
      Blob<float> * const layer_blob = layer_blobs[bix].get();
      dims_t const & blob_dims = blob->dims;
      assert_st( blob_dims.dims(3) == (uint32_t)layer_blob->width() );
      assert_st( blob_dims.dims(2) == (uint32_t)layer_blob->height() );
      assert_st( blob_dims.dims(1) == (uint32_t)layer_blob->channels() );
      assert_st( blob_dims.dims(0) == (uint32_t)layer_blob->num() );
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
  void set_layer_blobs( p_Net_float net_, string const & layer_name, vect_p_nda_float_t & blobs ) {
    uint32_t const layer_ix = get_layer_ix( net_, layer_name );
    set_layer_blobs( net_, layer_ix, blobs );
  }


  // note: assumes/includes chans_to_area conversion
  u32_pt_t run_cnet_t::get_one_blob_img_out_sz( void ) {
    assert_st( net );
    uint32_t const out_layer_ix = get_layer_ix( net, out_layer_name );
    const vector<Blob<float>*>& output_blobs = net->top_vecs()[ out_layer_ix ];
    assert_st( output_blobs.size() == 1 );
    Blob<float> const * const output_blob = output_blobs[0];
    return u32_pt_t( output_blob->width(), output_blob->height() ).scale( u32_ceil_sqrt( output_blob->channels() ) );
  }


  p_nda_float_t run_one_blob_in_one_blob_out( p_Net_float net, string const & out_layer_name, p_nda_float_t const & in ) {
    assert_st( net );
    vect_p_nda_float_t in_data; 
    in_data.push_back( in ); // assume single input blob
    raw_do_forward( net, in_data );
    vect_p_nda_float_t out_data; 
    copy_output_blob_data( net, out_layer_name, out_data );
    assert( out_data.size() == 1 ); // assume single output blob
    return out_data.front();
  }
  p_nda_float_t run_cnet_t::run_one_blob_in_one_blob_out( void ) { 
    return boda::run_one_blob_in_one_blob_out( net, out_layer_name, in_batch ); }
  p_nda_float_t run_cnet_t::run_one_blob_in_one_blob_out_upsamp( void ) { 
    return boda::run_one_blob_in_one_blob_out( upsamp_net, out_layer_name, in_batch ); }
  p_conv_pipe_t run_cnet_t::cache_pipe( caffe::NetParameter & net_param ) { 
    // note; there is an unfortunate potential circular dependency
    // here: we may need the pipe info about the network before we
    // have set it up if the desired size of the input image depends
    // on the net architeture (i.e. support size / padding / etc ).
    // currently, the only thing we need the pipe for before setup is
    // the number of images. this is determined by the blf_pack code
    // which needs the supports sizes and padding info from the
    // pipe. but, since the pipe doesn't care about the the number of
    // input images (note: it does currently use in_sz to create the
    // conv_ios here, but that could be delayed), we can get away with
    // creating the net_param first, then the pipe, then altering
    // num_input_images input_dim field of the net_param, then setting
    // up the net. hmm.

    // note: we only handle a (very) limited set of possible layers/networks here.
    p_conv_pipe_t conv_pipe = make_p_conv_pipe_t_init_and_check_unused_from_lexp( parse_lexp("()"), 0 );
    //vect_string const & layer_names = net->layer_names();
    uint32_t last_out_chans = 0;
    for( int32_t i = 0; i != net_param.layers_size(); ++i ) { 
      caffe::LayerParameter const & lp = net_param.layers(i);
      assert_st( lp.has_name() );
      p_conv_op_t conv_op;
      if( 0 ) {
      } else if( lp.has_convolution_param() ) {
	caffe::ConvolutionParameter const & cp = lp.convolution_param();
	conv_op = get_conv_op_from_param( cp );
	conv_op->type = "conv";
	assert_st( cp.num_output() >= 0 ); // should zero be allowed?
	conv_op->out_chans = cp.num_output();
	last_out_chans = conv_op->out_chans;
      } else if( lp.has_pooling_param() ) {
	caffe::PoolingParameter const & pp = lp.pooling_param();
	conv_op = get_conv_op_from_param( pp );
	conv_op->type = "pool";
	// global pooling iff kernel size is all zeros (we use as a special value)
	assert_st( conv_op->kern_sz.is_zeros() == pp.global_pooling() ); 
	conv_op->out_chans = last_out_chans; // assume unchanged from last conv layer 
      } else if( lp.has_inner_product_param() ) {
	caffe::InnerProductParameter const & ipp = lp.inner_product_param();
	conv_op.reset( new conv_op_t );
	conv_op->type = "pool";
	conv_op->out_chans = ipp.num_output();
      }
      if( conv_op ) { 
	conv_op->tag = lp.name();
	conv_pipe->convs->push_back( *conv_op ); 
      }
      if( out_layer_name == lp.name() ) { 
	conv_pipe->calc_support_info();
	return conv_pipe;
      }
    }
    rt_err( strprintf("layer out_layer_name=%s not found in network\n",str(out_layer_name).c_str() )); 
  }

  struct synset_elem_t {
    string id;
    string tag;
  };
  std::ostream & operator <<(std::ostream & os, synset_elem_t const & v) { return os << "id=" << v.id << " tag=" << v.tag; }
  

  void read_synset( p_vect_synset_elem_t out, filename_t const & fn ) {
    p_ifstream ifs = ifs_open( fn );  
    string line;
    while( !ifs_getline( fn.in, ifs, line ) ) {
      size_t const spos = line.find( ' ' );
      if( spos == string::npos ) { rt_err( "failing to parse synset line: no space found: line was '" + line + "'" ); }
      size_t cpos = line.find( ',', spos+1 );
      if( cpos == string::npos ) { cpos = line.size(); }
      assert( spos < cpos );
      uint32_t const tag_len = cpos - spos - 1;
      if( !tag_len ) { rt_err( "failing to parse synset line: no tag found, (comma after first space) or (first space at end of line (note: implies no command after first space)): line was '" + line + "'" ); }
      synset_elem_t t;
      t.id = string( line, 0, spos );
      t.tag = string( line, spos+1, tag_len );
      out->push_back( t );
    }
    //printf( "out=%s\n", str(out).c_str() );
#if 0
    p_ofstream tags = ofs_open("tags.txt");
    for( vect_synset_elem_t::const_iterator i = out->begin(); i != out->end(); ++i ) {
      (*tags) << ( i->tag + "\n" );
    }
#endif
  }


  void run_cnet_t::main( nesi_init_arg_t * nia ) { 
    setup_cnet();
  }

  void run_cnet_t::create_net_param( void ) {
    p_string ptt_str = read_whole_fn( ptt_fn );
    net_param.reset( new caffe::NetParameter );
#if 0 
    // old method: use special boda template prototxt, format it to
    // fill in the fields we want, then create the net_param from that
    string out_pt_str;
    str_format_from_nvm_str( out_pt_str, *ptt_str, 
			     strprintf( "(xsize=%s,ysize=%s,num=%s,chan=%s)", 
					str(in_sz.d[0]).c_str(), str(in_sz.d[1]).c_str(), 
					str(in_num_imgs).c_str(), str(in_num_chans).c_str() ) );
    bool const ret = google::protobuf::TextFormat::ParseFromString( out_pt_str, net_param.get() );
    assert_st( ret );
    // FIXME: old method incomplete/unmaintained
#endif

    // read the 'stock' deploy prototxt, and then override
    // the input dims using knowledge of the protobuf format.
    bool const ret = google::protobuf::TextFormat::ParseFromString( *ptt_str, net_param.get() );
    assert_st( ret );
    assert_st( net_param->input_dim_size() == 4 );
    net_param->set_input_dim(0,in_num_imgs);
    net_param->set_input_dim(1,in_num_chans);
    net_param->set_input_dim(2,in_sz.d[1]);
    net_param->set_input_dim(3,in_sz.d[0]);
    if( enable_upsamp_net ) {
      upsamp_net_param.reset( new caffe::NetParameter( *net_param ) ); // start with copy of net_param
      // halve the stride and kernel size for the first layer and rename it to avoid caffe trying to load weights for it
      assert_st( upsamp_net_param->layers_size() ); // better have at least one layer
      caffe::LayerParameter * lp = upsamp_net_param->mutable_layers(0);
      if( !lp->has_convolution_param() ) { rt_err( "first layer of net not conv layer; don't know how to create upsampled network"); }
      caffe::ConvolutionParameter * cp = lp->mutable_convolution_param();
      p_conv_op_t conv_op = get_conv_op_from_param( *cp );
      // FIXME: we probably need to deal with padding better here?
      conv_op->kern_sz = ceil_div( conv_op->kern_sz, u32_pt_t{2,2} );
      assert_st( conv_op->in_pad.bnds_are_same() );
      conv_op->in_pad.p[0] = ceil_div( conv_op->in_pad.p[0], u32_pt_t{2,2} );
      conv_op->in_pad.p[1] = conv_op->in_pad.p[0];
      for( uint32_t i = 0; i != 2; ++i ) {
	if( (conv_op->stride.d[i]&1) ) { rt_err( "first conv layer has odd stride; don't know how to create upsampled network" ); }
	conv_op->stride.d[i] /= 2;
      }
      set_param_from_conv_op( *cp, conv_op );
      assert_st( lp->has_name() );
      lp->set_name( lp->name() + "-in-2X-us" );
    }
  }

  // most clients call this. others might inline it to be able to call setup_cnet_adjust_in_num_imgs()
  void run_cnet_t::setup_cnet( void ) {
    setup_cnet_param_and_pipe();
    // optionally, could call setup_cnet_adjust_in_num_imgs( ... ) here
    setup_cnet_net_and_batch();
  }

  conv_support_info_t const & run_cnet_t::get_ol_csi( bool const & from_upsamp_net ) {
    if( from_upsamp_net ) { assert_st( enable_upsamp_net && conv_pipe_upsamp ); return conv_pipe_upsamp->conv_sis.back(); }
    return conv_pipe->conv_sis.back();
  }

  void run_cnet_t::setup_cnet_param_and_pipe( void ) {
    assert( !net_param );
    create_net_param();
    conv_pipe = cache_pipe( *net_param );
    out_s = u32_ceil_sqrt( conv_pipe->convs->back().out_chans );
    if( enable_upsamp_net ) { 
      conv_pipe_upsamp = cache_pipe( *upsamp_net_param );
      assert_st( out_s == u32_ceil_sqrt( conv_pipe_upsamp->convs->back().out_chans ) ); // FIXME: too strong?
    }
    // FIXME: only for non-upsamp net
    conv_ios = conv_pipe->calc_sizes_forward( in_sz, 0 ); 

  }
  void run_cnet_t::setup_cnet_adjust_in_num_imgs( uint32_t const in_num_imgs_ ) {
    assert_st( net_param && conv_pipe );
    assert_st( net_param->input_dim_size() == 4 );
    in_num_imgs = in_num_imgs_;
    net_param->set_input_dim(0,in_num_imgs);
    if( enable_upsamp_net ) { 
      upsamp_net_param->set_input_dim(0,in_num_imgs); } // FIXME/TODO: for now, run upsamp on all planes
  }

#if 0 // untested/unused example code for weight manipulation
  void create_identity_weights( p_Net_float net ) {
    vect_p_nda_float_t blobs;
    copy_layer_blobs( net, 0, blobs );
    assert_st( blobs.size() == 2 ); // filters, biases
    p_nda_float_t biases = blobs[1];
    for( dims_iter_t di( biases->dims ) ; ; ) { biases->at(di.di) = 0; if( !di.next() ) { break; } } // all biases 0
    p_nda_float_t filts = blobs[0];

    uint32_t const width = filts->dims.dims(3);
    uint32_t const height = filts->dims.dims(2); 
    uint32_t const channels = filts->dims.dims(1);
    uint32_t const num = filts->dims.dims(0);

    assert_st( channels == num ); // for now, only handling case where input chans == output chans

    // it's unclear how to handle even width/height, depending on padding in particular
    assert_st( width & 1 );
    assert_st( height & 1 );

    //for( uint32_t i = 0; i != num; ++i ) { filts->at4( i, i, (h+1)/2, (w+1)/2 ) = 1; }

    for( dims_iter_t di( filts->dims ) ; ; ) { 
      float val = 0; // FIXME: add noise here
      if( (di.di[2] == ((height+1)/2)) && // center y pel in filt
	  (di.di[3] == ((width+1)/2)) && // center x pel in filt
	  (di.di[0] == di.di[1]) ) // in_chan == out_chan
      { val += 1; }

      filts->at(di.di) = val;
      if( !di.next() ) { break; } 
    }    

    set_layer_blobs( net, 0, blobs );
  }
#endif


  void run_cnet_t::create_upsamp_layer_0_weights( void ) {
    vect_p_nda_float_t usl_blobs;
    copy_layer_blobs( net, 0, usl_blobs );

    vect_p_nda_float_t usl_blobs_upsamp;
    copy_layer_blobs( upsamp_net, 0, usl_blobs_upsamp );

    assert_st( usl_blobs.size() == 2 ); // filters, biases
    assert_st( usl_blobs_upsamp.size() == 2 ); // filters, biases
    assert_st( usl_blobs[1]->dims == usl_blobs_upsamp[1]->dims ); // biases should be same shape (and same strides?)
    usl_blobs_upsamp[1] = usl_blobs[1]; // use biases unchanged in upsamp net
    assert_st( usl_blobs[0]->dims.dims(0) == usl_blobs_upsamp[0]->dims.dims(0) );
    assert_st( usl_blobs[0]->dims.dims(1) == usl_blobs_upsamp[0]->dims.dims(1) );
    assert_st( u32_ceil_div( usl_blobs[0]->dims.dims(2), 2 ) == usl_blobs_upsamp[0]->dims.dims(2) );
    assert_st( u32_ceil_div( usl_blobs[0]->dims.dims(3), 2 ) == usl_blobs_upsamp[0]->dims.dims(3) );

    for( dims_iter_t di( usl_blobs_upsamp[0]->dims ) ; ; ) { usl_blobs_upsamp[0]->at(di.di) = 0; 
      if( !di.next() ) { break; } 
    }

    for( dims_iter_t di( usl_blobs[0]->dims ) ; ; ) { 
      usl_blobs_upsamp[0]->at4(di.di[0],di.di[1],di.di[2]>>1,di.di[3]>>1) += usl_blobs[0]->at( di.di );
      if( !di.next() ) { break; } 
    }

    set_layer_blobs( upsamp_net, 0, usl_blobs_upsamp );
  }

  void run_cnet_t::setup_cnet_net_and_batch( void ) {
    assert_st( !net );
    init_caffe( gpu_id ); // FIXME/note: only does something on first call
    net = caffe_create_net( *net_param, trained_fn.exp );      
    if( enable_upsamp_net ) { 
      upsamp_net = caffe_create_net( *upsamp_net_param, trained_fn.exp ); 
      create_upsamp_layer_0_weights();
    }

    assert_st( !in_batch );
    dims_t in_batch_dims( 4 );
    in_batch_dims.dims(3) = in_sz.d[0];
    in_batch_dims.dims(2) = in_sz.d[1];
    in_batch_dims.dims(1) = in_num_chans; 
    in_batch_dims.dims(0) = in_num_imgs;
    in_batch.reset( new nda_float_t );
    in_batch->set_dims( in_batch_dims );


  }

  void cnet_predict_t::main( nesi_init_arg_t * nia ) { 
    setup_cnet();
    setup_predict();
    p_img_t img_in( new img_t );
    img_in->load_fn( img_in_fn.exp );
    do_predict( img_in, 1 );
  }

  void cnet_predict_t::setup_predict( void ) {
    assert_st( !out_labels );
    out_labels.reset( new vect_synset_elem_t );
    read_synset( out_labels, out_labels_fn );

    assert( pred_state.empty() );
    if( scale_infos.empty() ) { setup_scale_infos(); } // if not specified, assume whole image / single scale 

    uint32_t const out_chans = conv_pipe->convs->back().out_chans;

    if( get_ol_csi(0).support_sz.is_zeros() ) { // only sensible in single-scale case 
      assert_st( scale_infos.size() == 1 );
      assert_st( scale_infos.back().img_sz == nominal_in_sz );
      assert_st( scale_infos.back().place.is_zeros() );
      assert_st( scale_infos.back().bix == 0 );
      //assert_st( enable_upsamp_net == 0 ); // too strong?
    }

    for( vect_scale_info_t::iterator i = scale_infos.begin(); i != scale_infos.end(); ++i ) {
      i->psb = pred_state.size();
      for( uint32_t bc = 0; bc < out_chans; ++bc ) {
	for( int32_t by = i->feat_box.p[0].d[1]; by < i->feat_box.p[1].d[1]; ++by ) {
	  for( int32_t bx = i->feat_box.p[0].d[0]; bx < i->feat_box.p[1].d[0]; ++bx ) {
	    pred_state.push_back( pred_state_t{} );
	    pred_state_t & ps = pred_state.back();
	    ps.label_ix = bc; 
	    u32_pt_t const feat_xy = {bx, by};
	    u32_box_t feat_pel_box{feat_xy,feat_xy+u32_pt_t{1,1}};
	    i32_box_t valid_in_xy, core_valid_in_xy; // note: core_valid_in_xy unused
	    unchecked_out_box_to_in_boxes( valid_in_xy, core_valid_in_xy, u32_to_i32( feat_pel_box ), 
					   get_ol_csi(i->from_upsamp_net), i->img_sz );
	    valid_in_xy -= u32_to_i32(i->place); // shift so image nc is at 0,0
	    valid_in_xy = valid_in_xy * u32_to_i32(nominal_in_sz) / u32_to_i32(i->img_sz); // scale for scale
	    ps.img_box = valid_in_xy;
	  }
	}
      }
    }
  
  }

  // single scale case
  void cnet_predict_t::setup_scale_infos( void ) {
    u32_pt_t const & feat_sz = conv_ios->back().sz;
    i32_box_t const valid_feat_box{{},u32_to_i32(feat_sz)};
    assert_st( valid_feat_box.is_strictly_normalized() );
    i32_box_t const valid_feat_img_box = valid_feat_box.scale(out_s);
    nominal_in_sz = in_sz;
    scale_infos.push_back( scale_info_t{nominal_in_sz,0,0,{},valid_feat_box,valid_feat_img_box} );
    assert_st( !enable_upsamp_net ); // unhandled(/nonsensical?)
  }

  void cnet_predict_t::setup_scale_infos( uint32_t const & interval, vect_u32_pt_t const & sizes, 
					  vect_u32_pt_w_t const & placements,
					  u32_pt_t const & nominal_in_sz_ ) {
    nominal_in_sz = nominal_in_sz_;
    conv_pipe->dump_pipe( std::cout );
    if( get_ol_csi(0).support_sz.is_zeros() ) {
      rt_err( "global pooling and/or\n inner product layers + trying to "
	      "compute dense features = madness!" );
    } 
    assert( scale_infos.empty() );

    if( enable_upsamp_net ) {
      // should be at least one octave for using upsampling net to make sense
      assert_st( sizes.size() >= interval ); 
      scale_infos.resize( interval ); // preallocate space for the upsampled octave sizes
    }

    for( uint32_t six = 0; six < sizes.size(); ++six ) {
      uint32_t const bix = placements.at(six).w;
      u32_pt_t const dest = placements.at(six);
      u32_pt_t const sz = sizes.at(six);

      u32_box_t per_scale_img_box{dest,dest+sz};
      // assume we've ensured that there is eff_tot_pad around the scale_img
      per_scale_img_box.p[0] -= get_ol_csi(0).eff_tot_pad.p[0];
      per_scale_img_box.p[1] += get_ol_csi(0).eff_tot_pad.p[1];
      i32_box_t valid_feat_box;
      in_box_to_out_box( valid_feat_box, per_scale_img_box, cm_valid, get_ol_csi(0) );
      assert_st( valid_feat_box.is_strictly_normalized() );      
      i32_box_t valid_feat_img_box = valid_feat_box.scale(out_s);
      scale_infos.push_back( scale_info_t{sz,0,bix,dest,valid_feat_box,valid_feat_img_box} ); // note: from_upsamp_net=0

      // if we're in the first placed octave, and the upsampling net
      // is enabled, add scale_infos for the in-net-upsampled octave
      // here.
      if( enable_upsamp_net && (six < interval) ) { 
	per_scale_img_box = u32_box_t{dest,dest+sz};
	// assume we've ensured that there is eff_tot_pad around the scale_img
	per_scale_img_box.p[0] -= get_ol_csi(1).eff_tot_pad.p[0];
	per_scale_img_box.p[1] += get_ol_csi(1).eff_tot_pad.p[1];

	in_box_to_out_box( valid_feat_box, per_scale_img_box, cm_valid, get_ol_csi(1) );
	assert_st( valid_feat_box.is_strictly_normalized() );
	valid_feat_img_box = valid_feat_box.scale(out_s); // FIXME: sort-of-not-right (wrong net out_s)
	scale_infos[six] = scale_info_t{sz,1,bix,dest,valid_feat_box,valid_feat_img_box}; // note: from_upsamp_net=1
      }
    }
    

    printf( "scale_infos=%s\n", str(scale_infos).c_str() );
  }

  template< typename T > struct gt_filt_prob {
    vector< T > const & v;
    gt_filt_prob( vector< T > const & v_ ) : v(v_) {}
    bool operator()( uint32_t const & ix1, uint32_t const & ix2 ) { 
      assert_st( ix1 < v.size() );
      assert_st( ix2 < v.size() );
      return v[ix1].filt_prob > v[ix2].filt_prob;
    }
  };

  typedef map< i32_box_t, anno_t > anno_map_t;

  // example command line for testing/debugging detection code:
  // boda capture_classify --cnet-predict='(in_sz=600 600,ptt_fn=%(models_dir)/nin_imagenet_nopad/deploy.prototxt,trained_fn=%(models_dir)/nin_imagenet_nopad/best.caffemodel,out_layer_name=relu12)' --capture='(cap_res=640 480)'

  //p_vect_anno_t cnet_predict_t::do_predict( p_img_t const & img_in, bool const print_to_terminal ) { }

  p_vect_anno_t cnet_predict_t::do_predict( p_img_t const & img_in, bool const print_to_terminal ) {
    p_img_t img_in_ds = resample_to_size( img_in, in_sz );
    subtract_mean_and_copy_img_to_batch( in_batch, 0, img_in_ds );
    p_nda_float_t out_batch = run_one_blob_in_one_blob_out();
    p_nda_float_t out_batch_upsamp;
    if( enable_upsamp_net ) { out_batch_upsamp = run_one_blob_in_one_blob_out_upsamp(); }
    return do_predict( out_batch, out_batch_upsamp, print_to_terminal );
  }

  p_vect_anno_t cnet_predict_t::do_predict( p_nda_float_t const & out_batch, p_nda_float_t const & out_batch_upsamp, 
					    bool const print_to_terminal ) {

    for( vect_scale_info_t::iterator i = scale_infos.begin(); i != scale_infos.end(); ++i ) {
      p_nda_float_t scale_batch = i->from_upsamp_net ? out_batch_upsamp : out_batch;
      dims_t const & sbd = scale_batch->dims;
      assert( sbd.sz() == 4 );
      assert( sbd.dims(1) == out_labels->size() );

      dims_t img_e( scale_batch->dims.sz() );
      dims_t img_b( img_e.sz() );
      img_b.dims(0) = i->bix;
      img_e.dims(0) = i->bix + 1;
      img_b.dims(1) = 0;
      img_e.dims(1) = sbd.dims(1);
      img_b.dims(2) = i->feat_box.p[0].d[1];
      img_e.dims(2) = i->feat_box.p[1].d[1];
      img_b.dims(3) = i->feat_box.p[0].d[0];
      img_e.dims(3) = i->feat_box.p[1].d[0];
      assert_st( img_e.fits_in( sbd ) );
      assert_st( img_b.fits_in( img_e ) );
      do_predict_region( scale_batch, img_b, img_e, i->psb );
    }
    return pred_state_to_annos( print_to_terminal );
  }


  i32_box_t cnet_predict_t::nms_grid_op( bool const & do_set, i32_box_t const & img_box ) {
    uint32_t tot_pel = 0;
    uint32_t over_pel = 0;

    i32_box_t shrunk_quant_img_box = floor_div( img_box.scale_and_round( nms_core_rat ), nms_grid_pels );

    nms_grid_t::iterator ci = nms_grid.find( shrunk_quant_img_box.center_rd() );
    i32_box_t center_match;
    if( ci != nms_grid.end() ) { center_match = ci->second; }
    uint32_t center_match_cnt = 0;

    for( int32_t by = shrunk_quant_img_box.p[0].d[1]; by < shrunk_quant_img_box.p[1].d[1]; ++by ) {
      for( int32_t bx = shrunk_quant_img_box.p[0].d[0]; bx < shrunk_quant_img_box.p[1].d[0]; ++bx ) {
	i32_pt_t const pel{bx,by};
	if( do_set ) { nms_grid[pel] = img_box; }
	else {
	  ++tot_pel;
	  nms_grid_t::iterator i = nms_grid.find( pel );
	  if( i != nms_grid.end() ) { ++over_pel; if( i->second == center_match ) { ++center_match_cnt; } }
	}
      }
    }
    if( center_match_cnt * 4 > tot_pel * 3 ) {  // mostly covers an existing match, so maybe add anno to that match
      assert_st( over_pel );
      return center_match;
    } else if( over_pel ) {
      return i32_box_t{};
    } else { return img_box; } // doesn't overlap
  }

  p_vect_anno_t cnet_predict_t::pred_state_to_annos( bool const print_to_terminal ) {
    anno_map_t annos;
    if( print_to_terminal ) {
      printf("\033[2J\033[1;1H");
      printf("---- frame -----\n");
    }
    vect_uint32_t disp_list;
    for( uint32_t i = 0; i < pred_state.size(); ++i ) {  if( pred_state[i].to_disp ) { disp_list.push_back(i); } }
    sort( disp_list.begin(), disp_list.end(), gt_filt_prob<pred_state_t>( pred_state ) );
    uint32_t num_disp = 0;
    nms_grid.clear();
    for( vect_uint32_t::const_iterator ii = disp_list.begin(); ii != disp_list.end(); ++ii ) {
      if( num_disp == max_num_disp ) { break; }
      pred_state_t const & ps = pred_state[*ii];
      // check nms
      i32_box_t const nms_box = nms_grid_op( 0, ps.img_box );
      if( nms_box == i32_box_t{} ) { continue; } //nms suppression condition: overlaps other core and no close-center-match
      anno_map_t::iterator ami = annos.find( nms_box ); // either ps.img_box or a close-matching overlap
      // nms supression condition: existing-anno-full 
      if( (ami != annos.end()) && (ami->second.item_cnt >= max_labels_per_anno) ) { continue; } 
      
      anno_t & anno = annos[nms_box];
      if( ami == annos.end() ) { // was new, init
	assert( nms_box == ps.img_box );
	anno.item_cnt = 0;
	nms_grid_op( 1, ps.img_box );
      } 
      bool const did_ins = anno.seen_label_ixs.insert( ps.label_ix ).second;
      if( !did_ins ) { continue; } // ignore dup labels 
      string anno_str;
      if( anno_mode == 0 ) {
	anno_str = strprintf( "%-20s -- filt_p=%-10s p=%-10s\n", str(out_labels->at(ps.label_ix).tag).c_str(), 
			      str(ps.filt_prob).c_str(),
			      str(ps.cur_prob).c_str() );
      } else if ( anno_mode == 1 ) {
	anno_str = strprintf( "%s\n", str(out_labels->at(ps.label_ix).tag).c_str() ); 
      } else if ( anno_mode == 3 ) {
	anno_str = strprintf( "%-4s %s\n", str(ii - disp_list.begin() + 1).c_str(), str(out_labels->at(ps.label_ix).tag).c_str() ); 
      } else if ( anno_mode == 2 ) {
	anno_str = strprintf( "%-20s -- sz=%s\n", 
			      str(out_labels->at(ps.label_ix).tag).c_str(), str(ps.img_box.sz()).c_str() ); 
      }

      anno.str += anno_str;
      ++anno.item_cnt;
      if( print_to_terminal ) { printstr( anno_str ); }
      ++num_disp;
    }
    if( print_to_terminal ) { printf("---- end frame -----\n"); }
    //printf( "obd=%s\n", str(obd).c_str() );
    //(*ofs_open( out_fn )) << out_pt_str;
    p_vect_anno_t ret_annos( new vect_anno_t );
    for( anno_map_t::const_iterator i = annos.begin(); i != annos.end(); ++i ) { 
      ret_annos->push_back( i->second ); 
      anno_t & anno = ret_annos->back();
      anno.box = i->first; 
      anno.fill = 0; anno.box_color = rgba_to_pel(170,40,40); anno.str_color = rgba_to_pel(220,220,255);
    }
    return ret_annos;
  }

  // fills in (part of) pred_state
  void cnet_predict_t::do_predict_region( p_nda_float_t const & out_batch, dims_t const & obb, dims_t const & obe,
					  uint32_t const & psb ) {
    dims_t const ob_span = obe - obb;
    assert( ob_span.sz() == 4 );
    assert( ob_span.dims(0) == 1 );
    assert( ob_span.dims(1) == out_labels->size() );

    uint32_t const num_pred = ob_span.dims_prod();
    assert_st( (psb+num_pred) <= pred_state.size() );
    uint32_t const num_pels = ob_span.dims(2)*ob_span.dims(3);
    vect_double pel_sums( num_pels, 0.0 );
    vect_double pel_maxs( num_pels, 0.0 );

    { uint32_t pel_ix = 0; uint32_t psix = psb; for( dims_iter_t di( obb, obe ) ; ; ++psix, ++pel_ix ) { 
	if( pel_ix == num_pels ) { pel_ix = 0; }
	double p = out_batch->at(di.di);
	pred_state.at(psix).cur_prob = p;
	max_eq( pel_maxs[pel_ix], p ) ;
	pel_sums[pel_ix] += p;
	if( !di.next() ) { break; } 
      }
    }    

    vect_uint32_t pel_is_pdf( num_pels, 0 );
    uint32_t tot_num_pdf = 0;
    // if the pel looks ~like a PDF, we leave it as is. otherwise, we apply a softmax
    for( uint32_t i = 0; i != num_pels; ++i ) {
      if( (fabs( pel_sums[i] - 1.0 ) < .01) && (pel_maxs[i] < 1.01)  ) { pel_is_pdf[i] = 1; ++tot_num_pdf; }
      pel_sums[i] = 0; // reused below if pel_is_pdf is false
    }
    //printf( "num_pels=%s tot_num_pdf=%s\n", str(num_pels).c_str(), str(tot_num_pdf).c_str() );

    // FIXME: is it wrong/bad for is_pdf to ber per-pel? should it be all-or-none somehow?
    { uint32_t pel_ix = 0; for( uint32_t psix = psb; psix != psb+num_pred; ++psix, ++pel_ix ) { 
	if( pel_ix == num_pels ) { pel_ix = 0; }
	if( !pel_is_pdf.at(pel_ix) ) { // if not already a PDF, apply a softmax
	  double & p = pred_state[psix].cur_prob;
	  double exp_p = exp(p - pel_maxs[pel_ix]);
	  p = exp_p;
	  pel_sums[pel_ix] += exp_p;
	}
      }
    }
    //printf( "pel_sums=%s pel_maxs=%s\n", str(pel_sums).c_str(), str(pel_maxs).c_str() );
    { uint32_t pel_ix = 0; for( uint32_t psix = psb; psix != psb+num_pred; ++psix, ++pel_ix ) { 
	if( pel_ix == num_pels ) { pel_ix = 0; }
	if( !pel_is_pdf.at(pel_ix) ) { pred_state.at(psix).cur_prob /= pel_sums.at(pel_ix); } // rest of softmax
      }
    }    

    // temportal filtering and setting to_disp
    { uint32_t pel_ix = 0; for( uint32_t psix = psb; psix != psb+num_pred; ++psix, ++pel_ix ) { 
	if( pel_ix == num_pels ) { pel_ix = 0; }
	pred_state_t & ps = pred_state.at(psix);

	if( !ps.filt_prob_init ) { ps.filt_prob_init = 1; ps.filt_prob = ps.cur_prob; }
	else { ps.filt_prob *= (1 - filt_rate); ps.filt_prob += ps.cur_prob * filt_rate; }

	if( ps.filt_prob >= filt_show_thresh ) { ps.to_disp = 1; }
	else if( ps.filt_prob <= filt_drop_thresh ) { ps.to_disp = 0; }
      }
    }
  }

#include"gen/caffeif.H.nesi_gen.cc"
}

