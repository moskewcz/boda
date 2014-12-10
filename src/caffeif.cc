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

  void copy_batch_to_img( p_nda_float_t const & out_batch, uint32_t img_ix, p_img_t const & img ) {
    
    dims_t const & obd = out_batch->dims;
    assert( obd.sz() == 4 );
    assert_st( img_ix < obd.dims(0) );

    u32_pt_t const out_sz( obd.dims(3), obd.dims(2) );
    uint32_t sqrt_out_chan;
    u32_pt_t img_sz;
    chans_to_area( sqrt_out_chan, img_sz, out_sz, obd.dims(1) );
    assert( sqrt_out_chan );
    assert( (sqrt_out_chan*sqrt_out_chan) >= obd.dims(1) );

    // set up dim iterators that span only the image we want to process
    dims_t img_e = out_batch->dims;
    dims_t img_b( img_e.sz() );
    img_b.dims(0) = img_ix;
    img_e.dims(0) = img_ix + 1;
    float const out_max = nda_reduce( *out_batch, max_functor<float>(), 0.0f, img_b, img_e ); // note clamp to 0
    //float const out_min = nda_reduce( *out_batch, min_functor<float>(), 0.0f, img_b, img_e ); // note clamp to 0
    //assert_st( out_min == 0.0f ); // shouldn't be any negative values
    //float const out_rng = out_max - out_min;

    assert_st( img->sz == img_sz );

    for( uint32_t y = 0; y < img->sz.d[1]; ++y ) {
      for( uint32_t x = 0; x < img->sz.d[0]; ++x ) {
	uint32_t const bx = x / sqrt_out_chan;
	uint32_t const by = y / sqrt_out_chan;
	uint32_t const bc = (y%sqrt_out_chan)*sqrt_out_chan + (x%sqrt_out_chan);
	uint32_t gv;
	if( bc < obd.dims(1) ) {
	  //float const norm_val = ((out_batch->at4(img_ix,bc,by,bx)-out_min) / out_rng );
	  float const norm_val = ((out_batch->at4(img_ix,bc,by,bx)) / out_max );
	  gv = grey_to_pel( uint8_t( std::min( 255.0, 255.0 * norm_val ) ) );
	  //gv = grey_to_pel( uint8_t( std::min( 255.0, 255.0 * (log(.01) - log(std::max(.01f,norm_val))) / (-log(.01)) )));
	} else { gv = grey_to_pel( 0 ); }
	img->set_pel( {x, y}, gv );
      }
    }
  }

  p_Net_float init_caffe( string const & param_str, string const & trained_fn ) {
    static bool caffe_is_init = 0;
    if( !caffe_is_init ) {
      caffe_is_init = 1;
      timer_t t("caffe_init");
      google::InitGoogleLogging("boda_caffe");
      Caffe::set_phase(Caffe::TEST);
      Caffe::set_mode(Caffe::GPU);
      Caffe::SetDevice(0);
      //Caffe::set_mode(Caffe::CPU);
    }
    timer_t t("caffe_load_net");
    caffe::NetParameter param;
    bool const ret = google::protobuf::TextFormat::ParseFromString( param_str, &param );
    assert_st( ret );

    p_Net_float net( new Net_float( param ) );
    net->CopyTrainedLayersFrom( trained_fn );

#if 0
    int total_iter = 10; // atoi(argv[3]);
    LOG(ERROR) << "Running " << total_iter << " iterations.";

    double test_accuracy = 0;
    for (int i = 0; i < total_iter; ++i) {
      const vector<Blob<float>*>& result = net->ForwardPrefilled();
      test_accuracy += result[0]->cpu_data()[0];
      LOG(ERROR) << "Batch " << i << ", accuracy: " << result[0]->cpu_data()[0];
    }
    test_accuracy /= total_iter;
    LOG(ERROR) << "Test accuracy: " << test_accuracy;
#endif
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
  p_conv_pipe_t run_cnet_t::get_pipe( void ) { 
    // note; there is an unfortunate potential circular dependency here: we may need the pipe info
    // about the network before we have set it up if the desired size of the input image depends on
    // the net architeture (i.e. support size / padding / etc ). we could potentially create a
    // 'dummy' net with some semi-arbitrary input sizes here. but, for now, we can squeak around the
    // dependency in our current use cases ...
    assert_st( net ); // net must already be set up
    // note: we only handle a (very) limited set of possible layers/networks here.
    if( conv_pipe ) { return conv_pipe; } // already created
    conv_pipe = make_p_conv_pipe_t_init_and_check_unused_from_lexp( parse_lexp("()"), 0 );
    vect_string const & layer_names = net->layer_names();
    uint32_t last_out_chans = 0;
    for( uint32_t i = 0; i != layer_names.size(); ++i ) { 
      caffe::LayerParameter const & lp = net->layers()[i]->layer_param();
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
      }
      if( conv_op ) { 
	assert( lp.has_name() );
	conv_op->tag = lp.name();
	conv_pipe->convs->push_back( *conv_op ); 
      }
      if( out_layer_name == layer_names[i] ) { 
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
  }


  void run_cnet_t::main( nesi_init_arg_t * nia ) { 
    setup_cnet();
  }

  void run_cnet_t::setup_cnet( void ) {
    p_string ptt_str = read_whole_fn( ptt_fn );
    string out_pt_str;

    dims_t in_batch_dims( 4 );
    in_batch_dims.dims(3) = in_sz.d[0];
    in_batch_dims.dims(2) = in_sz.d[1];
    in_batch_dims.dims(1) = in_num_chans; 
    in_batch_dims.dims(0) = in_num_imgs;

    str_format_from_nvm_str( out_pt_str, *ptt_str, 
			     strprintf( "(xsize=%s,ysize=%s,num=%s,chan=%s)", 
					str(in_batch_dims.dims(3)).c_str(), str(in_batch_dims.dims(2)).c_str(), 
					str(in_batch_dims.dims(0)).c_str(), str(in_batch_dims.dims(1)).c_str() ) );
    assert_st( !net );
    net = init_caffe( out_pt_str, trained_fn.exp );      

    assert_st( !in_batch );
    in_batch.reset( new nda_float_t );
    in_batch->set_dims( in_batch_dims );
  }

  void cnet_predict_t::main( nesi_init_arg_t * nia ) { 
    setup_predict();
    p_img_t img_in( new img_t );
    img_in->load_fn( img_in_fn.exp );
    do_predict( img_in, 1 );
  }

  void cnet_predict_t::setup_predict( void ) {
    assert_st( !out_labels );
    out_labels.reset( new vect_synset_elem_t );
    read_synset( out_labels, out_labels_fn );
    setup_cnet();
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

  p_vect_anno_t cnet_predict_t::do_predict( p_img_t const & img_in, bool const print_to_terminal ) {
    conv_support_info_t const & ol_csi = get_pipe()->conv_sis.back();
    p_img_t img_in_ds = resample_to_size( img_in, in_sz );
    subtract_mean_and_copy_img_to_batch( in_batch, 0, img_in_ds );
    p_nda_float_t out_batch = run_one_blob_in_one_blob_out();

    dims_t const & obd = out_batch->dims;
    assert( obd.sz() == 4 );
    assert( obd.dims(0) == 1 );
    assert( obd.dims(1) == out_labels->size() );
    //assert( obd.dims(2) == 1 );
    //assert( obd.dims(3) == 1 );
    bool const init_filt_prob = pred_state.empty();
    if( init_filt_prob ) { pred_state.resize( obd.dims_prod() ); }
    
    {
      uint32_t ix = 0;
      for( dims_iter_t di( obd ) ; ; ++ix ) { 
	float const p = out_batch->at(di.di);
	pred_state_t & ps = pred_state[ix];
	ps.cur_prob = p;
	if( init_filt_prob ) { 
	  ps.filt_prob = p; ps.to_disp = 0; ps.label_ix = di.di[1]; 
	  u32_pt_t const feat_xy = {di.di[3],di.di[2]};
	  u32_box_t feat_pel_box{feat_xy,feat_xy+u32_pt_t{1,1}};

	  i32_box_t valid_in_xy, core_valid_in_xy; // note: core_valid_in_xy unused
	  unchecked_out_box_to_in_boxes( valid_in_xy, core_valid_in_xy, u32_to_i32( feat_pel_box ), 
					 ol_csi, in_sz );
	  ps.img_box = valid_in_xy;
	}
	else { ps.filt_prob *= (1 - filt_rate); ps.filt_prob += p * filt_rate; }
	if( ps.filt_prob >= filt_show_thresh ) { ps.to_disp = 1; }
	else if( ps.filt_prob <= filt_drop_thresh ) { ps.to_disp = 0; }
	if( !di.next() ) { break; } 
      }
    }

    p_vect_anno_t annos( new vect_anno_t );
    annos->push_back( anno_t{{{0,0},u32_to_i32(img_in->sz)}, rgba_to_pel(170,40,40), 0, "", rgba_to_pel(220,220,255) } );

    if( print_to_terminal ) {
      printf("\033[2J\033[1;1H");
      printf("---- frame -----\n");
    }
    vect_uint32_t disp_list;
    for( uint32_t i = 0; i < pred_state.size(); ++i ) {  if( pred_state[i].to_disp ) { disp_list.push_back(i); } }
    sort( disp_list.begin(), disp_list.end(), gt_filt_prob<pred_state_t>( pred_state ) );
    uint32_t num_disp = 0;
    for( vect_uint32_t::const_iterator ii = disp_list.begin(); ii != disp_list.end(); ++ii ) {
      if( num_disp == max_num_disp ) { break; }
      pred_state_t const & ps = pred_state[*ii];
      string const anno_str = strprintf( "%-20s -- filt_p=%-10s p=%-10s\n", str(out_labels->at(ps.label_ix).tag).c_str(), 
					 str(ps.filt_prob).c_str(),
					 str(ps.cur_prob).c_str() );
      annos->back().str += anno_str;
      if( print_to_terminal ) { printstr( anno_str ); }
      ++num_disp;
    }
    if( print_to_terminal ) { printf("---- end frame -----\n"); }
    //printf( "obd=%s\n", str(obd).c_str() );
    //(*ofs_open( out_fn )) << out_pt_str;
    return annos;
  }

#include"gen/caffeif.H.nesi_gen.cc"
}

