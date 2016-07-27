// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"timers.H"
#include"str_util.H"
#include"conv_util.H"
#include"lexp.H"
#include"nesi.H"
#include"caffepb.H"
#include<google/protobuf/text_format.h> 
#include<google/protobuf/io/zero_copy_stream_impl.h>
#include<google/protobuf/io/coded_stream.h>
// our local copy of caffe.proto, which better be identical to the caffe version if we're compiling with caffe support.
#include"caffe.pb.h" 
#include"rand_util.H"
#include"imagenet_util.H"
#include"img_io.H"

// we get this function from our hacked-up version of
// upgrade_proto.cpp, so we can upgrade NetParameters from V1->V2 the
// code for V0->V1 is still in there and could be made to work with a
// bit of effort to rewrite the error handling a bit. note that in
// general we're a little stricter and less verbose than the original
// code.
namespace boda_caffe { bool UpgradeNetAsNeeded(const std::string& param_file, boda::net_param_t* param); }
// similarly, we have a mostly-copied version of StateMeetsRule extracted from the Net class:
namespace boda_caffe { bool layer_included_for_state( caffe::LayerParameter const & layer_param, caffe::NetState const & net_state ); }

namespace boda 
{
  void dims_t_to_shape( dims_t const & dims, caffe::BlobShape & bs ) {
    assert( bs.dim_size() == 0 );
    for( uint32_t i = 0; i != dims.sz(); ++i ) { bs.add_dim( dims.dims(i) ); }
  }

  dims_t shape_to_dims_t( caffe::BlobShape const & shape ) {
    dims_t ret;
    uint32_t const num_dims = shape.dim_size();
    ret.resize_and_zero( num_dims );
    for( uint32_t i = 0; i != num_dims; ++i ) { ret.dims(i) = shape.dim(i); }
    return ret;
  }

  // note: this function is only used in a few places, and is quite old. the conv_op it is passed should always be
  // more-or-less of type Convolution_coi, although the type is not actually set in the usages. conv_op used only as a
  // temporary to capture kern_sz/in_pad/stride from protobuf ConvolutionParam, modify them, and write them back to
  // another ConvolutionParam.
  template< typename CP > void set_param_from_conv_op( CP & cp, p_conv_op_t conv_op ) {
    // TODO/NOTE: non-square (_w/_h) handling is untested
    // SIGH: three cases are not quite consistent enough to be worth folding/sharing things more?
    cp.clear_pad_w(); cp.clear_pad_h(); cp.clear_pad();
    if( has( conv_op->dims_vals, "in_pad" ) ) { 
      // FIXME: could attempt to handle ND case here
      u32_pt_t const pad = conv_op->in_pad();
      if( pad.dims_are_same() ) { cp.add_pad( pad.d[0] ); }
      else { cp.set_pad_w( pad.d[0] ); cp.set_pad_h( pad.d[1] ); }
    }

    // FIXME: could attempt to handle ND case here
    u32_pt_t const kern_sz = conv_op->kern_sz();
    cp.clear_kernel_w(); cp.clear_kernel_h(); cp.clear_kernel_size();
    if( kern_sz.dims_are_same() ) { cp.add_kernel_size( kern_sz.d[0] ); }
    else { cp.set_kernel_w( kern_sz.d[0] ); cp.set_kernel_h( kern_sz.d[1] ); }

    if( has( conv_op->dims_vals, "stride" ) ) { 
      // FIXME: could attempt to handle ND case here
      cp.clear_stride_w(); cp.clear_stride_h(); cp.clear_stride();
      if( conv_op->stride().dims_are_same() ) { cp.add_stride( conv_op->stride().d[0] ); }
      else { cp.set_stride_w( conv_op->stride().d[0] ); cp.set_stride_h( conv_op->stride().d[1] ); }
    }
  }
  template void set_param_from_conv_op< caffe::ConvolutionParameter >( caffe::ConvolutionParameter & cp, p_conv_op_t conv_op );

  void multi_dim_err( uint32_t const & num, string const & tag ) { 
    rt_err( strprintf( "saw %s %s values. arbitrary # of spatial dims case not supported. use 1 or 2 %s values or %s_w/%s_h.", 
		       str(num).c_str(), tag.c_str(), tag.c_str(), tag.c_str(), tag.c_str() ) );
  }
  // joy. now that caffe has multi-dim support for conv, but not pooling, we can't seem to share this code between the two cases anymore ...
  void fill_in_conv_op_from_param( p_conv_op_t const & conv_op, caffe::ConvolutionParameter const & cp ) {
    // TODO/NOTE: non-square (_w/_h) handling is untested
    // FIXME: xxx_size() == 2 case, untested, is order right?
    // SIGH: three cases are not quite consistent enough to be worth folding/sharing things more?
    // FIXME: in caffe, if any of the following dims have a single dimention, this is interpreted to mean the number of
    // spatial dims in the input blob. for now, we'll assume that is two, which certainly isn't right in
    // general. however, it's unclear if we really want to allow the dimentionality of the operation to be implicit
    // depending on the inputs ...
    if( !(cp.has_pad_w() || cp.has_pad_h()) ){
      // for the 0-dims case, we use the implicit default of 0 (from comments in caffe.proto) AND our default of 2 spatial axes
      if( cp.pad_size() == 0 ) { } // leave un-set; will be set by coi default
      else if( cp.pad_size() == 1 ) { conv_op->dims_vals["in_pad"] = dims_t{ { cp.pad(0), cp.pad(0) }, {"y","x"}, "none" }; }
      else if( cp.pad_size() == 2 ) { conv_op->dims_vals["in_pad"] = dims_t{ { cp.pad(0), cp.pad(1) }, {"y","x"}, "none" }; }
      else { multi_dim_err( cp.pad_size(), "in_pad" ); } // FIXME: we could handle this now as a first step for N-D support
    } else { assert_st( cp.has_pad_w() && cp.has_pad_h() && (!cp.pad_size()) );
      conv_op->dims_vals["in_pad"] = dims_t{ { cp.pad_h(), cp.pad_w() }, {"y","x"}, "none" }; 
    }
    if( !(cp.has_stride_w() || cp.has_stride_h()) ){ 
      // for the 0-dims case, we use the implicit default of 1 (from comments in caffe.proto) AND our default of 2 spatial axes
      if( cp.stride_size() == 0 ) { } // leave un-set; will be set by coi default
      else if( cp.stride_size() == 1 ) { conv_op->dims_vals["stride"] = dims_t{ { cp.stride(0), cp.stride(0) }, {"y","x"}, "none" }; }
      else if( cp.stride_size() == 2 ) { conv_op->dims_vals["stride"] = dims_t{ { cp.stride(0), cp.stride(1) }, {"y","x"}, "none" }; }
      else { multi_dim_err( cp.stride_size(), "stride" ); } // FIXME: we could handle this now as a first step for N-D support
    } else { assert_st( cp.has_stride_w() && cp.has_stride_h() && (!cp.stride_size()) );
      conv_op->dims_vals["stride"] = dims_t{ { cp.stride_h(), cp.stride_w() }, {"y","x"}, "none" }; 
    }
    if( !(cp.has_kernel_w() || cp.has_kernel_h()) ){ 
      // for the 0-dims case, we use the implicit default of 1 (from comments in caffe.proto) AND our default of 2 spatial axes
      if( cp.kernel_size_size() == 0 ) { } // leave un-set; will be set by coi default
      else if( cp.kernel_size_size() == 1 ) { conv_op->dims_vals["kern_sz"] = dims_t{ { cp.kernel_size(0), cp.kernel_size(0) }, {"y","x"}, "none" }; }
      else if( cp.kernel_size_size() == 2 ) { conv_op->dims_vals["kern_sz"] = dims_t{ { cp.kernel_size(0), cp.kernel_size(1) }, {"y","x"}, "none" }; }
      else { multi_dim_err( cp.kernel_size_size(), "kernel_size" ); }
    } else { assert_st( cp.has_kernel_w() && cp.has_kernel_h() && (!cp.kernel_size_size()) );
      conv_op->dims_vals["kern_sz"] = dims_t{ { cp.kernel_h(), cp.kernel_w() }, {"y","x"}, "none" }; 
    }
  }
  void fill_in_conv_op_from_param( p_conv_op_t const & conv_op, caffe::PoolingParameter const & cp ) {
    // TODO/NOTE: non-square (_w/_h) handling is untested
    // SIGH: three cases are not quite consistent enough to be worth folding/sharing things more?
    if( !(cp.has_pad_w() || cp.has_pad_h()) ){
      conv_op->dims_vals["in_pad"] = dims_t{ { cp.pad(), cp.pad() }, {"y","x"}, "none" };
    } else { assert_st( cp.has_pad_w() && cp.has_pad_h() && (!cp.has_pad()) );
      conv_op->dims_vals["in_pad"] = dims_t{ { cp.pad_h(), cp.pad_w() }, {"y","x"}, "none" };
    }
    if( !(cp.has_stride_w() || cp.has_stride_h()) ){ 
      conv_op->dims_vals["stride"] = dims_t{ { cp.stride(), cp.stride() }, {"y","x"}, "none" };
    } else { assert_st( cp.has_stride_w() && cp.has_stride_h() && (!cp.has_stride()) );
      conv_op->dims_vals["stride"] = dims_t{ { cp.stride_h(), cp.stride_w() }, {"y","x"}, "none" };
    }
    if( cp.has_kernel_size() ) {
      assert_st( (!cp.has_kernel_w()) && (!cp.has_kernel_h()) ); 
      conv_op->dims_vals["kern_sz"] = dims_t{ { cp.kernel_size(), cp.kernel_size() }, {"y","x"}, "none" };
    } else if( cp.has_kernel_w() || cp.has_kernel_h() ) { 
      assert_st( cp.has_kernel_w() && cp.has_kernel_h() );
      conv_op->dims_vals["kern_sz"] = dims_t{ { cp.kernel_h(), cp.kernel_w() }, {"y","x"}, "none" };
    } else { } // has neither kernel_size nor kernel_{w,h} // leave un-set; will be set by coi default
  }

  p_conv_op_t make_p_conv_op_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );

#define RF_TO_VEC( V, RF ) { for( int32_t i = 0; i != RF##_size(); ++i ) { V.push_back( RF(i) ); } }
  void net_input_err( p_net_param_t const &net_param ) {
    rt_err( strprintf( "error: top-level input blob specification error. the number of input blobs (net_param->input_size()=%s) must be equal either to the number of input shapes (net_param->input_shape_size()=%s) or to the number of input dims (net_param->input_dim_size()=%s) divided by four. also, only one of the two shape-specification methods (input_dim or input_shape) may be used at the same time. \n", str(net_param->input_size()).c_str(), str(net_param->input_shape_size()).c_str(), str(net_param->input_dim_size()).c_str() ) );
  }

  void maybe_override_dims_and_calc_strides( dims_t & dims, map_str_uint32_t const & overrides ) {
    map_str_uint32_t check_or = overrides;
    for( uint32_t i = 0; i != dims.size(); ++i ) { 
      dims.dims(i) = get_and_rem( check_or, dims.names(i), dims.dims(i) );
    }
    if( !check_or.empty() ) { 
      // FIXME: error message is specific to current use case(s) ...
      rt_err( strprintf( "error: unused/unknown dims in in_dims. original in_dims=%s; unused_dims=%s\n", 
			 str(dims).c_str(), str(check_or).c_str() ) );
    }
    dims.calc_strides();
  }

  void add_data_img_node( p_conv_pipe_t const & conv_pipe, string const & data_img_node_name, dims_t const & data_img_node_dims ) {
    p_conv_node_t const data_img_node = conv_pipe->get_or_make_node( data_img_node_name, 0, 0 );
    conv_pipe->data_img_node_names.push_back( data_img_node_name );
    data_img_node->csi.init_as_source();
    if( !conv_pipe->data_num_imgs.v ) { conv_pipe->data_num_imgs.v = data_img_node_dims.dsz("img"); }
    if( conv_pipe->data_num_imgs.v != data_img_node_dims.dsz("img") ) { rt_err( "unhandled: multiple data layers with differing numbers of images." ); }
    data_img_node->dims = data_img_node_dims;
  }

  p_conv_pipe_t create_pipe_from_param( p_net_param_t const &net_param, map_str_uint32_t const &in_dims, 
					string const &out_node_name, bool const &add_bck_ops ) {
    // note: we only handle a (very) limited set of possible layers/networks here.
    p_conv_pipe_t conv_pipe( new conv_pipe_t );
    conv_pipe->orig_net_param = net_param; // FIXME: see note/FIXME in conv_util.H
    conv_pipe->net_state.reset( new caffe::NetState );
    if( add_bck_ops ) { conv_pipe->net_state->set_phase( caffe::TRAIN ); }
    //vect_string const & layer_names = net->layer_names();
    // note: if out_node_name == empty string, won't match anything, so all layers will be read (barring a node actually named "")
    conv_pipe->out_node_name = out_node_name; 
    bool found_out_node = 0;
    // handle 'old style' top-level input blobs by treating them as data layers
    bool input_uses_shape = 0;
    if( net_param->input_size() == net_param->input_shape_size() ) {
      if( net_param->input_dim_size() != 0 ) { net_input_err( net_param ); }
      input_uses_shape = 1;
    } else if( net_param->input_size()*4 == (net_param->input_dim_size()) ) {
      if( net_param->input_shape_size() != 0 ) { net_input_err( net_param ); }
    } else { net_input_err( net_param ); }
      
    for( int32_t in_ix = 0; in_ix != net_param->input_size(); ++in_ix ) {
      dims_t input_dims;
      if( input_uses_shape ) { input_dims = shape_to_dims_t( net_param->input_shape(in_ix) ); }
      else {
	uint32_t const num_dims = 4;
	input_dims.resize_and_zero( num_dims );
	for( uint32_t i = 0; i != num_dims; ++i ) { input_dims.dims(i) = net_param->input_dim(i + 4*in_ix); }
      } 
      if( input_dims.size() == 4 ) { input_dims.names(0) = "img"; input_dims.names(1) = "chan"; 
	input_dims.names(2) = "y"; input_dims.names(3) = "x";}
      else { printf("warning: input blob with number of dims != 4, not handled, dims will be unnamed");  }
      input_dims.tn = "float";
      maybe_override_dims_and_calc_strides( input_dims, in_dims );
      add_data_img_node( conv_pipe, net_param->input(in_ix), input_dims );
      if( add_bck_ops ) { printf("warning: add_bck_ops + top-level input blobs unhandled ... label input may be missing"); }
    }

    for( int32_t i = 0; i != net_param->layer_size(); ++i ) { 
      caffe::LayerParameter const & lp = net_param->layer(i);
      assert_st( lp.has_name() );
      assert_st( lp.has_type() );
      //printf( "to_pipe: lp.name()=%s\n", str(lp.name()).c_str() );
      if( !boda_caffe::layer_included_for_state( lp, *conv_pipe->net_state ) ) { 
	//printf( "skip lp.name()=%s\n", str(lp.name()).c_str() ); 
	continue; 
      } 
      p_conv_op_t conv_op( new conv_op_t );
      conv_op->tag = lp.name();
      conv_op->type = lp.type();
      RF_TO_VEC( conv_op->bots, lp.bottom );
      RF_TO_VEC( conv_op->tops, lp.top );

      if( 0 ) {
      } else if( lp.type() == Convolution_coi.type ) {
	assert_st( lp.has_convolution_param() );
	caffe::ConvolutionParameter const & cp = lp.convolution_param();
	fill_in_conv_op_from_param( conv_op, cp );
	assert_st( cp.num_output() >= 0 ); // should zero be allowed?
	conv_op->str_vals["out_chans"] = str(cp.num_output());
	assert_st( has( conv_op->dims_vals, "kern_sz" ) ); // FIXME: convolutions *must* specify kernel size, i think? check in caffe
	// add (make explicit) filts and biases as inputs 
	conv_op->bots.push_back( lp.name() + "_filts" );
	conv_op->bots.push_back( lp.name() + "_biases" );
	//conv_pipe->some_kinda_node_names.push_back( {filts,biases}_node_name ); // FIXME?
      } else if( lp.type() == ReLU_coi.type ) {
      } else if( lp.type() == Scale_coi.type ) { // FIXME: actually read params
      } else if( lp.type() == BatchNorm_coi.type ) { // FIXME: actually read params
      } else if( lp.type() == Dropout_coi.type ) {
	//rt_err( "TODO: handle dropout" );
	caffe::DropoutParameter const & p = lp.dropout_param();	
	conv_op->str_vals["dropout_ratio"] = str(p.dropout_ratio());
        if( !add_bck_ops ) { // if not adding bck ops, treat dropout as no-op
          if( conv_op->tops != conv_op->bots ) {
            rt_err( "UNHANDLED: non in-place dropout layer, can't ignore even for forward prop only" );
          }
          conv_op.reset();
        }
      } else if( lp.type() == LRN_coi.type ) {
	//assert_st( lp.has_lrn_param() );
	caffe::LRNParameter const & p = lp.lrn_param();	
	conv_op->str_vals["alpha"] = str(p.alpha());
	conv_op->str_vals["beta"] = str(p.beta());
	conv_op->str_vals["local_size"] = str(p.local_size());
	conv_op->str_vals["k"] = str(p.k());
      } else if( lp.type() == Softmax_coi.type ) {
	// this may be inconvieniently strong; it's probably okay to ignore this here
	//rt_err( "Saw unexpected Softmax layer in caffpb caffe->boda net conversion. should have been stripped out?" );
	printf( "Warning, Saw unexpected Softmax layer in caffpb caffe->boda net conversion. should have been stripped out? ignoring.\n" );
	conv_op.reset();
      } else if( lp.type() == SoftmaxWithLoss_coi.type ) {
	// this layer should only be present when add_bck_ops==1, and all outputs should be produced by it
	if( !add_bck_ops ) { 
	  //rt_err( "Saw unexpected SoftmaxWithLoss layer in caffpb caffe->boda net conversion given add_bck_ops==0." ); 
	  printf( "Warning, Saw unexpected SoftmaxWithLoss layer in caffpb caffe->boda net conversion given add_bck_ops==0. ignoring.\n" ); 
	  conv_op.reset();
	} else {
	  if( conv_op->tops.size() > 1 ) { rt_err( "expected 0 or 1 outputs for SoftmaxWithLoss (i.e. just a loss output or nothing (an implicit missing loss output)). saw '"+str(conv_op->tops.size())+"' outputs." ); }
	  if( conv_op->tops.size() == 0 ) { rt_err( "FIXME: restore loss auto-naming or fix prototxts"); conv_op->tops.push_back( "loss" ); } 
	  conv_op->tops.insert( conv_op->tops.begin(), conv_op->bots[0] + "_grad_loss" ); // add gradient output for fwd_top input
	}
      } else if( lp.type() == Pooling_coi.type ) {
	assert_st( lp.has_pooling_param() );
	caffe::PoolingParameter const & pp = lp.pooling_param();
	fill_in_conv_op_from_param( conv_op, pp );
	uint32_t avg_pool;
	if( pp.pool() == caffe::PoolingParameter_PoolMethod_AVE ) { avg_pool = 1; } 
	else if( pp.pool() == caffe::PoolingParameter_PoolMethod_MAX ) { avg_pool = 0; }
	else { printf( "warning: unhanded pooling method pp.pool()=%s\n", str(pp.pool()).c_str() ); }
	conv_op->str_vals["avg_pool"] = str(avg_pool);
	assert_st( has( conv_op->dims_vals, "kern_sz" ) != pp.global_pooling() ); // global pooling iff no kernel size specified
      } else if( lp.type() == InnerProduct_coi.type ) {
	assert_st( lp.has_inner_product_param() );
	caffe::InnerProductParameter const & ipp = lp.inner_product_param();
	conv_op->str_vals["out_chans"] = str(ipp.num_output());
      } else if( lp.type() == Data_coi.type ) {
	// note/FIXME: if there are multiple data layers, any values in in_dims will apply to all of them
	assert_st( lp.has_data_param() );
	caffe::DataParameter const * const dp = &lp.data_param();
	// if( hdf5 ) { data_dims = ...; } // FIXME: get dims from data layer 'better'
	uint32_t const data_dims_chan = 3;
	// if( gray ) { data_dims_chan = 1; } 
	assert_st( lp.has_transform_param() );
	caffe::TransformationParameter const * const tp = &lp.transform_param();
	dims_t data_dims( vect_uint32_t{ dp->batch_size(), data_dims_chan, tp->crop_size(), tp->crop_size() }, 
                          vect_string{ "img", "chan", "y", "x" }, "float" );
	maybe_override_dims_and_calc_strides( data_dims, in_dims );
	if( lp.bottom_size() != 0 ) { rt_err( "unhandled caffe data layer with num inputs != 0" ); }
	if( lp.top_size() != 2 ) { rt_err( "unhandled caffe data layer with num outputs != 2" ); }
	conv_op.reset(); // for now, don't add an op for data layers.
	// but, create the outputs as source nodes, and set chans and set info for them
	add_data_img_node( conv_pipe, lp.top(0), data_dims );
	if( add_bck_ops ) {
	  string const data_label_node_name = lp.top(1);
	  p_conv_node_t const data_label_node = conv_pipe->get_or_make_node(data_label_node_name, 0, 0 );
	  conv_pipe->data_label_node_names.push_back( data_label_node_name );
	  assert( !data_label_node->csi.valid() );
	  data_label_node->csi.support_sz = u32_pt_t(1,1);
	  data_label_node->csi.support_stride = u32_pt_t(1,1);
	}
      } else if( lp.type() == Accuracy_coi.type ) {
	conv_op.reset(); // for now, just silently ignore acc layers.
      } else if( lp.type() == Concat_coi.type ) { // no parameters to get
      } else if( lp.type() == Eltwise_coi.type ) { // FIXME: actually read params
      } else {
	conv_op.reset(); printf( "warning: ignoring layer with lp.type()=%s\n", str(lp.type()).c_str() );
      }

      bool layer_has_out_node = 0;
      for( int32_t i = 0; i != lp.top_size(); ++i ) {
	if( out_node_name == lp.top(i) ) { layer_has_out_node = 1; found_out_node = 1;}
      }
      if( found_out_node ) { if( !layer_has_out_node ) { break; } } 
      if( conv_op ) { conv_pipe->add_conv( conv_op ); }      
    }
    // FIXME? this is too strong now, and will be checked later -- but check something here? can't?
    //if( !found_out_node ) { rt_err( strprintf("node out_node_name=%s not found as layer output in network\n",str(out_node_name).c_str() )); }
    if( add_bck_ops ) { conv_pipe->add_bck_ops(); }
    conv_pipe->calc_support_info( 1 );
    conv_pipe->calc_dims();
    return conv_pipe;
  }
#undef RF_TO_VEC

  p_net_param_t parse_and_upgrade_net_param_from_text_file( filename_t const & ptt_fn ) {
    p_string ptt_str = read_whole_fn( ptt_fn );
    p_net_param_t net_param( new net_param_t );
    bool const ret = google::protobuf::TextFormat::ParseFromString( *ptt_str, net_param.get() );
    assert_st( ret );
    boda_caffe::UpgradeNetAsNeeded( ptt_fn.exp, net_param.get() );
    return net_param;
  }
  
  uint32_t parse_datum_into( p_nda_float_t & out, uint32_t const out_ix, void const * const bytes, uint32_t const bytes_sz ) {
    caffe::Datum caffe_datum;
    bool const cdp_ret = caffe_datum.ParseFromArray( bytes, bytes_sz );
    assert_st( cdp_ret );

    bool const has_data = caffe_datum.has_data();
    bool const has_float_data = caffe_datum.float_data_size();
    if( has_data + has_float_data != 1 ) {
      rt_err( strprintf( "datum must have exactly 1 of data and float_data, but: has_data=%s has_float_data=%s\n", 
			 str(has_data).c_str(), str(has_float_data).c_str() ) );
    }
    if( has_float_data ) { rt_err( "TODO: datum has float_data handling." ); }
    if( caffe_datum.encoded() ) { rt_err( "TODO: datum encoded=1 handling." ); }
    if( !caffe_datum.has_channels() ) { rt_err( "datum missing channels field" ); }
    if( !caffe_datum.has_height() ) { rt_err( "datum missing height field" ); }
    if( !caffe_datum.has_width() ) { rt_err( "datum missing width field" ); }

    uint32_t const chans = caffe_datum.channels();
    uint32_t const hi = caffe_datum.width();
    uint32_t const wi = caffe_datum.height();

    if( !out ) { out.reset( new nda_float_t( dims_t( { 1, chans, hi, wi }, "float" ) ) ); }

    assert( out->dims.sz() == 4 );
    assert_st( out_ix < out->dims.dims(0) );

    assert_st( chans == out->dims.dims(1) );
    uint32_t const ho = out->dims.dims(2);
    uint32_t const wo = out->dims.dims(3);

    assert_st( ho <= hi );
    assert_st( wo <= wi );

    // take center crop
    uint32_t xi = (wi - wo) >> 1;
    uint32_t yi = (hi - ho) >> 1;

    assert_st( (ho+yi) <= hi );
    assert_st( (wo+xi) <= wi );

    float * const ret_data = out->elems_ptr() + (out_ix * (chans*wo*ho));

    if( chans*hi*wi != caffe_datum.data().size() ) {
      rt_err( strprintf( "inconsistency in datum data size vs datum dims: chans=%s hi=%s wi=%s so chans*h*w=%s but caffe_datum.data().size()=%s\n", 
			 str(chans).c_str(), str(hi).c_str(), str(wi).c_str(), str(chans*hi*wi).c_str(), str(caffe_datum.data().size()).c_str() ) );
    }
    uint8_t const * const caffe_data = (uint8_t const *)(&caffe_datum.data()[0]);

    for( uint32_t c = 0; c < chans; ++c ) {
      for( uint32_t y = 0; y < wo; ++y ) {
	for( uint32_t x = 0; x < ho; ++x ) {
	  uint8_t const v = caffe_data[(c * hi + yi + y ) * wi + xi + x];
	  // note: data is assumed to be in BGR format
	  ret_data[ ( c * ho + y ) * wo + x ] = v - float(uint8_t(u32_bgra_inmc >> (c*8)));
	}
      }
    }
    return caffe_datum.label();
  }

  p_datum_t parse_datum( void const * const bytes, uint32_t const bytes_sz ) {
    p_datum_t datum( new datum_t );
    datum->label = parse_datum_into( datum->val, 0, bytes, bytes_sz );
    return datum;
  }

  uint8_t float_to_pel( float const & v ) { 
    int32_t ret = nearbyintl(v);
    //return std::min(255,std::max(0,ret)); // allows de-normalized values 
    assert_st( ret >= 0 );
    assert_st( ret <= 255 );
    return ret;
  }
  // currently only for debugging / display
  p_img_t datum_to_img( p_datum_t datum ) {
    p_nda_float_t const & v = datum->val;
    assert_st( v->dims.sz() == 4 );
    assert_st( v->dims.dims(0) == 1 ); // only handling BGR here
    assert_st( v->dims.dims(1) == 3 ); // only handling BGR here

    uint32_t const h = v->dims.dims(2);
    uint32_t const w = v->dims.dims(3);
    
    p_img_t ret( new img_t );
    ret->set_sz_and_alloc_pels( {w,h} );

    float const * const vd = v->elems_ptr();

    uint32_t const c_off = w*h;
    for( uint32_t y = 0; y != h; ++y ) {
      for( uint32_t x = 0; x < w; ++x ) {
	uint32_t const pix = y*w + x;
	uint32_t const rgba_val = rgba_to_pel( float_to_pel(vd[pix+c_off*2]+inmc_r), 
					       float_to_pel(vd[pix+c_off]+inmc_g), 
					       float_to_pel(vd[pix]+inmc_b) );
	ret->set_pel( {x, y}, rgba_val );
      }
    }
    return ret;
  }



  struct cnet_ana_t : virtual public nesi, public has_main_t // NESI(help="show info from caffe prototxt net. ",bases=["has_main_t"], type_id="cnet_ana")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t ptt_fn; //NESI(default="%(models_dir)/%(in_model)/train_val.prototxt",help="input net prototxt template filename")
    string out_node_name;//NESI(default="",help="trim off network after named layer (note: keeps whole network if empty string).")
    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="text output filename")
    map_str_uint32_t in_dims; //NESI(default="()",help="input dims. any non-present dims will use values derived from the (single) data layer.")
    p_uint32_t out_sz; //NESI(help="calculate sizes at all layers for the given output size and dump pipe")
    uint32_t ignore_padding_for_sz; //NESI(default=0,help="if 1, ignore any padding specified when calculating the sizes at each layer for the in_sz or out_sz options")
    uint32_t print_ops; //NESI(default=0,help="if non-zero, write ops to file with fn given by print_opts_fn.")
    filename_t print_ops_fn; //NESI(default="%(boda_output_dir)/out.py",help="print_opts output filename")
    uint32_t add_bck_ops; //NESI(default=0,help="if non-zero, add bck (aka backwards/backprop/gradients) operations.")
    
    virtual void main( nesi_init_arg_t * nia ) { 
      p_ofstream out = ofs_open( out_fn.exp );
      p_net_param_t net_param = parse_and_upgrade_net_param_from_text_file( ptt_fn );
      p_conv_pipe_t conv_pipe = create_pipe_from_param( net_param, in_dims, out_node_name, add_bck_ops );
      //(*out) << convs << "\n";
      conv_pipe->dump_pipe( *out ); 
      if( in_dims.size() ) { 
	(*out) << ">> calculating network sizes forward given an in_dims of " << in_dims << "\n";
	conv_pipe->dump_ios( *out ); 
      }
      if( print_ops ) {	conv_pipe->dump_ops( *ofs_open( print_ops_fn.exp ) ); }
      if( out_sz ) { 
	(*out) << ">> calculating network sizes backward given an out_sz of " << *out_sz << "\n";
	conv_pipe->calc_sizes_back( u32_pt_t( *out_sz, *out_sz ), ignore_padding_for_sz ); 
	conv_pipe->dump_ios( *out ); 
      }
    }
  };


  p_net_param_t must_read_binary_proto( filename_t const & fn, filename_t const & alt_fn ) {
    p_net_param_t net( new net_param_t );
    filename_t load_fn = fn;
    if( !alt_fn.in.empty() ) { load_fn = ensure_one_is_regular_file( fn, alt_fn ); }
    p_istream is = ifs_open(load_fn);
    google::protobuf::io::IstreamInputStream iis( is.get() );
    google::protobuf::io::CodedInputStream cis( &iis );
    cis.SetTotalBytesLimit( int32_t_const_max, 536870912 );
    bool const ret = net->ParseFromCodedStream( &cis );
    if( !ret ) { rt_err( strprintf( "failed to parse Netparamter from binary prototxt file %s", 
				    str(load_fn.exp).c_str() ) ); }
    boda_caffe::UpgradeNetAsNeeded( load_fn.exp, net.get() );
    return net;
  }

  uint32_t maybe_get_layer_ix( net_param_t const & net_param, string const & layer_name ) {
    for( int i = 0; i != net_param.layer_size(); ++i ) { if( net_param.layer(i).name() == layer_name ) { return i; } }
    return uint32_t_const_max;
  }
  uint32_t get_layer_ix( net_param_t const & net_param, string const & layer_name ) {
    uint32_t const ret = maybe_get_layer_ix( net_param, layer_name );
    if( ret == uint32_t_const_max ) { rt_err( strprintf("layer layer_name=%s not found in network\n",str(layer_name).c_str() )); }
    return ret;
  }

  void alloc_layer_blobs( p_conv_pipe_t const & pipe, string const & layer_name, vect_p_nda_float_t & blobs ) {
    p_conv_op_t const & cop = pipe->get_op( layer_name );
    if( cop->is( Convolution_coi ) ) { 
      blobs.push_back( p_nda_float_t( new nda_float_t( pipe->must_get_node( cop->bots[1] )->dims ) ) ); // filts
      blobs.push_back( p_nda_float_t( new nda_float_t( pipe->must_get_node( cop->bots[2] )->dims ) ) ); // biases
    } else {
      rt_err( "don't know how to alloc blobs for layer of type" + cop->type );
    }
  }

  void copy_layer_blobs( caffe::LayerParameter const & dest_lp, vect_p_nda_float_t & blobs ) {
    timer_t t("caffe_copy_layer_blob_data");
    blobs.clear();
    for( uint32_t bix = 0; bix < (uint32_t)dest_lp.blobs_size(); ++bix ) {
      caffe::BlobProto const & lbp = dest_lp.blobs( bix );
      dims_t blob_dims;
      if( lbp.has_num() || lbp.has_channels() || lbp.has_height() || lbp.has_width() ) {
	blob_dims.resize_and_zero( 4 );
	blob_dims.dims(3) = lbp.width();
	blob_dims.dims(2) = lbp.height();
	blob_dims.dims(1) = lbp.channels();
	blob_dims.dims(0) = lbp.num();
      } else { blob_dims = shape_to_dims_t( lbp.shape() ); }
      blob_dims.tn = "float"; // FIXME_TNDA: it's probably okay to assume caffe always uses float?
      blob_dims.calc_strides();
      p_nda_float_t blob( new nda_float_t( blob_dims ) );
      assert_st( blob->elems_sz() == uint32_t(lbp.data_size()) );
      float * const dest = blob->elems_ptr();
      float const * const src = lbp.data().data();
      for( uint32_t i = 0; i != blob->elems_sz() ; ++i ) { dest[i] = src[i]; }
      blobs.push_back( blob );
    }

    // add dim names (and maybe do other fixups) for specific known layer types
    if( dest_lp.type() == Convolution_coi.type ) {
      assert( blobs.size() == 2 ); // filts, biases
      dims_t & fd = blobs.at(0)->dims;
      // for filter blobs, assume they are in the following format:
      assert_st( fd.sz() == 4 );
      fd.names(0) = "out_chan"; fd.names(1) = "in_chan"; fd.names(2) = "y"; fd.names(3) = "x";

      dims_t & bd = blobs.at(1)->dims;
      // for 'old style' bias blobs, squwish out leading size 1 dims
      if( bd.sz() == 4 ) {
	for( uint32_t i = 0; i != bd.sz()-1; ++i ) { assert_st( bd.dims(i) == 1 ); }
	bd = dims_t( vect_uint32_t{ bd.dims(3) }, bd.tn );
      }
      assert( bd.sz() == 1 );
      bd.names(0) = "out_chan";
    }

  }
  void copy_layer_blobs( p_net_param_t const & net, uint32_t const & layer_ix, vect_p_nda_float_t & blobs ) {
    assert_st( layer_ix < (uint32_t)net->layer_size() );
    caffe::LayerParameter const & dest_lp = net->layer( layer_ix );
    copy_layer_blobs( dest_lp, blobs );
  }
  void copy_layer_blobs( p_net_param_t const & net, string const & layer_name, vect_p_nda_float_t & blobs ) {
    uint32_t const layer_ix = get_layer_ix( *net, layer_name );
    copy_layer_blobs( net, layer_ix, blobs );
  }

  void set_layer_blobs( p_net_param_t const & net, uint32_t const & layer_ix, vect_p_nda_float_t & blobs ) {
    timer_t t("caffe_set_layer_blob_data");
    assert_st( layer_ix < (uint32_t)net->layer_size() );
    caffe::LayerParameter & dest_lp = *net->mutable_layer(layer_ix);
    dest_lp.clear_blobs();
    for( uint32_t bix = 0; bix < blobs.size(); ++bix ) {
      p_nda_float_t const & blob = blobs[bix];
      dims_t const & blob_dims = blob->dims;
      caffe::BlobProto & lbp = *dest_lp.add_blobs();
      caffe::BlobShape & lbp_shape = *lbp.mutable_shape();
      dims_t_to_shape( blob_dims, lbp_shape );
      assert( lbp.data_size() == 0 );
      const float * const src = blob->elems_ptr();
      for( uint32_t i = 0; i != blob->elems_sz() ; ++i ) { lbp.add_data( src[i] ); }
    }
  }
  void set_layer_blobs( p_net_param_t const & net, string const & layer_name, vect_p_nda_float_t & blobs ) {
    uint32_t const layer_ix = get_layer_ix( *net, layer_name );
    set_layer_blobs( net, layer_ix, blobs );
  }

  // we iterate of dest, and for every layer with a matching name found in src, we copy the blobs from src->dest
  void copy_matching_layer_blobs_from_param_to_param( p_net_param_t const & src, p_net_param_t const & dest ) {
    for( int i = 0; i != dest->layer_size(); ++i ) { 
      caffe::LayerParameter & dest_lp = *dest->mutable_layer(i);
      uint32_t const src_lix = maybe_get_layer_ix( *src, dest_lp.name() );
      if( src_lix == uint32_t_const_max ) { continue; } // layer not found in src
      caffe::LayerParameter const & src_lp = src->layer(src_lix);
      dest_lp.clear_blobs();
      for( int j = 0; j != src_lp.blobs_size(); ++j ) { *dest_lp.add_blobs() = src_lp.blobs(j); }
    }
  }
  void copy_matching_layer_blobs_from_param_to_pipe( p_net_param_t const & blob_src, p_conv_pipe_t const & cp ) {
    for( map_str_p_conv_op_t::const_iterator i = cp->convs->begin(); i != cp->convs->end(); ++i ) {
      uint32_t const blob_src_lix = maybe_get_layer_ix( *blob_src, i->first );
      if( blob_src_lix == uint32_t_const_max ) { continue; } // layer not found in src
      caffe::LayerParameter const & blob_src_lp = blob_src->layer(blob_src_lix);
      p_vect_p_nda_float_t blob_src_blobs( new vect_p_nda_float_t );
      copy_layer_blobs( blob_src_lp, *blob_src_blobs );
      cp->add_layer_blobs( i->first, blob_src_blobs );
    }
  }

  void create_identity_weights( p_net_param_t net, p_conv_pipe_t pipe, string const & layer_name, uint32_t const noise_mode ) {
    if( noise_mode >= 2 ) { rt_err( strprintf( "unsupported noise_mode=%s\n", str(noise_mode).c_str() ) ); }
    vect_p_nda_float_t blobs;

    alloc_layer_blobs( pipe, layer_name, blobs );
    //copy_layer_blobs( net, layer_name, blobs );
    assert_st( blobs.size() == 2 ); // filters, biases
    p_nda_float_t biases = blobs[1];
    for( dims_iter_t di( biases->dims ) ; ; ) { biases->at(di.di) = 0; if( !di.next() ) { break; } } // all biases 0
    p_nda_float_t filts = blobs[0];

    assert_st( filts->dims.sz() == 4 );
    uint32_t const width = filts->dims.dims(3);
    uint32_t const height = filts->dims.dims(2); 
    uint32_t const channels = filts->dims.dims(1);
    uint32_t const num = filts->dims.dims(0);

    assert_st( channels == num ); // for now, only handling case where input chans == output chans

    // it's unclear how to handle even width/height, depending on padding in particular
    assert_st( width & 1 );
    assert_st( height & 1 );

    //for( uint32_t i = 0; i != num; ++i ) { filts->at4( i, i, (h+1)/2, (w+1)/2 ) = 1; }
    uint32_t const num_inputs = width*height*channels; // for adding xavier noise
    float const xavier_noise_mag = 3.0l / double( num_inputs );
    boost::random::mt19937 rand_gen;
    boost::random::uniform_real_distribution<> const xavier_noise_dist( -xavier_noise_mag, xavier_noise_mag );
    for( dims_iter_t di( filts->dims ) ; ; ) { 
      float val = 0; 
      if( noise_mode == 1 ) { val += xavier_noise_dist(rand_gen); }
      if( (di.di[2] == (height/2)) && // center y pel in filt
	  (di.di[3] == (width/2)) && // center x pel in filt
	  (di.di[0] == di.di[1]) ) // in_chan == out_chan
      { val += 1; }

      filts->at(di.di) = val;
      if( !di.next() ) { break; } 
    }    

    set_layer_blobs( net, layer_name, blobs );
  }

  void resize_1d( float const * const in, uint32_t const & in_sz, float * const out, uint32_t const & out_sz ) {
    for( uint32_t i = 0; i != out_sz; ++i ) { out[i] = 0.0; }
    double const scale = double(out_sz) / in_sz;
    for( uint32_t i = 0; i != in_sz; ++i ) {
      float const v = in[i];
      // calc range of out for in_sz
      double const ob = double(out_sz) * i / in_sz;
      double const oe = double(out_sz) * (i+1) / in_sz;
      for( uint32_t o = floor(ob); o != ceil(oe); ++o ) {
	double const span = 1.0 - ((o<ob)?(ob - o):0) - ((oe<(o+1))?(o + 1 - oe):0);
	assert(o < out_sz);
	out[o] += v*span/scale;
      }
    }
  }

  void print_kernel( p_nda_float_t const & in, uint32_t const i, uint32_t const j ) {
    u32_pt_t const in_ksz = {in->dims.dims(3),in->dims.dims(2)};
    printf("kernel\n");
    float * kernel = &in->at2(i,j);
    for( uint32_t y = 0; y != in_ksz.d[1]; ++y ) { 
      for( uint32_t x = 0; x != in_ksz.d[0]; ++x ) { 
	printf("  % 02.3f", kernel[y*in_ksz.d[0]+x] );
      }
      printf("\n");
    }
    printf("\n");
    
  }

  void resize_kernel( p_nda_float_t const & in, p_nda_float_t const & out ) {
    
    // coiterate over outer dims
    assert_st( in->dims.dims(0) == out->dims.dims(0) );
    assert_st( in->dims.dims(1) == out->dims.dims(1) );
    u32_pt_t const in_ksz = {in->dims.dims(3),in->dims.dims(2)};
    u32_pt_t const out_ksz = {out->dims.dims(3),out->dims.dims(2)};

    printf( "in_ksz=%s out_ksz=%s\n", str(in_ksz).c_str(), str(out_ksz).c_str() );

    vect_float kbuf;
    kbuf.resize( in_ksz.d[1]*out_ksz.d[0] );
    vect_float kbuf2;
    kbuf2.resize( in_ksz.d[1] );
    vect_float kbuf3;
    kbuf3.resize( out_ksz.d[1], 0 );
    
    for( uint32_t i = 0; i != in->dims.dims(0); ++i ) {
      for( uint32_t j = 0; j != in->dims.dims(1); ++j ) {
	//print_kernel( in, i, j );
	for( uint32_t y = 0; y != in_ksz.d[1]; ++y ) { resize_1d( &in->at3(i,j,y), in_ksz.d[0], &kbuf[y*out_ksz.d[0]], out_ksz.d[0] ); }
	for( uint32_t x = 0; x != out_ksz.d[0]; ++x ) { 
	  for( uint32_t y = 0; y != in_ksz.d[1]; ++y ) { kbuf2[y] = kbuf[y*out_ksz.d[0] + x]; }
	  resize_1d( &kbuf2[0], in_ksz.d[1], &kbuf3[0], out_ksz.d[1] );
	  for( uint32_t y = 0; y != out_ksz.d[1]; ++y ) { out->at4(i,j,y,x) = kbuf3[y]; }
	}
	//print_kernel( out, i, j );
      }
    }
  } 


  struct cnet_bpt_dump_t : virtual public nesi, public has_main_t // NESI(help="base class for utilities to modify caffe nets",
			   // bases=["has_main_t"], type_id="cnet_bpt_dump" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t trained_fn; //NESI(default="%(models_dir)/%(in_model)/best.caffemodel",help="input trained net from which to copy params")
    uint32_t remove_data; //NESI(default=1,help="if non-zero, remove data fields from blobs")

    p_net_param_t trained_net;

    void main( nesi_init_arg_t * nia ) { 
      trained_net = must_read_binary_proto( trained_fn, filename_t() );

      uint32_t const numl = (uint32_t)trained_net->layer_size();
      if( remove_data ) { // if requested, delete all blob data
	for( uint32_t i = 0; i != numl; ++i ) {
	  caffe::LayerParameter & lp = *trained_net->mutable_layer(i);
	  for( int j = 0; j != lp.blobs_size(); ++j ) { 
	    caffe::BlobProto & lbp = *lp.mutable_blobs( j );
	    lbp.clear_data();
	  }
	}
      }
      // dump to string
      string trained_str;
      bool const pts_ret = google::protobuf::TextFormat::PrintToString( *trained_net, &trained_str );
      assert_st( pts_ret );
      printstr( trained_str );
    }
  };

  struct cnet_mod_t : virtual public nesi // NESI(help="base class for utilities to modify caffe nets" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t ptt_fn; //NESI(default="%(models_dir)/%(in_model)/train_val.prototxt",help="input net prototxt template filename")
    filename_t trained_fn; //NESI(default="%(models_dir)/%(in_model)/best.caffemodel",help="input trained net from which to copy params")
    filename_t mod_fn; //NESI(default="%(models_dir)/%(out_model)/train_val.prototxt",help="output net prototxt template filename")
    filename_t mod_weights_fn; //NESI(default="%(models_dir)/%(out_model)/boda_gen.caffemodel",help="output net weights binary prototxt template filename")
    map_str_uint32_t in_dims; //NESI(default="()",help="input dims. any non-present dims will use values derived from the (single) data layer. (for this mode, specifing this is generally unneeded and may yield invalid results if used and the dim sizes specified differ from those in the data layer)" )
    p_net_param_t net_param;
    p_conv_pipe_t net_pipe;

    p_net_param_t mod_net_param;
    p_conv_pipe_t mod_net_pipe;
    p_net_param_t trained_net;

    void ensure_out_dir( nesi_init_arg_t * const nia ) { ensure_is_dir( nesi_filename_t_expand( nia, "%(models_dir)/%(out_model)" ), 1 ); }
    void create_net_params( void ) {
      net_param = parse_and_upgrade_net_param_from_text_file( ptt_fn );
      net_pipe = create_pipe_from_param( net_param, in_dims, "", 0 );
      mod_net_param.reset( new net_param_t( *net_param ) ); // start with copy of net_param
    }
    void write_mod_pt( void ) {
      string mod_str;
      bool const pts_ret = google::protobuf::TextFormat::PrintToString( *mod_net_param, &mod_str );
      assert_st( pts_ret );
      write_whole_fn( mod_fn, mod_str );
      // assuming the mod net pt is now finalized, create the pipe for it
      mod_net_pipe = create_pipe_from_param( mod_net_param, in_dims, "", 0 );
    }
    void load_nets( void ) {
      trained_net = must_read_binary_proto( trained_fn, filename_t() );
      copy_matching_layer_blobs_from_param_to_param( trained_net, net_param );
      copy_matching_layer_blobs_from_param_to_param( trained_net, mod_net_param );
    }
    void write_mod_net( void ) {
      bool const ret = mod_net_param->SerializeToOstream( ofs_open( mod_weights_fn ).get() );
      if( !ret ) { rt_err( strprintf( "failed to write NetParamter to binary prototxt file %s", 
				      str(mod_weights_fn.exp).c_str() ) ); }
    }
  };

  struct cnet_copy_t : virtual public nesi, public cnet_mod_t, public has_main_t // NESI(help="utility to modify caffe nets",
		       // bases=["cnet_mod_t","has_main_t"], type_id="cnet_copy")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    void main( nesi_init_arg_t * nia ) { 
      create_net_params();
      ensure_out_dir( nia );
      write_mod_pt();
      load_nets();
      write_mod_net();
    }
  };


  void create_upsamp_layer_weights( p_conv_pipe_t const & conv_pipe, string const & cpln, 
				    p_conv_pipe_t const & conv_pipe_upsamp, string const & cpuln ) {
    p_vect_p_nda_float_t usl_blobs = must_find( *conv_pipe->layer_blobs, cpln );

    p_vect_p_nda_float_t usl_blobs_upsamp( new vect_p_nda_float_t );
    alloc_layer_blobs( conv_pipe_upsamp, cpuln, *usl_blobs_upsamp );

    assert_st( usl_blobs->size() == 2 ); // filters, biases
    assert_st( usl_blobs_upsamp->size() == 2 ); // filters, biases
    assert_st( usl_blobs->at(1)->dims == usl_blobs_upsamp->at(1)->dims ); // biases should be same shape (and same strides?)
    usl_blobs_upsamp->at(1) = usl_blobs->at(1); // use biases unchanged in upsamp net
    assert_st( usl_blobs->at(0)->dims.dims(0) == usl_blobs_upsamp->at(0)->dims.dims(0) );
    assert_st( usl_blobs->at(0)->dims.dims(1) == usl_blobs_upsamp->at(0)->dims.dims(1) );
    assert_st( u32_ceil_div( usl_blobs->at(0)->dims.dims(2), 2 ) == usl_blobs_upsamp->at(0)->dims.dims(2) );
    assert_st( u32_ceil_div( usl_blobs->at(0)->dims.dims(3), 2 ) == usl_blobs_upsamp->at(0)->dims.dims(3) );

    for( dims_iter_t di( usl_blobs_upsamp->at(0)->dims ) ; ; ) { usl_blobs_upsamp->at(0)->at(di.di) = 0; 
      if( !di.next() ) { break; } 
    }

    for( dims_iter_t di( usl_blobs->at(0)->dims ) ; ; ) { 
      usl_blobs_upsamp->at(0)->at4(di.di[0],di.di[1],di.di[2]>>1,di.di[3]>>1) += usl_blobs->at(0)->at( di.di );
      if( !di.next() ) { break; } 
    }
    conv_pipe_upsamp->add_layer_blobs( cpuln, usl_blobs_upsamp );
  }


  struct cnet_resize_conv_t : virtual public nesi, public cnet_mod_t, public has_main_t // NESI(help="utility to modify caffe nets",
		       // bases=["cnet_mod_t","has_main_t"], type_id="cnet_resize_conv")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string to_resize_ln;//NESI(default="conv1",help="name of conv layer to resize ")    
    u32_pt_t targ_sz; //NESI(default="5 5",help="kernel size for resized layer")

    void resize_conv_weights( string const & to_resize_name ) {
      vect_p_nda_float_t blobs;
      copy_layer_blobs( net_param, to_resize_name, blobs );

      vect_p_nda_float_t blobs_mod;
      alloc_layer_blobs( mod_net_pipe, to_resize_name + "-resized", blobs_mod );

      assert_st( blobs.size() == 2 ); // filters, biases
      assert_st( blobs_mod.size() == 2 ); // filters, biases
      // assert_st( blobs[1]->dims == blobs_mod[1]->dims ); // biases should be same shape (and same strides?) too strong
      assert_st( blobs[1]->dims.dims_prod() == blobs_mod[1]->dims.dims_prod() );
      blobs_mod[1]->copy_elems_with_reshape( *blobs[1] ); // reshape

      assert_st( blobs[0]->dims.dims(0) == blobs_mod[0]->dims.dims(0) );
      assert_st( blobs[0]->dims.dims(1) == blobs_mod[0]->dims.dims(1) );
      assert_st( targ_sz.d[1] == blobs_mod[0]->dims.dims(2) );
      assert_st( targ_sz.d[0] == blobs_mod[0]->dims.dims(3) );
      resize_kernel( blobs[0], blobs_mod[0] );
      set_layer_blobs( mod_net_param, to_resize_name + "-resized", blobs_mod );
    }

    void main( nesi_init_arg_t * nia ) { 
      create_net_params();

      uint32_t const to_resize_ix = get_layer_ix( *net_param, to_resize_ln );
      caffe::LayerParameter * lp = mod_net_param->mutable_layer(to_resize_ix);
      if( !lp->has_convolution_param() ) { 
	rt_err( strprintf("layer %s of net not conv layer; don't know how to resize",to_resize_ln.c_str())); }
      caffe::ConvolutionParameter * cp = lp->mutable_convolution_param();
      p_conv_op_t conv_op( new conv_op_t );
      fill_in_conv_op_from_param( conv_op, *cp );
      set_xy_dims( conv_op->dims_vals["kern_sz"], targ_sz );

      set_param_from_conv_op( *cp, conv_op );
      assert_st( lp->has_name() );
      lp->set_name( lp->name() + "-resized" );

      uint32_t const numl = (uint32_t)mod_net_param->layer_size();
      // find and rename all fc layers
      for( uint32_t i = to_resize_ix + 1; i != numl; ++i ) {
	caffe::LayerParameter * lp = mod_net_param->mutable_layer(i);
	if( lp->type() == InnerProduct_coi.type ) {
	  // FIXME: convert to conv layer. for now, just rename.
	  printf("WARNING: renaming fc/InnerProduct %s layer to avoid size mismatch when loading weights. note that the renamed layer in the output model will *not* get any copied weights from the input model!\n",lp->name().c_str()); 
	  lp->set_name( lp->name() + "-renamed-due-to-resize" );
	} 
      }
      ensure_out_dir( nia );
      write_mod_pt();
      load_nets();
      resize_conv_weights( to_resize_ln );
      write_mod_net();
    }
  };

  // FIXME: we might want to enhance cnet_fc_to_conv_t so that the weights file is optional.
  struct cnet_fc_to_conv_t : virtual public nesi, public cnet_mod_t, public has_main_t // NESI(help="utility to modify caffe nets",
			     // bases=["cnet_mod_t","has_main_t"], type_id="cnet_fc_to_conv")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    void main( nesi_init_arg_t * nia ) { 
      create_net_params();
      trained_net = must_read_binary_proto( trained_fn, filename_t() ); // we need to load the original weights 'early' to infer input dims

      vect_string converted_layer_names;
      uint32_t const numl = (uint32_t)mod_net_param->layer_size();
      // find and rename all fc layers
      for( uint32_t i = 0; i != numl; ++i ) {
	caffe::LayerParameter * lp = mod_net_param->mutable_layer(i);
	if( lp->type() != InnerProduct_coi.type ) { continue; }
	vect_p_nda_float_t blobs;
	copy_layer_blobs( trained_net, lp->name(), blobs );

	caffe::InnerProductParameter * ipp = lp->mutable_inner_product_param();
	converted_layer_names.push_back( lp->name() );
	lp->set_name( lp->name() + "-conv" );
	lp->set_type( Convolution_coi.type );

	assert_st( !lp->has_convolution_param() );
	caffe::ConvolutionParameter * cp = lp->mutable_convolution_param();
	assert_st( ipp->has_num_output() );
	if( ipp->has_num_output() ) { cp->set_num_output( ipp->num_output() ); }
	if( ipp->has_bias_term() ) { cp->set_bias_term( ipp->bias_term() ); }
	if( ipp->has_weight_filler() ) { *cp->mutable_weight_filler() = ipp->weight_filler(); }
	if( ipp->has_bias_filler() ) { *cp->mutable_bias_filler() = ipp->bias_filler(); }

	assert_st( blobs.size() == 2 ); // filters, biases
	//printf( "lp->name()=%s\n", str(lp->name()).c_str() );
	//printf( "net_param->mutable_layer(i)->name()=%s\n", str(net_param->mutable_layer(i)->name()).c_str() );
	printf( "blobs[0]->dims=%s\n", str(blobs[0]->dims).c_str() );
	// FIXME: it's not clear what versions of blob format are possible here or which we want to support ...
	uint32_t num_w = 0;
	if( blobs[0]->dims.sz() == 4 ) {
	  if( blobs[0]->dims.dims(0) == ipp->num_output() ) {
	    assert_st( blobs[0]->dims.dims(0) == ipp->num_output() );
	    num_w = blobs[0]->dims.dims(1);
	    assert_st( blobs[0]->dims.dims(2) == 1 );
	    assert_st( blobs[0]->dims.dims(3) == 1 );
	  } else if ( blobs[0]->dims.dims(2) == ipp->num_output() ) {
	    assert_st( blobs[0]->dims.dims(0) == 1 );
	    assert_st( blobs[0]->dims.dims(1) == 1 );
	    assert_st( blobs[0]->dims.dims(2) == ipp->num_output() );
	    num_w = blobs[0]->dims.dims(3);
	  } else {
	    rt_err( strprintf( "unknown IP blobs dim layout: blobs[0]->dims=%s\n", str(blobs[0]->dims).c_str() ).c_str() );
	  }
	} else if( blobs[0]->dims.sz() == 2 ) {
	  assert_st( blobs[0]->dims.dims(0) == ipp->num_output() );
	  num_w = blobs[0]->dims.dims(1);
	} else {
	  rt_err( strprintf( "unknown/unhandled IP blobs dim layout, dims.sz() not 2 or 4: blobs[0]->dims=%s\n", 
			     str(blobs[0]->dims).c_str() ).c_str() );
	}

	// get number of input chans
	if( lp->bottom_size() != 1) { rt_err( "unhandled: bottom_size() != 1"); }
	string const bot_bn = lp->bottom(0);
	uint32_t const num_in_chan = net_pipe->must_get_node( bot_bn )->dims.dsz("chan");

	// FIXME: we assume input is spactially square, which may not be true
	assert_st( !(num_w % num_in_chan) );
	uint32_t kern_sz = sqrt(num_w / num_in_chan);
	assert_st( kern_sz*kern_sz*num_in_chan == num_w );
	cp->add_kernel_size( kern_sz );
	lp->clear_inner_product_param();
      }
      ensure_out_dir( nia );
      write_mod_pt();
      copy_matching_layer_blobs_from_param_to_param( trained_net, mod_net_param );
      //mod_net = caffe_create_net( *mod_net_param, trained_fn.exp );
      for( vect_string::const_iterator i = converted_layer_names.begin(); i != converted_layer_names.end(); ++i ) {
	fc_weights_to_conv_weights( *i );
      }
      write_mod_net();
    }

    void fc_weights_to_conv_weights( string const & layer_name ) {
      vect_p_nda_float_t blobs;
      copy_layer_blobs( trained_net, layer_name, blobs );

      vect_p_nda_float_t blobs_mod;
      alloc_layer_blobs( mod_net_pipe, layer_name + "-conv", blobs_mod );

      assert_st( blobs.size() == 2 ); // filters, biases
      assert_st( blobs_mod.size() == 2 ); // filters, biases
      printf( "blobs[1]->dims=%s blobs_mod[1]->dims=%s\n", str(blobs[1]->dims).c_str(), str(blobs_mod[1]->dims).c_str() );
      // assert_st( blobs[1]->dims == blobs_mod[1]->dims ); // biases should be same shape (and same strides?) too strong
      assert_st( blobs[1]->dims.dims_prod() == blobs_mod[1]->dims.dims_prod() );
      blobs_mod[1]->copy_elems_with_reshape( *blobs[1] ); // reshape

      assert( blobs_mod[0]->dims.dims_prod() == blobs[0]->dims.dims_prod() );
      assert( blobs_mod[0]->elems_sz() == blobs[0]->elems_sz() );
      blobs_mod[0]->copy_elems_with_reshape( *blobs[0] ); // reshape

      set_layer_blobs( mod_net_param, layer_name + "-conv", blobs_mod );
    }

  };

  struct cnet_util_t : virtual public nesi, public cnet_mod_t, public has_main_t // NESI(help="utility to modify caffe nets",
		       // bases=["cnet_mod_t","has_main_t"], type_id="cnet_util")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string add_before_ln;//NESI(default="conv4",help="name of layer before which to add identity layer")    
    uint32_t noise_mode; //NESI(default=0,help="type of noise: 0==no noise, 1==xavier")

    void main( nesi_init_arg_t * nia ) { 
      create_net_params();

      uint32_t const add_before_ix = get_layer_ix( *net_param, add_before_ln );
      uint32_t const orig_num_layers = (uint32_t)net_param->layer_size();

      mod_net_param->clear_layer(); // remove all layers
      for( uint32_t i = 0; i != add_before_ix; ++i ) { *mod_net_param->add_layer() = net_param->layer(i); }
      if( add_before_ix+1 > orig_num_layers ) {
	rt_err( "unhandled: expecting at least 1 layer (a ReLU) after add_before_ln"); }
      caffe::LayerParameter const & post_relu_layer = net_param->layer( add_before_ix + 1 );
      if( post_relu_layer.type() != ReLU_coi.type ) {
	rt_err( "unhandled: layer prior to add_before_ln is not RELU"); }

      if( add_before_ix < 2 ) { rt_err( "unhandled: expecting at least 2 layers prior to add_before_ln"); }

      caffe::LayerParameter const * pre_conv_layer = 0;
      uint32_t pcl_num_output = 0;
      for( uint32_t i = add_before_ix; i != 0; --i ) {
	pre_conv_layer = &net_param->layer( i - 1 );
	if( pre_conv_layer->has_convolution_param() ) {
	  pcl_num_output = pre_conv_layer->convolution_param().num_output();
	  break;
	}
	pre_conv_layer = 0;
      }
      if( !pre_conv_layer ) {
	rt_err( "unhandled: no conv layer prior to add_before_ln (need it for new layer num_outputs)."); }
      caffe::LayerParameter const * const pre_layer = &net_param->layer( add_before_ix - 1 );
      if( pre_layer->top_size() != 1) { rt_err( "unhandled: pre_layer->top_size() != 1"); }
      string const pre_layer_top = pre_layer->top(0);
      // add new layer
      string const new_layer_name = "pre_" + add_before_ln;
      caffe::LayerParameter * new_conv_layer = mod_net_param->add_layer();
      *new_conv_layer = net_param->layer(add_before_ix); // start with clone of layer we're adding before
      new_conv_layer->set_name( new_layer_name );
      new_conv_layer->clear_bottom(); new_conv_layer->add_bottom( pre_layer_top );
      new_conv_layer->clear_top(); new_conv_layer->add_top( new_layer_name );
      new_conv_layer->mutable_convolution_param()->set_num_output( pcl_num_output );
      // add new relu layer (FIXME: too strong to require ReLU for this layer?
      caffe::LayerParameter * new_relu_layer = mod_net_param->add_layer();
      *new_relu_layer = post_relu_layer; // start with clone of RELU from after layer we're adding before
      new_relu_layer->set_name( "relu_" + new_layer_name );
      new_relu_layer->clear_bottom(); new_relu_layer->add_bottom( new_layer_name );
      new_relu_layer->clear_top(); new_relu_layer->add_top( new_layer_name );

      for( uint32_t i = add_before_ix; i != orig_num_layers; ++i ) { 
	caffe::LayerParameter * nl = mod_net_param->add_layer();
	*nl = net_param->layer(i); 
	if( i == add_before_ix ) { // adjust bottom for layer we added a layer before
	  if( nl->bottom_size() != 1) { rt_err( "unhandled: add_before_layer->bottom_size() != 1"); }
	  nl->clear_bottom();
	  nl->add_bottom( new_layer_name );
	}
      }
      ensure_out_dir( nia );
      write_mod_pt();
      //return; // for testing, skip weights processing
      load_nets();
      create_identity_weights( mod_net_param, mod_net_pipe, new_layer_name, noise_mode );
      write_mod_net();
    }
  };

#include"gen/caffepb.cc.nesi_gen.cc"
}
