// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"conv_util.H"

#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"has_conv_fwd.H"
#include"io_util.H"
#include"nesi.H"
#include"caffepb.H"

namespace boda 
{
  // FIXME: we lost the ability to have NESI-based support for types and help for the per-operation params when we moved
  // them into conv_op_info_t. now they are just all strings ...

  // avg_pool: help="0 for max pooling, 1 for average pooling (others unsupported for compute)"
  conv_op_info_t const Pooling_coi{ "Pooling", {"in"}, {"out"}, {{"avg_pool","0"}} };
  conv_op_info_t const Convolution_coi{ "Convolution", { "in", "filts", "biases" }, { "out" }, {{"out_chans","0"}} };
  conv_op_info_t const ReLU_coi{ "ReLU", {"in"}, {"out"} };
  conv_op_info_t const Dropout_coi{ "Dropout", {"in"}, {"out"} };
  conv_op_info_t const LRN_coi{ "LRN", {"in"}, {"out"},{{"local_size","5"},{"alpha","1.0"},{"beta","0.75"},{"k","1.0"}} };
  conv_op_info_t const Accuracy_coi{ "Accuracy", {"in"}, {"out"} };
  conv_op_info_t const Softmax_coi{ "Softmax", {"in"}, {"prob"} };
  conv_op_info_t const SoftmaxWithLoss_coi{ "SoftmaxWithLoss", { "in", "label" },{ "in_grad_loss", "loss" } };
  conv_op_info_t const Data_coi{ "Data", {}, {"out"} }; // note: no inputs / source
  conv_op_info_t const Concat_coi{ "Concat", {"ins"}, {"out"}, {}, zi_bool(1) };
  conv_op_info_t const InnerProduct_coi{ "InnerProduct", {"in"}, {"out"} };
  // backwards-specific layers. there might be better/more-common names for these (and we will change/update them as
  // makes sense), but the idea is that they are operations in thier own right, not just 'backwards' versions of some
  // other ops. so we try to understand what they do functionally and name them accordingly.
  conv_op_info_t const Spreading_coi{ "Spreading", { "out", "out_grad_loss", "in" }, { "in_grad_loss" }, 
										     {{ "avg_pool", "0" }} };
  conv_op_info_t const ZeroIfNonPos_coi{ "ZeroIfNonPos", {"in","cond"}, {"out"} }; // note: dims(cond)==dims(out)==dims(in);out=(cond>=0)?in:0
  conv_op_info_t const BckConv_coi{ "BckConv", { "in", "filts", "biases", "out_grad_loss" },
    { "in_grad_loss", "filts_grad_loss", "biases_grad_loss" } };

  vect_rp_conv_op_info_t conv_op_infos{ &Pooling_coi, &Convolution_coi, &ReLU_coi, &Dropout_coi, &LRN_coi, 
      &Accuracy_coi, &Softmax_coi, &SoftmaxWithLoss_coi, &Data_coi, &Concat_coi, &InnerProduct_coi, &Spreading_coi,
      &ZeroIfNonPos_coi, &BckConv_coi };

  string conv_op_info_t::bot_an( uint32_t const & ix ) const {
    if( has_var_bots.v ) { assert_st( bots.size() == 1 ); return bots[0] + "_" + str(ix); }
    else { assert_st( ix < bots.size() ); return bots[ix]; }
  }
  string conv_op_info_t::top_an( uint32_t const & ix ) const { assert_st( ix < tops.size() ); return tops[ix]; }

  // type string checking + verify input/output argument count and other sanity checks
  bool conv_op_t::is( conv_op_info_t const & coi_ ) const { assert_st( coi ); return coi == &coi_; }
  void conv_op_t::set_and_check_coi( void ) { 
    assert_st( !coi );
    for( vect_rp_conv_op_info_t::const_iterator i = conv_op_infos.begin(); i != conv_op_infos.end(); ++i ) {
      if( type == (*i)->type ) { coi = *i; }
    }
    if( !coi ) { rt_err( strprintf( "Unknown operation of type '%s'.", str(type).c_str() ) ); }
    if( coi->tops.size() != tops.size() ) {
      rt_err( strprintf( "Wrong number of output arguments for operation of type '%s'. "
			 "had: tops.size()=%s, expected: coi->tops.size()=%s\n", 
			 str(coi->type).c_str(), str(tops.size()).c_str(), str(coi->tops.size()).c_str() ) );
    }
    if( coi->has_var_bots.v ? ( coi->bots.size() > bots.size() ) : ( coi->bots.size() != bots.size() ) ) {
      rt_err( strprintf( "Wrong number of input arguments for operation of type '%s'. "
			 "had: bots.size()=%s, expected: coi->bots.size()%s=%s\n", 
			 str(coi->type).c_str(), str(bots.size()).c_str(), 
			 coi->has_var_bots.v ? ">" : "",
			 str(coi->bots.size()).c_str() ) );
    }
    // check all params are set, set any missing ones to defaults
    for( map_str_str::const_iterator i = coi->params.begin(); i != coi->params.end(); ++i ) {
      if( !has( params, i->first ) ) { params[i->first] = i->second; }
    }
    // check that there are no extra/unknown params
    for( map_str_str::const_iterator i = params.begin(); i != params.end(); ++i ) {
      if( !has( coi->params, i->first ) ) { 
	rt_err( strprintf( "Unknown/invalid/extra parameter '%s' for operation of type '%s'.",
			   i->first.c_str(), str(coi->type).c_str() ) );
      }
    }
  }
  
  u32_pt_t conv_op_t::in_sz_to_out_sz( u32_pt_t const & in_sz, bool const ignore_padding ) const { 
    if( kern_sz.is_zeros() ) { // handle non-conv cases
      assert( !is(Convolution_coi) ); 
      if( is(Pooling_coi) || is(InnerProduct_coi) ) { return u32_pt_t{1,1}; } // global pooling / inner product special cases
      return in_sz; // otherwise, assume no effect on spatial dims (e.g. relu, lrn)
    }
    u32_pt_t const pad_in_sz = in_sz+(ignore_padding?u32_pt_t():(in_pad+in_pad));
    if( !pad_in_sz.both_dims_ge(kern_sz) ) { return u32_pt_t(); } // padded input too small to create any output
    if( is(Convolution_coi) ) { return (pad_in_sz-kern_sz)/stride + u32_pt_t(1,1); }
    else if( is(Pooling_coi) ) { return ceil_div( pad_in_sz-kern_sz,stride ) + u32_pt_t(1,1); }
    else { rt_err("unknown layer type"); }
  }
  u32_pt_t conv_op_t::out_sz_to_in_sz( u32_pt_t const & out_sz, bool const ignore_padding ) const { 
    if( kern_sz.is_zeros() ) { // handle non-conv cases
      assert( !is(Convolution_coi) );
      if( is(Pooling_coi) || is(InnerProduct_coi) ) { // inner product and global pooling special cases
	if( out_sz != u32_pt_t{1,1} ) { rt_err( "global pooling layer can't produce an out_sz other than {1,1}" ); }
	return u32_pt_t{0,0};  // special value means all input will be used ...
      } else { // otherwise, assume no effect on spatial dims (e.g. relu, lrn)
        return out_sz;
      }
    } 
    assert( out_sz.both_dims_non_zero() ); // this seems like it would be hard/confusing to handle
    u32_pt_t const no_pad_in_sz =  kern_sz + (out_sz-u32_pt_t(1,1))*stride;
    if( ignore_padding ) { return no_pad_in_sz; }
    // if the following assert does not hold, the result would be
    // negative, indicating *no input* yields a larger out_sz than
    // requested (due to padding). this might be valid, but it's
    // unclear what to return (zero?), so for now we refuse to try.
    assert_st( no_pad_in_sz.both_dims_ge( in_pad+in_pad ) ); 
    return no_pad_in_sz - (in_pad+in_pad);
  }

  dims_t conv_pipe_t::get_data_img_dims( void ) const {
    if( data_img_node_names.size() != 1 ) { rt_err( "not exactly one data img input node in net; can't process. data img input nodes are: " + str(data_img_node_names) ); }
    return must_get_node( data_img_node_names[0] )->dims;
  }
  u32_pt_t conv_pipe_t::get_data_img_xy_dims_3_chans_only( void ) const {
    // FIXME: better errors here if named dims don't exist?
    dims_t const data_dims = get_data_img_dims();
    uint32_t const data_dims_chan = data_dims.dsz("chan");
    if( data_dims_chan != 3 ) { rt_err( "unsupported number of fata img input node chans; must == 3; saw '"+str(data_dims_chan)+"'" ); }
    return u32_pt_t{ data_dims.dsz("x"), data_dims.dsz("y") }; 
  }
  
  // if out_node_name is empty, this returns the single unique output node of the net or throws an error. if out_node_name is
  // non-empty, it returns the single output node of the layer with name out_node_name (or throws an error).
  p_conv_node_t conv_pipe_t::get_single_top_node( void ) const {
    if( out_node_name.empty() ) {
      if( tops.size() != 1 ) { rt_err( "not exactly one sink/output node in net; can't process. output nodes are: " + str(tops) ); }
      return must_get_node( *tops.begin() ); 
    } else {
      if( !has( *nodes, out_node_name ) ) { 
	rt_err( "node '"+out_node_name+"' specified for use as producing the primary net output not found in net." ); 
      }
      return must_get_node( out_node_name );
    }
  }
  
  p_conv_node_t conv_pipe_t::get_or_make_node( string const & name, bool const is_bot, bool const is_top ) {
    p_conv_node_t & ret = (*nodes)[name];
    if( !ret ) { ret.reset( new conv_node_t{name} ); tops.insert(name); bots.insert(name); }
    if( is_bot ) { tops.erase(name); } if( is_top ) { bots.erase(name); }
    return ret;
  }
  p_conv_node_t conv_pipe_t::must_get_node( string const & name ) const {
    map_str_p_conv_node_t::const_iterator i = nodes->find( name );
    assert_st( i != nodes->end() );
    return i->second;
  }
  p_conv_op_t conv_pipe_t::get_op( string const & name ) const {
    map_str_p_conv_op_t::const_iterator i = convs->find( name );
    assert_st( i != convs->end() );
    return i->second;
  }
  void conv_pipe_t::add_conv( p_conv_op_t const & conv ) {
    conv->set_and_check_coi();
    bool in_place = 0;
    if( conv->is(ReLU_coi) || conv->is(Dropout_coi) || conv->is(ZeroIfNonPos_coi) ) { 
      if( conv->is(ZeroIfNonPos_coi) ) { assert_st( conv->tops[0] == conv->bots[0] ); }
      else { assert_st( conv->tops == conv->bots ); }
      get_or_make_node(conv->bots[0], 0, 0 )->in_place_ops.push_back( conv );
      in_place = 1;
    }
    bool did_ins = convs->insert( make_pair( conv->tag, conv ) ).second;
    if( !did_ins ) { rt_err( strprintf( "duplicate conv op '%s' seen; can't process net", conv->tag.c_str() ) ); }
    if( in_place ) { return; } // don't add in-place ops to top_for and bot_for
    for( vect_string::const_iterator i = conv->tops.begin(); i != conv->tops.end(); ++i ) {
      get_or_make_node( *i, 0, 1 )->top_for.push_back( conv->tag );
    }
    for( vect_string::const_iterator i = conv->bots.begin(); i != conv->bots.end(); ++i ) {
      get_or_make_node( *i, 1, 0 )->bot_for.push_back( conv->tag );
    }
  }

  // if the node has one top_for (a single writer), return it. if it has no writers, return null.
  // otherwise, throw an error.
  p_conv_op_t conv_pipe_t::maybe_get_single_writer( p_conv_node_t const & node ) const {
    if( node->top_for.empty() ) { return p_conv_op_t(); }
    if( node->top_for.size() != 1 ) { 
      rt_err( "unhandled multiple writers for node: " + node->name ); 
    }
    return get_op( node->top_for[0] );
  }
  p_conv_op_t conv_pipe_t::get_single_writer( p_conv_node_t const & node ) const {
    p_conv_op_t ret = maybe_get_single_writer( node );
    if( !ret ) { rt_err( "unhandled no writer (i.e. was primary input) for node: " + node->name ); }
    return ret;
  }

  // if the op has one input, return maybe_get_single_writer() for than
  // input. otherwise throw an error.
  p_conv_op_t conv_pipe_t::maybe_get_single_parent( p_conv_op_t const & cop ) const {
    assert_st( !cop->bots.empty() );
    if( cop->bots.size() != 1 ) {
      printf( "WARNING: unhandled multi-input op in support calc, using first input. cop->bots=%s\n", str(cop->bots).c_str() );
    }
    return maybe_get_single_writer( must_get_node(cop->bots[0]) );
  }


  void conv_pipe_t::calc_support_forward_op( p_conv_op_t const & cop, bool const ignore_padding ) {
    assert_st( cop->tops.size() >= 1 );
    p_conv_node_t const & node_out = must_get_node(cop->tops[0]);
    conv_support_info_t & csi_out = node_out->csi;
    if( csi_out.valid() ) { rt_err( "unhandled: node with multiple writers:"+node_out->name ); }

    // FIXME?: for now, we don't try to calculate support info for bck operations 
    if( cop->is( BckConv_coi ) ) {
    } else if( cop->is( Spreading_coi ) ) { 
    } else if( cop->is( SoftmaxWithLoss_coi ) ) { 
      csi_out.support_stride = u32_pt_t{};
      assert_st( cop->in_pad.is_zeros() ); csi_out.eff_tot_pad = must_get_node(cop->bots[0])->csi.eff_tot_pad;
      p_conv_node_t const & loss_node = must_get_node( cop->tops[1] );
      loss_node->csi.support_sz = u32_pt_t{};
      loss_node->csi.eff_tot_pad = csi_out.eff_tot_pad; // FIXME: correct? needed? maybe set to bogus/sentinel value?
    } else if( cop->is( Concat_coi ) ) {
      assert_st( cop->has_one_top() );
      for( vect_string::const_iterator j = cop->bots.begin(); j != cop->bots.end(); ++j ) {
	p_conv_node_t const & j_node = must_get_node(*j);
	conv_support_info_t const & csi_in = j_node->csi;
	if( !csi_in.valid() ) {  rt_err( "calc_support_info(): needed input support info for node not set. node name: " + str(*j) ); }
	if( (j == cop->bots.begin()) || (csi_in.support_stride.dims_max() > csi_out.support_stride.dims_max()) ) { // first input or bigger stride
	  if( j != cop->bots.begin() ) { 
	    printf( "WARNING: unhandled Concat layer '%s' with different strided inputs. "
		    "Note: support will be max size over inputs with largest stride in any dim.\n", str(cop->bots).c_str() );
	  }
	  csi_out.support_stride = csi_in.support_stride;
	  csi_out.support_sz = csi_in.support_sz;
	} else { 
	  if( csi_in.support_stride == csi_out.support_stride ) { csi_out.support_sz.max_eq( csi_in.support_sz ); }
	}
	csi_out.eff_tot_pad.max_eq( csi_in.eff_tot_pad );
      }
    } else {    
      assert_st( cop->has_one_top() );
      if( !cop->is( Convolution_coi ) ) { 
	if( cop->bots.size() != 1 ) { rt_err( "calc_support_forward_op(): unhandled multi-input operation: "+cop->tag+" of type " + cop->type+" " ); }
      }
      p_conv_node_t const & j_node = must_get_node(cop->bots[0]);
      conv_support_info_t const & csi_in = j_node->csi;
      if( !csi_in.valid() ) {  rt_err( "calc_support_info(): needed input support info for node not set. node name: " + str(cop->bots[0]) ); }
      u32_pt_t const in_sz_1x1 = cop->out_sz_to_in_sz( u32_pt_t(1,1), ignore_padding ); // == cop.kern_sz (if ign_pad)
      if( in_sz_1x1.is_zeros() || csi_in.support_sz.is_zeros() )  { // special values that means use all input
	csi_out.support_sz = u32_pt_t{};
      } else {
	assert_st( in_sz_1x1.both_dims_non_zero() );
	csi_out.support_sz = csi_in.support_sz + ( in_sz_1x1 - u32_pt_t(1,1) )*csi_in.support_stride;
      }
      assert_st( cop->stride.both_dims_non_zero() );
      csi_out.support_stride = csi_in.support_stride*cop->stride;
      csi_out.eff_tot_pad = csi_in.eff_tot_pad + ( cop->in_pad * csi_in.support_stride );
    }
  }
  void conv_pipe_t::calc_support_forward_rec( string const & node_name, bool const ignore_padding ) {
    p_conv_node_t const & node = must_get_node( node_name );
    // propogate support info forward from node to all ops that it feeds and thier outputs
    for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) {
      p_conv_op_t const & cop = get_op( *i );
      if( !cop->on_seen_bot() ) { continue; } // wait till we've seen all bottoms
      calc_support_forward_op( cop, ignore_padding );
      // depth-first recursive processing for any outputs
      for( vect_string::const_iterator i = cop->tops.begin(); i != cop->tops.end(); ++i ) { calc_support_forward_rec( *i, ignore_padding ); }
    }
  }
  // generally more sensible to with ignore_padding_for_support = 1 (but possibly interesting if = 0 too)
  void conv_pipe_t::calc_support_info( bool const ignore_padding ) {
    // support info for all needed root inputs should already be set by data layers. if not, it's a fatal error later
    // when we try to use it. note that support info for inputs/sources such as the filts/biases are not used and need
    // not be set.
    topo_visit_setup();
    for( set_string::const_iterator i = bots.begin(); i != bots.end(); ++i ) {  calc_support_forward_rec( *i, ignore_padding ); }
  }

  void conv_pipe_t::calc_dims_op( p_conv_op_t const & cop ) {
    assert_st( cop->tops.size() >= 1 );
    p_conv_node_t const & node_out = must_get_node(cop->tops[0]);
    dims_t & dims_out = node_out->dims;
    if( dims_out.size() ) { rt_err( "calc_dims_op(): unhandled: out dims already set (node with multiple writers):" + node_out->name ); }
    
    if( cop->is( BckConv_coi ) ) { // { in, filts, biases, out_grad_loss } --> { in_grad_loss, filts_grad_loss, biases_grad_loss }
      for( uint32_t i = 0; i != 3; ++i ) { // propogate # chans
	dims_t & od = must_get_node(cop->tops[i])->dims;
	if( od.size() ) { rt_err( "calc_dims_op(): unhandled: out dims already set (node with multiple writers):" + cop->tops[i] ); }
	od = must_get_node(cop->bots[i])->dims;
      }
    } else if( cop->is( Spreading_coi ) ) { 
      dims_out = must_get_node(cop->bots[2])->dims;
      // FIXME?: for now, we don't try to calculate support info for bck operations 
    } else if( cop->is( SoftmaxWithLoss_coi ) ) { 
      dims_out = must_get_node(cop->bots[0])->dims;
      dims_t & loss_dims = must_get_node( cop->tops[1] )->dims;
      // loss is a singleton (no img or chan dims anyway)... but, FIXME: why are there exactly 2 spatial dims? what else could you put? just 'x'?
      loss_dims = dims_t( vect_uint32_t{1,1}, vect_string{"y","x"}, 1 ); 
      // FIXME: even though the label is an input, we currently can't/don't try to set it's dims intially (i.e. from the data
      // layer), although perhaps that would make more sense. instead, we allow it to be set here, but check that it is
      // correct if it is already set. if it ever is set 'feed forward', this check is still good/okay. however, if it
      // is used by other things as an input, and they expect it to be set (i.e. becuase they use it), then that's no
      // good -- it might or might not get randomly set here depending on traversal order. really it's just not
      // generally okay to set it here.
      dims_t implied_label_dims( vect_uint32_t{ dims_out.dsz("img"), dims_out.dsz("y"), dims_out.dsz("x") }, vect_string{ "img", "y", "x" }, 1 );
      dims_t & label_dims = must_get_node( cop->bots[1] )->dims;
      if( label_dims.empty() ) { label_dims = implied_label_dims; }
      else if( label_dims != implied_label_dims ) { rt_err( "error: label used by multiple SoftmaxWithLoss layers with differing xy size or # imgs" ); }

      uint32_t & label_max_val = must_get_node( cop->bots[1] )->cio.max_val;
      uint32_t const implied_label_max_val = dims_out.dsz("chan");
      if( label_max_val == 0 ) { label_max_val = implied_label_max_val; }
      if( label_max_val != implied_label_max_val  ) { rt_err( "error: label used by multiple SoftmaxWithLoss layers with differing #s of chans." ); }

    } else if( cop->is( Concat_coi ) ) {
      assert_st( cop->has_one_top() ); 
      uint32_t dims_out_chans = 0; // start at zero for concat layer accumulation across inputs case
      for( vect_string::const_iterator j = cop->bots.begin(); j != cop->bots.end(); ++j ) {
	dims_t const & j_dims = must_get_node(*j)->dims;
	dims_out_chans += j_dims.dsz("chan"); // sum chans across all inputs
	if( !dims_out.size() ) { dims_out = j_dims; dims_out.clear_strides(); dims_out.must_get_dim_by_name("chan").sz = 0; } // convert to template
	else if( !j_dims.matches_template( dims_out ) ) { 
	  rt_err( "concat layer had incompatible inputs; must have all same non-chan dims. template (from first input) was: " + 
		  str(dims_out) + ". mismatching input was (index="+str(j - cop->bots.begin())+"): " + str(j_dims) );
	}
      }
      dims_out.must_get_dim_by_name("chan").sz = dims_out_chans;
      dims_out.calc_strides();
    } else {    
      assert_st( cop->has_one_top() );
      p_conv_node_t const & j_node = must_get_node(cop->bots[0]);
      uint32_t out_chans = 0;
      if( cop->is( Convolution_coi ) ) { 
	u32_pt_t kern_sz = cop->kern_sz;
	if( kern_sz.is_zeros() ) { kern_sz = get_xy_dims( j_node->dims ); } // 'global' input special case
	dims_t filts_dims( vect_uint32_t{ cop->u32_param("out_chans"), j_node->dims.dsz("chan"), kern_sz.d[1], kern_sz.d[0] },
			   vect_string{ "out_chan", "in_chan", "y", "x" }, 1 );
	must_get_node( cop->bots[1] )->dims = filts_dims;
	out_chans = cop->u32_param("out_chans");
	dims_t bias_dims( vect_uint32_t{ out_chans }, vect_string{ "out_chan" }, 1 );
	must_get_node( cop->bots[2] )->dims = bias_dims;
      } else {
	if( cop->bots.size() != 1 ) { rt_err( "calc_dims(): unhandled multi-input operation: "+cop->tag+" of type " + cop->type+" " ); }
      }
      dims_t const & dims_in = j_node->dims;
      assert_st( cop->stride.both_dims_non_zero() ); // FIXME: still belongs here? handled in in_sz_to_out_sz?
      dims_out = dims_in; // starting point
      dims_out.must_get_dim_by_name("chan").sz = out_chans ? out_chans : dims_in.dsz("chan"); // reset or propogate num_chans
      u32_pt_t const dims_out_sz = cop->in_sz_to_out_sz( get_xy_dims( dims_in ), 0 );
      if( dims_out_sz.both_dims_non_zero() ) { // calculate used_sz for debugging/informational output in dump_ios()
	j_node->cio.used_sz.max_eq( cop->out_sz_to_in_sz( dims_out_sz, 0 ) ); 
      } // else if there's no output, we used no input (used_sz left at zero)
      set_xy_dims( dims_out, dims_out_sz );
      dims_out.calc_strides();
    }
    for( vect_string::const_iterator i = cop->tops.begin(); i != cop->tops.end(); ++i ) { calc_dims_rec( *i ); }
  }
  void conv_pipe_t::calc_dims_rec( string const & node_name ) {
    p_conv_node_t const & node = must_get_node( node_name );
    for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) {
      p_conv_op_t const & cop = get_op( *i );
      if( !cop->on_seen_bot() ) { continue; } // wait till we've seen all bottoms
      calc_dims_op( cop );
    }
  }
  void conv_pipe_t::calc_dims( void ) {
    topo_visit_setup(); 
    for( set_string::const_iterator i = bots.begin(); i != bots.end(); ++i ) {  calc_dims_rec( *i ); }  
    for( map_str_p_conv_node_t::const_iterator i = nodes->begin(); i != nodes->end(); ++i ) { 
      dims_t const & d = i->second->dims;
      //printf( "post calc_dims() %s dims: %s\n", i->first.c_str(), str(d).c_str() );
      if( d.empty() ) { rt_err( strprintf( "error: no dims calculated for node %s after calc_dims()", str(i->first).c_str() ) ); }
    }
  }
  
  void conv_pipe_t::topo_visit_setup( void ) {
    for( map_str_p_conv_op_t::iterator i = convs->begin(); i != convs->end(); ++i ) { i->second->seen = 0; }
  }

  // note: assumed to be called after sizes are set by set_dims(). overwrites the xy_dims for nodes it touches.
  // note: recursively sturctured, but only works for chains currently. it's unclear what the
  // extention to non-chains would be exactly, but it would seem to depend on handling some
  // particular type of conv_op with >1 input.
  void conv_pipe_t::calc_sizes_back_rec( p_conv_node_t const & node_out, bool const ignore_padding ) {
    u32_pt_t const & xy_dims_out = get_xy_dims( node_out->dims );
    p_conv_op_t cop = maybe_get_single_writer( node_out );
    if( !cop ) { return; } // reached source, done
    if( !cop->is( Convolution_coi ) ) { assert_st( cop->has_one_top_one_bot() ); }
    else { assert_st( cop->tops.size() == 1 ); }
    p_conv_node_t node_in = must_get_node(cop->bots[0]);
    u32_pt_t xy_dims_in = get_xy_dims( node_in->dims );
    if( xy_dims_in.is_zeros() ) { rt_err( "internal error: !cio_in.valid() in calc_sizes_back_rec() at node:"+node_out->name ); }
    if( !xy_dims_out.both_dims_non_zero() ) {
      rt_err( strprintf( "calc_sizes_back(): unhandled/questionable case: pipeline stage %s output is zero-area.",
			 cop->tag.c_str() ) );
    }
    xy_dims_in = cop->out_sz_to_in_sz( xy_dims_out, ignore_padding );
    node_in->cio.used_sz = xy_dims_in; // by semantics of out_sz_to_in_sz (but checked below)
    assert_st( xy_dims_out == cop->in_sz_to_out_sz( xy_dims_in, ignore_padding ) );
    set_xy_dims( node_in->dims, xy_dims_in );
    calc_sizes_back_rec( node_in, ignore_padding ); // depth-first recursive processing for the input
  }

  void conv_pipe_t::calc_sizes_back( u32_pt_t const & out_sz, bool const ignore_padding ) {
    // initialize support info for single output
    p_conv_node_t const & node = get_single_top_node();
    u32_pt_t xy_dims_in = get_xy_dims( node->dims );
    assert( !xy_dims_in.is_zeros() );
    xy_dims_in = out_sz;
    set_xy_dims( node->dims, xy_dims_in );
    calc_sizes_back_rec( node, ignore_padding ); // calculate support
  }

  void conv_pipe_t::dump_pipe_rec( std::ostream & out, string const & node_name ) {
    p_conv_node_t node = must_get_node( node_name );
    if( node->bot_for.size() > 1 ) { 
      out << strprintf("node used by multiple ops:" ); 
      for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) { out << " " << *i; }
      out << strprintf("\n");
    }
    if( !node->dims.get_dim_by_name( "out_chan" ) ) { // FIXME: for compatibility, for now, skip filts/biases
      conv_support_info_t const & csi = node->csi;
      out << strprintf( "support_sz=%s support_stride=%s eff_tot_pad=%s\n", 
			str(csi.support_sz).c_str(), 
			str(csi.support_stride).c_str(), str(csi.eff_tot_pad).c_str() );
    }
    for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) {
      p_conv_op_t const & cop = get_op( *i );
      if( !cop->on_seen_bot() ) { continue; } // wait till we've seen all bottoms
      out << strprintf( "    ----  conv=%s \n", str(*cop).c_str() );
      for( vect_string::const_iterator i = cop->tops.begin(); i != cop->tops.end(); ++i ) { dump_pipe_rec( out, *i ); }
    }
  }

  void conv_pipe_t::dump_pipe( std::ostream & out ) {
    out << strprintf( "== BEGIN CONV PIPE ==\n" );
    topo_visit_setup();
    for( set_string::const_iterator i = bots.begin(); i != bots.end(); ++i ) { dump_pipe_rec( out, *i ); }
    out << strprintf( "== END CONV PIPE ==\n" );
  }

  void conv_pipe_t::dump_ios_rec( std::ostream & out, string const & node_name ) {
    p_conv_node_t node = must_get_node( node_name );
    if( node->bot_for.size() > 1 ) { 
      out << strprintf("(-->" ); 
      for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) { out << " " << *i; }
      out << strprintf(")");
    }
    if( !node->dims.get_dim_by_name( "out_chan" ) ) { // FIXME: for compatibility, for now, skip filts/biases
      u32_pt_t const & used_sz = node->cio.used_sz;
      u32_pt_t const xy_sz = get_xy_dims( node->dims );
      out << strprintf( "sz=%s -> ", str(xy_sz).c_str() );
      string size_err;
      if( xy_sz != used_sz ) { 
	if( (used_sz.d[0] > xy_sz.d[0]) || (used_sz.d[1] > xy_sz.d[1]) ) { size_err += "IMPLICIT PAD; "; }
	if( (used_sz.d[0] < xy_sz.d[0]) || (used_sz.d[1] < xy_sz.d[1]) ) { size_err += "DATA DISCARDED; "; }
	out << strprintf( "[%sused_sz=%s] -> ", size_err.c_str(), str(used_sz).c_str() );
      }
    }
    for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) {
      p_conv_op_t const & cop = get_op( *i );
      if( !cop->on_seen_bot() ) { continue; } // wait till we've seen all bottoms
      if( cop->tops.size() == 1 ) {
	out << cop->tag << " -> ";
	dump_ios_rec( out, cop->tops[0] );
      } else {
	out << cop->tag << " (";
	for( uint32_t i = 0; i != cop->tops.size(); ++i ) {
	  out << cop->tag << " -> ";
	  dump_ios_rec( out, cop->tops[i] );
	  out << cop->tag << ",";
	}
	out << cop->tag << " )";
      }
    }
  }
  void conv_pipe_t::dump_ios( std::ostream & out ) {
    out << "CONV_IOS: ";
    topo_visit_setup();
    for( set_string::const_iterator i = bots.begin(); i != bots.end(); ++i ) { dump_ios_rec( out, *i ); }
    out << "\n";
  }

  void print_blob_decl( std::ostream & out, string const & bn, p_conv_node_t const & node ) {
    string isss;
    if( node->top_for.empty() ) { isss += " SOURCE"; }
    if( node->bot_for.empty() ) { isss += " SINK"; }
    dims_t const & dims = node->dims;
    out << strprintf( "net.add_nda( \"%s\", NDA(\"%s\"", bn.c_str(), bn.c_str() );
    for( uint32_t i = 0; i != dims.size(); ++i ) { out << "," << dims[i].sz; }
    out << ") ) #" << isss << " ";
    for( uint32_t i = 0; i != dims.size(); ++i ) { if( i ) { out << ","; } out << dims[i].name; }
    out << "\n";
  }
  // FIXME: expanded_ops support removed for now, as it was incorrect/incomplete post sz->dim refactoring. unneeded? see
  // prior version in git if ressurection desired.
  void print_op_decl( std::ostream & out, conv_pipe_t const * const pipe, p_conv_op_t const & cop ) {
    char const * const tag_id = cop->tag.c_str();
    string params = strprintf( ",in_pad=\"%s\",stride=\"%s\",kern_sz=\"%s\"",
			       str(cop->in_pad).c_str(), str(cop->stride).c_str(), str(cop->kern_sz).c_str());
    for( map_str_str::const_iterator i = cop->params.begin(); i != cop->params.end(); ++i ) {
      params += strprintf( ",%s=\"%s\"", i->first.c_str(), i->second.c_str() );
    }
    out << strprintf( "net.add_op( %s(name=\"%s\",bot_names=%s,top_names=%s%s) )\n", 
		      cop->type.c_str(), tag_id, as_py_str_list(cop->bots).c_str(), as_py_str_list(cop->tops).c_str(), params.c_str() );
  }
  void conv_pipe_t::dump_ops_rec( std::ostream & out, string const & node_name ) {
    p_conv_node_t node = must_get_node( node_name );
    // print source nodes here, otherwise print with thier writing op
    if( node->top_for.empty() ) { print_blob_decl( out, node_name, node ); }
    else { assert( node->top_for.size() == 1 ); } // multiple writers not handled
    // print in-place ops for this node
    for( vect_p_conv_op_t::const_iterator j = node->in_place_ops.begin(); j != node->in_place_ops.end(); ++j ) {
      print_op_decl( out, this, *j );
    }
    for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) {
      p_conv_op_t const & cop = get_op( *i );
      if( !cop->on_seen_bot() ) { continue; } // wait till we've seen all bottoms
      for( vect_string::const_iterator i = cop->tops.begin(); i != cop->tops.end(); ++i ) { 
	print_blob_decl( out, *i, must_get_node(*i) ); // print decls for all of this ops output nodes here
      } 
      print_op_decl( out, this, cop );
      for( vect_string::const_iterator j = cop->tops.begin(); j != cop->tops.end(); ++j ) { dump_ops_rec( out, *j ); }
    }
  }
  void conv_pipe_t::dump_ops( std::ostream & out ) {
    topo_visit_setup();
    for( set_string::const_iterator i = bots.begin(); i != bots.end(); ++i ) { dump_ops_rec( out, *i ); }
  }

  // running test case for add_bck_ops/gradient calculations:
  // boda test_compute --model-name=nin_imagenet --wins-per-image=1 --imgs='(pil_fn=%(boda_test_dir)/pascal/head_1/%%s.txt)' --run-cnet='(in_dims=(img=1),out_node_name=conv1_grad_loss,add_bck_ops=1)' --cf2="(mode=rtc,show_rtc_calls=0,per_call_fn=out.py,dump_vars=())" --max-err=2 && cat test_compute.txt

  void conv_pipe_t::add_bck_ops_op( vect_p_conv_op_t & bck_ops, p_conv_op_t const & cop ) {
    p_conv_op_t bcop;
    if( cop->is( Softmax_coi ) ) { assert_st(0); }
    else if( cop->is( SoftmaxWithLoss_coi ) ) {
      assert_st( cop->bots[0]+"_grad_loss" == cop->tops[0] );
    } else if( cop->is( Pooling_coi ) ) {
      bcop.reset( new conv_op_t );
      *bcop = *cop;
      bcop->coi = 0;
      bcop->type = Spreading_coi.type;
      bcop->tag += "_bck";
      swap( bcop->tops, bcop->bots );
      bcop->bots.push_back( bcop->bots[0] + "_grad_loss" );
      bcop->bots.push_back( bcop->tops[0] ); // take original input as input (need size and which-elem-is-max per window) could use mask instead)
      bcop->tops[0] += "_grad_loss"; // note: pooling has no params, so there is second output for parameter gradients (as with some bck ops)
    } else if( cop->is( ReLU_coi ) ) {
      bcop.reset( new conv_op_t );
      *bcop = *cop;
      bcop->coi = 0;
      bcop->type = ZeroIfNonPos_coi.type;
      bcop->tag += "_bck";
      swap( bcop->tops, bcop->bots );
      bcop->bots.push_back( bcop->tops[0] ); // take original input as input
      bcop->bots[0] += "_grad_loss";
      bcop->tops[0] += "_grad_loss"; // note: ReLU has no params, so there is second output for parameter gradients (as with some bck ops)
    } else if( cop->is( Convolution_coi ) ) {
      bcop.reset( new conv_op_t );
      *bcop = *cop;
      bcop->coi = 0;
      bcop->params.clear(); // don't need/want out_chans (arguably should be removed from Convolution after setup too?)
      bcop->type = BckConv_coi.type;
      bcop->bots.push_back( bcop->tops[0] + "_grad_loss" ); // take _grad_loss of fwd conv output as input as well
      bcop->tops.clear(); for( uint32_t i = 0; i != 3; ++i ) { bcop->tops.push_back( bcop->bots[i] + "_grad_loss" ); } // outputs grads
      bcop->tag += "_bck";
    } else {
      printf( "FIXME: add_bck_ops: unhandled cop->type=%s\n", str(cop->type).c_str() );
    }
    if( bcop ) { bck_ops.push_back( bcop ); }
  }
  void conv_pipe_t::add_bck_ops_rec( vect_p_conv_op_t & bck_ops, string const & node_name ) {
    p_conv_node_t node = must_get_node( node_name );
    if( node->bot_for.size() == 0 ) { 
      // when add_bck_ops==1, we assume that all net tops/sinks should be produced by a SoftmaxWithLoss operation. that
      // is, we assume that the 'real' or raw outputs of the fwd net are already 'capped' with a combo
      // loss-function/fwd-top-gradient-producing node. we check that here:
      assert_st( node->top_for.size() == 1 );
      if( !get_op(node->top_for[0])->is(SoftmaxWithLoss_coi) ) {
	rt_err( strprintf( "add_bck_ops: unhandled: top node %s not produced by SoftmaxWithLoss op", node_name.c_str() ) );
      }
    }
    for( vect_p_conv_op_t::const_reverse_iterator j = node->in_place_ops.rbegin(); j != node->in_place_ops.rend(); ++j ) {
      p_conv_op_t const & ip_cop = *j;
      // FIXME: handle bck for in_place_opts. note: as usual, in_place_ops seem to be problematic or at least special. 
      add_bck_ops_op( bck_ops, ip_cop );
    }
    for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) {
      p_conv_op_t cop = get_op( *i );
      if( !cop->on_seen_bot() ) { continue; } // wait till we've seen all bots to process an op
      add_bck_ops_op( bck_ops, cop );
      for( vect_string::const_iterator j = cop->tops.begin(); j != cop->tops.end(); ++j ) { 
	add_bck_ops_rec( bck_ops, *j ); 
      }
    }
    if( node->bot_for.size() > 1 ) { 
      // nodes that are used in multiple places may need reductions. if _grad_loss_N nodes were created for this node,
      // reduce them here into a single _grad_loss node
    }
  }
  void conv_pipe_t::add_bck_ops( void ) {
    vect_p_conv_op_t bck_ops;
    topo_visit_setup();
    for( set_string::const_iterator i = bots.begin(); i != bots.end(); ++i ) { add_bck_ops_rec( bck_ops, *i ); }
    while( !bck_ops.empty() ) { add_conv( bck_ops.back() ); bck_ops.pop_back(); }
    has_bck_ops.v = 1;
  }

  // this will add a param named 'conv_has_relu' to all Convolution operations, with a default of false. If there
  // (exactly) a ReLU operation using the Convolution's output, then 'conv_has_relu' is set to 1 and the ReLU is erased.
  // similarly, the 'write_xposed' param is set for k1conv operations. (FIXME/TODO: actually do that)

  // FIXME: since this actually modifies the graph, it should be backend-neutral, and should be called from a
  // backend. however, currently, rtc_fwd() calls this. normally there is only one backends, so this is kinda-okay. but,
  // it means that other backends would see a different conv_pipe depending on the orderings of multiple backends in
  // test_dense. we're currently lucky that since the relu conv stays in the 'convs' map, and caffe both uses that map
  // and ignores the 'in_place_ops' list, caffe will do the same thing before or after this mutation happens. in general
  // the caffe backend is brittle/not-quite-right in the way it processes the pipe (see the 'cheating' FIXMEs) ....
  void conv_pipe_t::fuse_relus_and_maybe_enable_write_xposed( void ) {
    topo_visit_setup();
    for( set_string::const_iterator i = bots.begin(); i != bots.end(); ++i ) { framewx_rec( *i ); }
  }
  void conv_pipe_t::framewx_rec( string const & node_name ) {
    p_conv_node_t node = must_get_node( node_name );
    for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) {
      p_conv_op_t const & cop = get_op( *i );
      if( !cop->on_seen_bot() ) { continue; } // wait till we've seen all bottoms
      if( cop->is( Convolution_coi ) ) {
	p_conv_node_t no = must_get_node( cop->tops[0] );
	bool const conv_has_relu = (no->in_place_ops.size() > 0) && (no->in_place_ops[0]->is(ReLU_coi));
	if( conv_has_relu ) { no->in_place_ops.erase( no->in_place_ops.begin() ); } // remove fused relu
	must_insert( cop->params, "conv_has_relu", str(conv_has_relu) );
      } 
      for( vect_string::const_iterator j = cop->tops.begin(); j != cop->tops.end(); ++j ) { framewx_rec( *j ); }
    }
  }


  void conv_pipe_t::fwd_alloc_ndas( p_map_str_p_nda_float_t const & fwd, bool const & sinks_only ) {
    for( map_str_p_conv_node_t::const_iterator i = nodes->begin(); i != nodes->end(); ++i ) {
      p_conv_node_t const & node = i->second;
      dims_t node_dims = node->dims;
      node_dims.calc_strides(); // for now, assume no padding
      if( node->top_for.empty() ) { 
	//printf( "must_find(*fwd,node->name)->dims=%s node_dims=%s\n", str(must_find(*fwd,node->name)->dims).c_str(), str(node_dims).c_str() );
	assert_st( must_find( *fwd, node->name )->dims == node_dims ); 
      }
      else if( (!sinks_only) || node->bot_for.empty() ) {
	must_insert( *fwd, node->name, make_shared<nda_float_t>( node_dims ) );
      }
    }
  }

  // determined set of needed inputs for single-image-in (with dummy labels if needed) and puts them in in_vns and fwd
  void conv_pipe_t::run_setup_input( p_nda_float_t const & in, p_map_str_p_nda_float_t const & fwd, vect_string & in_vns ) {
    if( data_img_node_names.size() != 1 ) { rt_err( "run_one_blob_in_one_blob_out only supports exactly one image input" );}
    (*fwd)[data_img_node_names[0]] = in;
    in_vns.push_back( data_img_node_names[0] );
    // FIXME: hack for now to set labels (if needed) to something arbirtraty
    if( data_label_node_names.size() ) {
      string const & lnn = data_label_node_names[0];
      in_vns.push_back( lnn );
      assert_st( data_label_node_names.size() == data_img_node_names.size() ); // currently true by construction
      conv_io_t const & label_cio = must_get_node( lnn )->cio;
      p_nda_float_t label( new nda_float_t( must_get_node( lnn )->dims ) );
      uint32_t lix = 0;
      for( dims_iter_t di( label->dims ) ; ; ) { label->at(di.di) = lix % label_cio.max_val; ++lix; if( !di.next() ) { break; } } 
      (*fwd)[lnn] = label;
    }
#if 0
    vect_string missing_inputs;
    for( set_string::const_iterator i = bots.begin(); i != bots.end(); ++i ) { if( !has(*fwd,*i) ) { missing_inputs.push_back( *i ); } }
    if( !missing_inputs.empty() ) { rt_err( "run_one_blob_in_one_blob_out: missing_inputs (not images/labesl from data layers? internal error?): " + 
					    str(missing_inputs) ); } 
#endif
  }

  // assumes the single input blob is an image data blob (and there shouldn't be others)
  p_nda_float_t conv_pipe_t::run_one_blob_in_one_blob_out( p_nda_float_t const & in, p_has_conv_fwd_t const & conv_fwd ) {
    p_map_str_p_nda_float_t fwd = make_shared<map_str_p_nda_float_t>(); // *op_params );
    vect_string to_set_vns;
    run_setup_input( in, fwd, to_set_vns );
    assert( conv_fwd );
    conv_fwd->run_fwd( to_set_vns, fwd, {get_single_top_node()->name} );
    return must_find( *fwd, get_single_top_node()->name );
  }

  // FIXME: we *alter* the dims (especially the names) of blobs here. does that makes sense? generally, the blobs are
  // unused after this by the caller *and* the modifications are correct/sensible. but maybe the caller should have done
  // these modifications, not us?
  void conv_pipe_t::add_layer_blobs( string const & rln, p_vect_p_nda_float_t const & blobs ) {
    if( blobs->empty() ) { return; } // if no blobs to copy, we don't require a matching op exist in the pipe
    p_conv_op_t const & cop = get_op( rln );
    vect_string bsb_names;
    if( cop->is( Convolution_coi ) ) { 
      assert( blobs->size() == 2 );
      bsb_names.push_back( cop->tag + "_filts" ); 
      bsb_names.push_back( cop->tag + "_biases" ); 
    }
    else { for( uint32_t i = 0; i != blobs->size(); ++i ) { bsb_names.push_back( cop->tag + "_" + str(i) ); } }
    assert_st( bsb_names.size() == blobs->size() );
    for( uint32_t i = 0; i != bsb_names.size(); ++i ) { 
      assert_st( op_params->insert( std::make_pair( bsb_names[i], blobs->at(i) ) ).second );
    }
    must_insert( *layer_blobs, rln, blobs );
  }

  struct conv_ana_t : virtual public nesi, public has_main_t // NESI(help="analysize pipeline of convolutions wrt sizes at each layer, strides, padding, and per-layer-input-sizes (aka support sizes). ",bases=["has_main_t"], type_id="conv_ana")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    p_vect_conv_op_t convs; //NESI(default="()",help="set of conv-ish ops")
    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="text output filename")
    // filename_t convs_fn; NESI(help="input: filename for list of convs",req=1)
    p_uint32_t in_sz; //NESI(help="calculate sizes at all layers for the given input size and dump pipe")
    uint32_t in_chans; //NESI(default=3,help="number of input chans (used only to properly print number of input chans)")
    p_uint32_t out_sz; //NESI(help="calculate sizes at all layers for the given output size and dump pipe")

    uint32_t print_ops; //NESI(default=0,help="if non-zero, print ops. note: uses in_sz of (1,1) if in_sz not set.")

    uint32_t ignore_padding_for_support; //NESI(default=1,help="if 1, ignore any padding specified when calculating the support_size for a single pel for each layer")
#if 0
    // FIXME-MAYBE: we lost the ability to handle ignore-padding for sz during the sz->dims refactoring. we could
    // perhaps add it back by dynamically removing padding from the input net and/or conv_pipe before doing the various
    // operations. this might not be quite the same as the old functionality, but maybe that's okay. or maybe we can
    // ignore this forever.
    uint32_t ignore_padding_for_sz; //xNESI(default=0,help="if 1, ignore any padding specified when calculating the sizes at each layer for the in_sz or out_sz options")
#endif
    
    virtual void main( nesi_init_arg_t * nia ) { 
      // convert 'legacy' conv_ana linear pipe input to general net
      p_conv_pipe_t conv_pipe( new conv_pipe_t ); 
      string cur_node_name = "input";

      p_conv_node_t const data_img_node = conv_pipe->get_or_make_node(cur_node_name, 0, 0 );
      data_img_node->csi.init_as_source();
      data_img_node->dims = dims_t( vect_uint32_t{ 1, in_chans, in_sz ? *in_sz : 1, in_sz ? *in_sz : 1 }, vect_string{ "img", "chan", "y", "x" }, 1 );

      for( vect_conv_op_t::const_iterator i = convs->begin(); i != convs->end(); ++i ) {
	p_conv_op_t cop( new conv_op_t( *i ) );
	assert_st( cop->tops.empty() && cop->bots.empty() );
	cop->bots.push_back( cur_node_name );
	if( cop->type == Convolution_coi.type ) { 
	  cop->bots.push_back( cop->tag + "_filts" );  
	  conv_pipe->get_or_make_node( cop->bots.back(), 0, 0 )->csi.init_as_source();
	  cop->bots.push_back( cop->tag + "_biases" );
	  conv_pipe->get_or_make_node( cop->bots.back(), 0, 0 )->csi.init_as_source();
	}
	cur_node_name = cop->tag + "_out";
	cop->tops.push_back( cur_node_name );
	conv_pipe->add_conv( cop );
      }

      p_ofstream out = ofs_open( out_fn.exp );
      //(*out) << convs << "\n";
      conv_pipe->calc_support_info( ignore_padding_for_support );
      conv_pipe->calc_dims();
      conv_pipe->dump_pipe( *out ); 
      if( in_sz ) { 
	(*out) << ">> calculating network sizes forward given an in_sz of " << *in_sz << "\n";
	conv_pipe->dump_ios( *out ); 
      }
      if( print_ops ) { conv_pipe->dump_ops( *out ); }
      if( out_sz ) { 
	(*out) << ">> calculating network sizes backward given an out_sz of " << *out_sz << "\n";
	conv_pipe->calc_sizes_back( u32_pt_t( *out_sz, *out_sz ), 0 ); // ignore_padding_for_sz ); 
	conv_pipe->dump_ios( *out ); 
      }
    }
  };

  p_net_param_t conv_pipe_t::as_net_param( void ) const { assert( orig_net_param ); return orig_net_param; }

#include"gen/conv_util.H.nesi_gen.cc"
#include"gen/conv_util.cc.nesi_gen.cc"

};
