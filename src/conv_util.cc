// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"conv_util.H"

#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"io_util.H"
#include"nesi.H"
#include"caffepb.H"

namespace boda 
{

  u32_pt_t conv_op_t::in_sz_to_out_sz( u32_pt_t const & in_sz, bool const ignore_padding ) const { 
    if( kern_sz.is_zeros() ) { // handle non-conv cases
      assert( type != Convolution_str ); 
      if( (type == Pooling_str) || (type == InnerProduct_str) ) { return u32_pt_t{1,1}; } // global pooling / inner product special cases
      return in_sz; // otherwise, assume no effect on spatial dims (e.g. relu, lrn)
    }
    u32_pt_t const pad_in_sz = in_sz+(ignore_padding?u32_pt_t():in_pad.bnds_sum());
    if( !pad_in_sz.both_dims_ge(kern_sz) ) { return u32_pt_t(); } // padded input too small to create any output
    if( type == Convolution_str ) { return (pad_in_sz-kern_sz)/stride + u32_pt_t(1,1); }
    else if( type == Pooling_str ) { return ceil_div( pad_in_sz-kern_sz,stride ) + u32_pt_t(1,1); }
    else { rt_err("unknown layer type"); }
  }
  u32_pt_t conv_op_t::out_sz_to_in_sz( u32_pt_t const & out_sz, bool const ignore_padding ) const { 
    if( kern_sz.is_zeros() ) { // handle non-conv cases
      assert( type != Convolution_str );
      if( (type == Pooling_str) || (type == InnerProduct_str) ) { // inner product and global pooling special cases
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
    assert_st( no_pad_in_sz.both_dims_ge( in_pad.bnds_sum() ) ); 
    return no_pad_in_sz - in_pad.bnds_sum();
  }

  void conv_pipe_t::finalize( void ) {
    assert_st( !finalized ); // could relax
    assert_st( tops.empty() );
    assert_st( bots.empty() );
    for( map_str_p_conv_node_t::const_iterator i = nodes->begin(); i != nodes->end(); ++i ) {
      p_conv_node_t const & in = i->second;
      if( in->top_for.empty() ) { bots.push_back( in->name ); }
      if( in->bot_for.empty() ) { tops.push_back( in->name ); }
    }
    finalized = 1;
  }

  // this returns the single unique input node of the net or throws an error
  p_conv_node_t conv_pipe_t::get_single_bot_node( void ) const {
    p_conv_node_t ret;
    for( map_str_p_conv_node_t::const_iterator i = nodes->begin(); i != nodes->end(); ++i ) {
      p_conv_node_t const & in = i->second;
      if( in->top_for.empty() ) { 
	if( ret ) { rt_err( strprintf( "multiple source/input nodes found in net; can't process. two examples:'%s','%s'", 
				       ret->name.c_str(), in->name.c_str() ) ); }
	ret = in;
      }
    }
    if( !ret ) { rt_err( "no source/input nodes found in net; can't process. perhaps this is an invalid circular net?" ); }
    return ret;
  }

  p_conv_node_t conv_pipe_t::get_single_top_node( void ) const {
    p_conv_node_t ret;
    for( map_str_p_conv_node_t::const_iterator i = nodes->begin(); i != nodes->end(); ++i ) {
      p_conv_node_t const & in = i->second;
      if( in->bot_for.empty() ) { 
	if( ret ) { rt_err( strprintf( "multiple sink/output nodes found in net; can't process. two examples:'%s','%s'", 
				       ret->name.c_str(), in->name.c_str() ) ); }
	ret = in;
      }
    }
    if( !ret ) { rt_err( "no sink/output nodes found in net; can't process. perhaps this is an invalid circular net?" ); }
    return ret;
  }
  
  p_conv_node_t conv_pipe_t::get_or_make_node( string const & name ) {
    p_conv_node_t & ret = (*nodes)[name];
    if( !ret ) { ret.reset( new conv_node_t{name} ); }
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
    assert_st( !finalized );
    bool did_ins = convs->insert( make_pair( conv->tag, conv ) ).second;
    if( !did_ins ) { rt_err( strprintf( "duplicate conv op '%s' seen; can't process net", conv->tag.c_str() ) ); }
    for( vect_string::const_iterator i = conv->tops.begin(); i != conv->tops.end(); ++i ) {
      get_or_make_node( *i )->top_for.push_back( conv->tag );
    }
    for( vect_string::const_iterator i = conv->bots.begin(); i != conv->bots.end(); ++i ) {
      get_or_make_node( *i )->bot_for.push_back( conv->tag );
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
  // if the op has one input, return maybe_get_single_writer() for than
  // input. otherwise throw an error.
  p_conv_op_t conv_pipe_t::maybe_get_single_parent( p_conv_op_t const & cop ) const {
    assert_st( !cop->bots.empty() );
    if( cop->bots.size() != 1 ) {
      printf( "WARNING: unhandled multi-input op in support calc, using first input. cop->bots=%s\n", str(cop->bots).c_str() );
    }
    return maybe_get_single_writer( must_get_node(cop->bots[0]) );
  }

  void conv_pipe_t::calc_support_forward_rec( p_conv_node_t const & node_in, bool const ignore_padding ) {
    // propogate support info forward from node to all ops that it feeds and thier outputs
    for( vect_string::const_iterator i = node_in->bot_for.begin(); i != node_in->bot_for.end(); ++i ) {
      p_conv_op_t const & cop = get_op( *i );
      if( !cop->on_seen_bot() ) { continue; } // wait till we've seen all bottoms
      assert_st( cop->has_one_top() );
      p_conv_node_t const & node_out = must_get_node(cop->tops[0]);
      conv_support_info_t & csi_out = node_out->csi;
      conv_io_t & cio_out = node_out->cio;
      if( csi_out.valid() ) { rt_err( "unhandled: node with multiple writers:"+node_out->name ); }
      assert_st( cio_out.chans == uint32_t_const_max ); // should not be set yet
      cio_out.chans = 0; // start at zero for concat layer accumulation across inputs case
      // FIXME: move to own func
      uint32_t const & out_chans = cop->out_chans; 
      for( vect_string::const_iterator j = cop->bots.begin(); j != cop->bots.end(); ++j ) {
	p_conv_node_t const & j_node = must_get_node(*j);
	conv_support_info_t const & csi_in = j_node->csi;
	conv_io_t const & cio_in = j_node->cio;
	if( cop->type == Concat_str ) {
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
	  assert( !out_chans ); // concat shouldn't have a # of output chans specified
	  cio_out.chans += cio_in.chans; // sum chans across all inputs
	} else {
	  if( j == cop->bots.begin() ) {
	    u32_pt_t const in_sz_1x1 = cop->out_sz_to_in_sz( u32_pt_t(1,1), ignore_padding ); // == cop.kern_sz (if ign_pad)
	    if( in_sz_1x1.is_zeros() || csi_in.support_sz.is_zeros() )  { // special values that means use all input
	      csi_out.support_sz = u32_pt_t{};
	    } else {
	      assert_st( in_sz_1x1.both_dims_non_zero() );
	      csi_out.support_sz = csi_in.support_sz + ( in_sz_1x1 - u32_pt_t(1,1) )*csi_in.support_stride;
	    }
	    assert_st( cop->stride.both_dims_non_zero() );
	    csi_out.support_stride = csi_in.support_stride*cop->stride;
	    csi_out.eff_tot_pad = csi_in.eff_tot_pad + cop->in_pad.scale_dims( csi_in.support_stride );
	    cio_out.chans = out_chans ? out_chans : cio_in.chans; // reset or propogate num_chans

	  } else { rt_err( "unhandled multi-input operation: "+cop->tag+" of type " + cop->type+" " ); }
	}
      }

#if 0
      // traverse backward to root to calculate eff_tot_pad
      for( p_conv_op_t cop_back = cop; cop_back; cop_back = maybe_get_single_parent(cop_back) ) {
	csi_out.eff_tot_pad = cop_back->in_pad + csi_out.eff_tot_pad.scale_dims( cop_back->stride );	
      }
#endif
      calc_support_forward_rec( node_out, ignore_padding ); // depth-first recursive processing for any outputs
    }
  }

  // generally more sensible to with ignore_padding_for_support = 1 (but possibly interesting if = 0 too)
  void conv_pipe_t::calc_support_info( bool const ignore_padding, uint32_t const & in_chans ) {
    // initialize support info for single root input
    p_conv_node_t const & node = get_single_bot_node();
    conv_support_info_t & csi = node->csi;
    assert( !csi.valid() );
    csi.support_sz = u32_pt_t(1,1);
    csi.support_stride = u32_pt_t(1,1);
    node->cio.chans = in_chans;
    topo_visit_setup();
    calc_support_forward_rec( node, ignore_padding ); // calculate support
  }

  
  void conv_pipe_t::clear_sizes( void ) {
    for( map_str_p_conv_node_t::iterator i = nodes->begin(); i != nodes->end(); ++i ) { i->second->cio = conv_io_t(); }
  }
  void conv_pipe_t::topo_visit_setup( void ) {
    for( map_str_p_conv_op_t::iterator i = convs->begin(); i != convs->end(); ++i ) { i->second->bots_seen = 0; }
  }

  void conv_pipe_t::calc_sizes_forward_rec( p_conv_node_t const & node_in, bool const ignore_padding ) {
    // propogate support info forward from node to all ops that it feeds and thier outputs
    for( vect_string::const_iterator i = node_in->bot_for.begin(); i != node_in->bot_for.end(); ++i ) {
      p_conv_op_t const & cop = get_op( *i );
      if( !cop->on_seen_bot() ) { continue; } // wait till we've seen all bottoms
      assert_st( cop->has_one_top() );
      p_conv_node_t const & node_out = must_get_node(cop->tops[0]);
      conv_io_t & cio_out = node_out->cio;
      if( !cio_out.sz.is_zeros() ) { rt_err( "node size calculation is not supported for reconvegent networks at node:"+node_out->name ); }

      // FIXME: move to own func
      if( (cop->bots.size() != 1) && (cop->type != Concat_str) ) { 
	rt_err( "unhandled multi-input operation: "+cop->tag+" of type " + cop->type+" " ); }
      for( vect_string::const_iterator j = cop->bots.begin(); j != cop->bots.end(); ++j ) {
	conv_io_t & cio_in = must_get_node(*j)->cio; // note: non-const since cio_in.used_sz is updated
	if( j == cop->bots.begin() ) { // first input 
	  cio_out.sz = cop->in_sz_to_out_sz( cio_in.sz, ignore_padding );
	  if( cio_out.sz.both_dims_non_zero() ) { 
	    cio_in.used_sz.max_eq( cop->out_sz_to_in_sz( cio_out.sz, ignore_padding ) );
	  } // else if there's no output, we used no input (used_sz left at zero)
	} else { // handle multiple inputs for concat layer (only!)
	  assert( cop->type == Concat_str );
	  // x/y dims must agree across all inputs
	  u32_pt_t const out_sz = cop->in_sz_to_out_sz( cio_in.sz, ignore_padding );
	  assert_st( out_sz == cio_out.sz );
	}
      }
      calc_sizes_forward_rec( node_out, ignore_padding ); // depth-first recursive processing for any outputs
    }
  }
  void conv_pipe_t::calc_sizes_forward( u32_pt_t const & in_sz, bool const ignore_padding ) {
    // initialize support info for single root input
    p_conv_node_t const & node = get_single_bot_node();
    conv_io_t & cio = node->cio;
    assert( cio.sz.is_zeros() ); // shouldn't be calculated yet
    cio.sz = in_sz;
    topo_visit_setup();
    calc_sizes_forward_rec( node, ignore_padding ); // calculate support
  }

  // note: recursively sturctured, but only works for chains currently. it's unclear what the
  // extention to non-chains would be exactly, but it would seem to depend on handling some
  // particular type of conv_op with >1 input.
  void conv_pipe_t::calc_sizes_back_rec( p_conv_node_t const & node_out, bool const ignore_padding ) {
    conv_io_t const & cio_out = node_out->cio;
    p_conv_op_t cop = maybe_get_single_writer( node_out );
    if( !cop ) { return; } // reached source, done
    assert_st( cop->has_one_top_one_bot() );
    p_conv_node_t const & node_in = must_get_node(cop->bots[0]);
    conv_io_t & cio_in = node_in->cio;
    if( !cio_in.sz.is_zeros() ) { rt_err( "internal error: cio_in.valid() in calc_sizes_back_rec() at node:"+node_out->name ); }
    if( !cio_out.sz.both_dims_non_zero() ) {
      rt_err( strprintf( "calc_sizes_back(): unhandled/questionable case: pipeline stage %s output is zero-area.",
			 cop->tag.c_str() ) );
    }
    cio_in.sz = cop->out_sz_to_in_sz( cio_out.sz, ignore_padding );
    cio_in.used_sz = cio_in.sz; // by semantics of out_sz_to_in_sz (but checked below)
    assert_st( cio_out.sz == cop->in_sz_to_out_sz( cio_in.sz, ignore_padding ) );
    calc_sizes_back_rec( node_in, ignore_padding ); // depth-first recursive processing for the input
  }

  void conv_pipe_t::calc_sizes_back( u32_pt_t const & out_sz, bool const ignore_padding ) {
    // initialize support info for single output
    p_conv_node_t const & node = get_single_top_node();
    conv_io_t & cio = node->cio;
    assert( cio.sz.is_zeros() );
    cio.sz = out_sz;
    calc_sizes_back_rec( node, ignore_padding ); // calculate support
  }


  void conv_pipe_t::dump_pipe_rec( std::ostream & out, string const & node_name ) {
    p_conv_node_t node = must_get_node( node_name );
    if( node->bot_for.size() > 1 ) { 
      out << strprintf("node used by multiple ops:" ); 
      for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) { out << " " << *i; }
      out << strprintf("\n");
    }
    conv_support_info_t const & csi = node->csi;
    out << strprintf( "support_sz=%s support_stride=%s eff_tot_pad=%s\n", 
		      str(csi.support_sz).c_str(), 
		      str(csi.support_stride).c_str(), str(csi.eff_tot_pad).c_str() );
    for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) {
      p_conv_op_t const & cop = get_op( *i );
      if( !cop->on_seen_bot() ) { continue; } // wait till we've seen all bottoms
      out << strprintf( "    ----  conv=%s \n", str(*cop).c_str() );

      assert_st( cop->has_one_top() );
      dump_pipe_rec( out, cop->tops[0] );
    }
  }

  void conv_pipe_t::dump_pipe( std::ostream & out ) {
    assert_st( finalized );
    out << strprintf( "== BEGIN CONV PIPE ==\n" );
    topo_visit_setup();
    for( vect_string::const_iterator i = bots.begin(); i != bots.end(); ++i ) { dump_pipe_rec( out, *i ); }
    out << strprintf( "== END CONV PIPE ==\n" );
  }

  void conv_pipe_t::dump_ios_rec( std::ostream & out, string const & node_name ) {
    p_conv_node_t node = must_get_node( node_name );
    if( node->bot_for.size() > 1 ) { 
      out << strprintf("(-->" ); 
      for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) { out << " " << *i; }
      out << strprintf(")");
    }
    conv_io_t const & cio = node->cio;
    out << strprintf( "sz=%s -> ", str(cio.sz).c_str() );
    string size_err;
    if( cio.sz != cio.used_sz ) { 
      if( (cio.used_sz.d[0] > cio.sz.d[0]) || (cio.used_sz.d[1] > cio.sz.d[1]) ) { size_err += "IMPLICIT PAD; "; }
      if( (cio.used_sz.d[0] < cio.sz.d[0]) || (cio.used_sz.d[1] < cio.sz.d[1]) ) { size_err += "DATA DISCARDED; "; }
      out << strprintf( "[%sused_sz=%s] -> ", size_err.c_str(), str(cio.used_sz).c_str() );
    }
    for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) {
      p_conv_op_t const & cop = get_op( *i );
      if( !cop->on_seen_bot() ) { continue; } // wait till we've seen all bottoms
      out << cop->tag << " -> ";
      assert_st( cop->has_one_top() );
      dump_ios_rec( out, cop->tops[0] );
    }
  }
  void conv_pipe_t::dump_ios( std::ostream & out ) {
    assert_st( finalized );
    out << "CONV_IOS: ";
    topo_visit_setup();
    for( vect_string::const_iterator i = bots.begin(); i != bots.end(); ++i ) { dump_ios_rec( out, *i ); }
    out << "\n";
  }

  void print_blob_decl( std::ostream & out, string const & bn, p_conv_node_t const & node ) {
    string isss;
    if( node->top_for.empty() ) { isss += " SOURCE"; }
    if( node->bot_for.empty() ) { isss += " SINK"; }
    conv_io_t & cio = node->cio;
    out << strprintf( "%s = NDA(\"%s\",num_img,%s,%s,%s) #%s num,chan,y,x\n", 
		      as_pyid(bn).c_str(), as_pyid(bn).c_str(), str(cio.chans).c_str(), 
		      str(cio.sz.d[1]).c_str(), str(cio.sz.d[0]).c_str(), 
		      isss.c_str() );
  }
  


  string get_conv_as_sgemm( string const & top_name, string const & bot_name, string const & filts_name,
			    uint32_t const M, uint32_t const N, uint32_t const K, string const & extra_params ) {
    string const buf_name = bot_name + "_one_row_per_patch_buf";
    string ret;
    ret += strprintf( "%s = NDA(\"%s\",%u,%u)\n",buf_name.c_str(),buf_name.c_str(),M,N);
    ret += strprintf( "for i in range(0,num_img):\n" );
    ret += strprintf( "  patches_to_rows( src=%s[i,:,:,:], dest=%s, %s ) # one copy per output elem\n",
		      bot_name.c_str(),buf_name.c_str(), extra_params.c_str() );
    ret += strprintf( "  %s = %s * transpose(reshape(%s,%u,%u)) # sgemm: MxNxK == %ux%ux%u\n", top_name.c_str(),buf_name.c_str(), 
		      filts_name.c_str(),K,N,M,N,K );
    return ret;
  }

  void print_op_decl( std::ostream & out, conv_pipe_t const * const pipe, p_conv_op_t const & cop, bool const expanded_ops ) {
    string extra_params;
    string expanded_op;
    string const tag_id_str = as_pyid( cop->tag );
    char const * const tag_id = tag_id_str.c_str();
    
    string const pad_and_stride = strprintf( "in_pad=\"%s\",stride=\"%s\"", cop->in_pad.parts_str().c_str(), str(cop->stride).c_str() );
    uint32_t M = 0, N = 0, K = 0;
    if( cop->type == Convolution_str || cop->type == InnerProduct_str ) {
      assert_st( cop->bots.size() == 1 );
      conv_io_t & cio_in = pipe->must_get_node( cop->bots[0] )->cio;
      u32_pt_t kern_sz = cop->kern_sz;
      if( kern_sz.is_zeros() ) { kern_sz = cio_in.sz; } // 'global' input special case

      out << strprintf( "%s_filts = NDA(\"%s_filts\",%s,%s,%s,%s) # SOURCE out_chan,in_chan,y,x\n", 
			tag_id, tag_id, str(cop->out_chans).c_str(), str(cio_in.chans).c_str(),
			str(kern_sz.d[1]).c_str(), str(kern_sz.d[0]).c_str() );
      out << strprintf( "%s_biases = NDA(\"%s_biases\",%s) # SOURCE out_chan\n", 
			tag_id, tag_id, str(cop->out_chans).c_str() );
      extra_params = strprintf( ",filts=%s_filts,biases=%s_biases", tag_id, tag_id );

      assert_st( cop->tops.size() == 1 );
      conv_io_t & cio_out = pipe->must_get_node( cop->tops[0] )->cio;
      M = cio_out.sz.d[0] * cio_out.sz.d[1];
      N = kern_sz.d[0]*kern_sz.d[1]*cio_in.chans;
      K = cop->out_chans;

      // get expanded op 
      expanded_op = get_conv_as_sgemm(cop->tops[0],cop->bots[0],tag_id_str+"_filts",M,N,K,pad_and_stride);
    }
    // print decls for all of this ops output nodes here
    for( vect_string::const_iterator i = cop->tops.begin(); i != cop->tops.end(); ++i ) {
      print_blob_decl( out, *i, pipe->must_get_node(*i) ); 
    }
    // print acutal op
    if( expanded_ops && !expanded_op.empty() ) { out << expanded_op; }
    else {
      out << strprintf( "%s(name=\"%s\",bots=%s,tops=%s%s,\n\t%s)\n", 
			cop->type.c_str(), tag_id, as_pylist(cop->bots).c_str(), as_pylist(cop->tops).c_str(),
			extra_params.c_str(), pad_and_stride.c_str() );
    }
    
    // print any in-place ops for any of the output nodes
    for( vect_string::const_iterator i = cop->tops.begin(); i != cop->tops.end(); ++i ) {
      p_conv_node_t const & out_node = pipe->must_get_node(*i);
      for( vect_p_conv_op_t::const_iterator j = out_node->in_place_ops.begin(); j != out_node->in_place_ops.end(); ++j ) {
	p_conv_op_t const & ip_cop = *j;
	out << strprintf( "%s(name=\"%s\",in_place=[%s])\n", 
			  ip_cop->type.c_str(), as_pyid(ip_cop->tag).c_str(), as_pyid(out_node->name).c_str() );
      }
    }
  }

  void conv_pipe_t::dump_ops_rec( std::ostream & out, string const & node_name, bool const & expand_ops ) {
    p_conv_node_t node = must_get_node( node_name );
    // print source nodes here, otherwise print with thier writing op
    if( node->top_for.empty() ) { print_blob_decl( out, node_name, node ); }
    else { assert( node->top_for.size() == 1 ); } // multiple writers not handled
    for( vect_string::const_iterator i = node->bot_for.begin(); i != node->bot_for.end(); ++i ) {
      p_conv_op_t const & cop = get_op( *i );
      if( !cop->on_seen_bot() ) { continue; } // wait till we've seen all bottoms
      print_op_decl( out, this, cop, expand_ops );
      assert_st( cop->has_one_top() );
      dump_ops_rec( out, cop->tops[0], expand_ops );
    }
  }

  void conv_pipe_t::dump_ops( std::ostream & out, bool const & expand_ops ) {
    assert_st( finalized );
    topo_visit_setup();
    for( vect_string::const_iterator i = bots.begin(); i != bots.end(); ++i ) { dump_ops_rec( out, *i, expand_ops ); }
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
    uint32_t ignore_padding_for_sz; //NESI(default=0,help="if 1, ignore any padding specified when calculating the sizes at each layer for the in_sz or out_sz options")
    uint32_t print_ops; //NESI(default=0,help="if non-zero, print ops. note: requires in_sz to be set.")
    uint32_t ignore_padding_for_support; //NESI(default=1,help="if 1, ignore any padding specified when calculating the support_size for a single pel for each layer")
    
    virtual void main( nesi_init_arg_t * nia ) { 
      // convert 'legacy' conv_ana linear pipe input to general net
      p_conv_pipe_t conv_pipe( new conv_pipe_t ); 
      string cur_node_name = "input";
      for( vect_conv_op_t::const_iterator i = convs->begin(); i != convs->end(); ++i ) {
	p_conv_op_t cop( new conv_op_t( *i ) );
	assert_st( cop->tops.empty() && cop->bots.empty() );
	cop->bots.push_back( cur_node_name );
	cur_node_name = cop->tag + "_out";
	cop->tops.push_back( cur_node_name );
	conv_pipe->add_conv( cop );
      }
      conv_pipe->finalize();

      p_ofstream out = ofs_open( out_fn.exp );
      //(*out) << convs << "\n";
      conv_pipe->calc_support_info( ignore_padding_for_support, in_chans );
      conv_pipe->dump_pipe( *out ); 
      if( out_sz ) { 
	(*out) << ">> calculating network sizes backward given an out_sz of " << *out_sz << "\n";
	conv_pipe->calc_sizes_back( u32_pt_t( *out_sz, *out_sz ), ignore_padding_for_sz ); 
	conv_pipe->dump_ios( *out ); 
	conv_pipe->clear_sizes();
      }
      p_vect_conv_io_t conv_ios;
      if( in_sz ) { 
	(*out) << ">> calculating network sizes forward given an in_sz of " << *in_sz << "\n";
	conv_pipe->calc_sizes_forward( u32_pt_t( *in_sz, *in_sz ), ignore_padding_for_sz ); 
	conv_pipe->dump_ios( *out ); 
	if( print_ops ) { conv_pipe->dump_ops( *out, 0 ); }
	conv_pipe->clear_sizes();	
      }
    }
  };

#include"gen/conv_util.H.nesi_gen.cc"
#include"gen/conv_util.cc.nesi_gen.cc"

};
