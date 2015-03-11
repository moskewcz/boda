// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"conv_util.H"

#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"io_util.H"
#include"nesi.H"

namespace boda 
{

  u32_pt_t conv_op_t::in_sz_to_out_sz( u32_pt_t const & in_sz, bool const ignore_padding ) const { 
    if( kern_sz.is_zeros() ) { // handle non-conv cases
      assert( type != "conv" ); 
      if( (type == "pool") || (type == "ip") ) { return u32_pt_t{1,1}; } // global pooling / inner product special cases
      return in_sz; // otherwise, assume no effect on spatial dims (e.g. relu)
    }
    u32_pt_t const pad_in_sz = in_sz+(ignore_padding?u32_pt_t():in_pad.bnds_sum());
    if( !pad_in_sz.both_dims_ge(kern_sz) ) { return u32_pt_t(); } // padded input too small to create any output
    if( type == "conv" ) { return (pad_in_sz-kern_sz)/stride + u32_pt_t(1,1); }
    else if( type == "pool" ) { return ceil_div( pad_in_sz-kern_sz,stride ) + u32_pt_t(1,1); }
    else { rt_err("unknown layer type"); }
  }
  u32_pt_t conv_op_t::out_sz_to_in_sz( u32_pt_t const & out_sz, bool const ignore_padding ) const { 
    if( kern_sz.is_zeros() ) { // handle non-conv cases
      assert( type != "conv" );
      if( (type == "pool") || (type == "ip") ) { // inner product and global pooling special cases
	if( out_sz != u32_pt_t{1,1} ) { rt_err( "global pooling layer can't produce an out_sz other than {1,1}" ); }
	return u32_pt_t{0,0};  // special value means all input will be used ...
      } else { // otherwise, assume no effect on spatial dims (e.g. relu)
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

  void conv_pipe_t::zero_conv_ios( vect_conv_io_t & conv_ios ) {
    conv_ios.clear();
    conv_ios.resize( convs->size() + 1 );
    for( vect_conv_io_t::iterator i = conv_ios.begin(); i != conv_ios.end(); ++i ) {
      i->sz = u32_pt_t(); i->used_sz = u32_pt_t(); i->chans = 0;
    }
  }

  // generally more sensible to with ignore_padding_for_support = 1 (but possibly interesting if = 0 too)
  void conv_pipe_t::calc_support_info( void ) {
    conv_sis.resize( convs->size() + 1 );
    conv_sis.front().support_sz = u32_pt_t(1,1);
    conv_sis.front().support_stride = u32_pt_t(1,1);
    for( uint32_t i = 0; i != convs->size(); ++i ) {
      conv_op_t const & cop = convs->at(i);
      //assert_st( cop.kern_sz.both_dims_non_zero() );
      u32_pt_t const in_sz_1x1 = cop.out_sz_to_in_sz( u32_pt_t(1,1), ignore_padding_for_support ); // == cop.kern_sz (if ign_pad)
      if( in_sz_1x1.is_zeros() || conv_sis[i].support_sz.is_zeros() )  { // special values that means use all input
	conv_sis[i+1].support_sz = u32_pt_t{};
      } else {
	assert_st( in_sz_1x1.both_dims_non_zero() );
	conv_sis[i+1].support_sz = conv_sis[i].support_sz + ( in_sz_1x1 - u32_pt_t(1,1) )*conv_sis[i].support_stride;
      }
      conv_sis[i+1].support_stride = conv_sis[i].support_stride*cop.stride;
    }
    // backward passes to calculate eff_tot_pad
    for( uint32_t j = 0; j != conv_sis.size(); ++j ) {
      for( uint32_t i = j; i; --i ) {
	conv_op_t const & cop = convs->at(i - 1);
	conv_sis[j].eff_tot_pad = cop.in_pad + conv_sis[j].eff_tot_pad.scale_dims( cop.stride );	
      }
    }
  }

  p_vect_conv_io_t conv_pipe_t::calc_sizes_back( u32_pt_t const & out_sz, bool const ignore_padding ) {
    p_vect_conv_io_t conv_ios( new vect_conv_io_t( convs->size() + 1 ) );
    conv_ios->back().sz = out_sz;
    for( uint32_t i = convs->size(); i; --i ) {
      conv_op_t const & cop = convs->at(i-1);
      if( !conv_ios->at(i).sz.both_dims_non_zero() ) {
	rt_err( strprintf( "calc_sizes_back(): unhandled/questionable case: pipeline stage %s output is zero-area.",
			   cop.tag.c_str() ) );
      }
      conv_ios->at(i-1).sz = cop.out_sz_to_in_sz( conv_ios->at(i).sz, ignore_padding );
      conv_ios->at(i-1).used_sz = conv_ios->at(i-1).sz; // by semantics of out_sz_to_in_sz (but checked below)
      assert_st( conv_ios->at(i).sz == cop.in_sz_to_out_sz( conv_ios->at(i-1).sz, ignore_padding ) );
    }
    return conv_ios;
  }
  p_vect_conv_io_t conv_pipe_t::calc_sizes_forward( u32_pt_t const & in_sz, uint32_t const & in_chans, bool const ignore_padding ) {
    p_vect_conv_io_t conv_ios( new vect_conv_io_t( convs->size() + 1 ) );
    conv_ios->front().sz = in_sz;
    conv_ios->front().chans = in_chans;
    for( uint32_t i = 0; i != convs->size(); ++i ) {
      conv_ios->at(i+1).sz = convs->at(i).in_sz_to_out_sz( conv_ios->at(i).sz, ignore_padding );
      if( conv_ios->at(i+1).sz.both_dims_non_zero() ) { 
	conv_ios->at(i).used_sz = convs->at(i).out_sz_to_in_sz( conv_ios->at(i+1).sz, ignore_padding );
      } // else if there's no output, we used no input (used_sz left at zero)
      // reset or propogate num_chans
      uint32_t const & out_chans = convs->at(i).out_chans; 
      conv_ios->at(i+1).chans = out_chans ? out_chans : conv_ios->at(i).chans;
    }
    return conv_ios;
  }
  void conv_pipe_t::dump_pipe( std::ostream & out ) const {
    out << strprintf( "== BEGIN CONV PIPE ==\n" );
    for( uint32_t i = 0; ; ++i ) {
      conv_support_info_t const & csi = conv_sis.at(i);
      out << strprintf( "support_sz=%s support_stride=%s eff_tot_pad=%s\n", 
			str(csi.support_sz).c_str(), 
			str(csi.support_stride).c_str(), str(csi.eff_tot_pad).c_str() );
      if( i == convs->size() ) { break; }
      out << strprintf( "    ----  conv=%s \n", str(convs->at(i)).c_str() );
    }
    out << strprintf( "== END CONV PIPE ==\n" );
  }
  void conv_pipe_t::dump_ios( std::ostream & out, p_vect_conv_io_t const & conv_ios ) const {
    out << "CONV_IOS: ";
    for( uint32_t i = 0; ; ++i ) {
      conv_io_t const & cio = conv_ios->at(i);
      out << strprintf( "sz=%s -> ", str(cio.sz).c_str() );
      string size_err;
      if( cio.sz != cio.used_sz ) { 
	if( (cio.used_sz.d[0] > cio.sz.d[0]) || (cio.used_sz.d[1] > cio.sz.d[1]) ) { size_err += "IMPLICIT PAD; "; }
	if( (cio.used_sz.d[0] < cio.sz.d[0]) || (cio.used_sz.d[1] < cio.sz.d[1]) ) { size_err += "DATA DISCARDED; "; }
	out << strprintf( "[%sused_sz=%s] -> ", size_err.c_str(), str(cio.used_sz).c_str() );
      }
      if( i == convs->size() ) { break; }
      out << convs->at(i).tag << " -> ";
    }
    out << "\n";
  }

  void conv_pipe_t::dump_ops( std::ostream & out, p_vect_conv_io_t const & conv_ios ) const {
    assert_st( conv_ios && (conv_ios->size() == convs->size()+1) );
    out << strprintf( "== BEGIN OPS ==\n" );
    
    for( uint32_t i = 0; i != convs->size(); ++i ) {
      conv_op_t const & conv_op = convs->at(i);
      conv_io_t const & cio_in = conv_ios->at(i);
      conv_io_t const & cio_out = conv_ios->at(i+1);
      printf( "cio_in.sz=%s conv_op.tag=%s cio_out.sz=%s\n", str(cio_in.sz).c_str(), str(conv_op.tag).c_str(), str(cio_out.sz).c_str() );    }
    out << strprintf( "== END OPS ==\n" );
  }

  struct conv_ana_t : virtual public nesi, public conv_pipe_t, public has_main_t // NESI(help="analysize pipeline of convolutions wrt sizes at each layer, strides, padding, and per-layer-input-sizes (aka support sizes). ",bases=["conv_pipe_t","has_main_t"], type_id="conv_ana")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="text output filename")
    // filename_t convs_fn; NESI(help="input: filename for list of convs",req=1)
    p_uint32_t in_sz; //NESI(help="calculate sizes at all layers for the given input size and dump pipe")
    uint32_t in_chans; //NESI(default=3,help="number of input chans (used only to properly print number of input chans)")
    p_uint32_t out_sz; //NESI(help="calculate sizes at all layers for the given output size and dump pipe")
    uint32_t ignore_padding_for_sz; //NESI(default=0,help="if 1, ignore any padding specified when calculating the sizes at each layer for the in_sz or out_sz options")
    uint32_t print_ops; //NESI(default=0,help="if non-zero, print ops. note: requires in_sz to be set.")
    
    virtual void main( nesi_init_arg_t * nia ) { 
      p_ofstream out = ofs_open( out_fn.exp );
      //(*out) << convs << "\n";
      calc_support_info();
      dump_pipe( *out ); 
      if( out_sz ) { 
	(*out) << ">> calculating network sizes backward given an out_sz of " << *out_sz << "\n";
	p_vect_conv_io_t conv_ios = calc_sizes_back( u32_pt_t( *out_sz, *out_sz ), ignore_padding_for_sz ); 
	dump_ios( *out, conv_ios ); 
      }
      p_vect_conv_io_t conv_ios;
      if( in_sz ) { 
	(*out) << ">> calculating network sizes forward given an in_sz of " << *in_sz << "\n";
	conv_ios = calc_sizes_forward( u32_pt_t( *in_sz, *in_sz ), in_chans, ignore_padding_for_sz ); 
	dump_ios( *out, conv_ios ); 
      }
      if( print_ops ) {
	if( !conv_ios ) { rt_err( "print_ops requires in_sz to be set in order to calculute the conv_ios." ); }
	dump_ops( *out, conv_ios );
      }
    }
  };

#include"gen/conv_util.H.nesi_gen.cc"
#include"gen/conv_util.cc.nesi_gen.cc"

};
