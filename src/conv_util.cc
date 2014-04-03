// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"timers.H"
#include"str_util.H"
#include"has_main.H"
#include"io_util.H"
#include"nesi.H"

namespace boda 
{
  using namespace boost;

  struct conv_op_t : virtual public nesi // NESI(help="conv_op descriptor") 
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    string tag; //NESI(help="tag to refer to conv op by",req=1)
    
    u32_box_t in_pad; //NESI(default="0 0 0 0",help="input padding")
    u32_pt_t kern_sz; //NESI(default="0 0",help="convolutional kernel size")
    u32_pt_t stride; //NESI(default="1 1",help="step/stride in input")

    // related to depth (optional?)
    uint32_t channels; //NESI(default="1",help="number of channels")
    uint32_t groups; //NESI(default="1",help="number of groups (equal partitions of inputs and outputs)")

    u32_pt_t out_sz_to_in_sz( u32_pt_t const & out_sz, bool const ignore_padding ) const { 
      return kern_sz + (out_sz-u32_pt_t(1,1))*stride - (ignore_padding?u32_pt_t():in_pad.bnds_sum());
    }
    u32_pt_t in_sz_to_out_sz( u32_pt_t const & in_sz, bool const ignore_padding ) const { 
      return (in_sz+(ignore_padding?u32_pt_t():in_pad.bnds_sum())-kern_sz)/stride + u32_pt_t(1,1); 
    }
  };

  typedef vector< conv_op_t > vect_conv_op_t; 
  typedef shared_ptr< conv_op_t > p_conv_op_t; 
  typedef vector< p_conv_op_t > vect_p_conv_op_t;
  typedef shared_ptr< vect_conv_op_t > p_vect_conv_op_t; 

  // struct metadata about inputs/outputs of conv ops
  struct conv_io_t {
    u32_pt_t sz;
    u32_pt_t used_sz;

    u32_pt_t support_sz;
    u32_pt_t support_stride;
    u32_box_t support_pad;
  };
  
  typedef vector< conv_io_t > vect_conv_io_t; 
  typedef shared_ptr< conv_io_t > p_conv_io_t; 
  typedef vector< p_conv_io_t > vect_p_conv_io_t;

  struct conv_pipe_t {
    p_vect_conv_op_t convs;
    vect_conv_io_t conv_ios; // note: imgs.size() == ( convs.size() + 1 )
    conv_pipe_t( p_vect_conv_op_t const & convs_, bool const ignore_padding_for_support ) : convs(convs_) {
      conv_ios.clear();
      conv_ios.resize( convs->size() + 1 );
      calc_support_sz( ignore_padding_for_support );
    }     

    void zero_conv_ios( void ) {
      for( vect_conv_io_t::iterator i = conv_ios.begin(); i != conv_ios.end(); ++i ) {
	i->sz = u32_pt_t(); i->used_sz = u32_pt_t();
      }
    }

    void calc_support_sz( bool const ignore_padding ) {
      conv_ios.front().support_sz = u32_pt_t(1,1);
      conv_ios.front().support_stride = u32_pt_t(1,1);
      for( uint32_t i = 0; i != convs->size(); ++i ) {
	conv_op_t const & cop = convs->at(i);
	assert_st( cop.kern_sz.both_dims_non_zero() );
	u32_pt_t const in_sz_1x1 = cop.out_sz_to_in_sz( u32_pt_t(1,1), ignore_padding );
	assert_st( in_sz_1x1.both_dims_non_zero() );
	conv_ios[i+1].support_sz = conv_ios[i].support_sz + ( in_sz_1x1 - u32_pt_t(1,1) )*conv_ios[i].support_stride;
	conv_ios[i+1].support_stride = conv_ios[i].support_stride*cop.stride;
      }
      // backward pass to calculate support_pad
      for( uint32_t i = convs->size(); i; --i ) {
	conv_op_t const & cop = convs->at(i-1);
	conv_ios[i-1].support_pad = cop.in_pad + conv_ios[i].support_pad.scale_dims( cop.stride );	
      }
    }


    void calc_sizes_back( u32_pt_t const & out_sz, bool const ignore_padding ) {
      zero_conv_ios();
      conv_ios.back().sz = out_sz;
      for( uint32_t i = convs->size(); i; --i ) {
	conv_ios[i-1].sz = convs->at(i-1).out_sz_to_in_sz( conv_ios[i].sz, ignore_padding );
	conv_ios[i-1].used_sz = conv_ios[i-1].sz; // by semantics of out_sz_to_in_sz (but checked below)
	assert_st( conv_ios[i].sz == convs->at(i-1).in_sz_to_out_sz( conv_ios[i-1].sz, ignore_padding ) );
      }
    }
    void calc_sizes_forward( u32_pt_t const & in_sz, bool const ignore_padding ) {
      zero_conv_ios(); 
      conv_ios.front().sz = in_sz;
      for( uint32_t i = 0; i != convs->size(); ++i ) {
	conv_ios[i+1].sz = convs->at(i).in_sz_to_out_sz( conv_ios[i].sz, ignore_padding );
	conv_ios[i].used_sz = convs->at(i).out_sz_to_in_sz( conv_ios[i+1].sz, ignore_padding );
      }
    }
    void dump_pipe( std::ostream & out ) {
      out << strprintf( "== BEGIN CONV PIPE ==\n" );
      for( uint32_t i = 0; ; ++i ) {
	conv_io_t const & cio = conv_ios[i];
	out << strprintf( "cio: sz=%s support_sz=%s support_stride=%s support_pad=%s\n", 
			  str(cio.sz).c_str(), str(cio.support_sz).c_str(), 
			  str(cio.support_stride).c_str(), str(cio.support_pad).c_str() );
	if( conv_ios[i].sz != conv_ios[i].used_sz ) {
	  out << "  --- DATA DISCARDED --- " << strprintf( "used_sz=%s\n", str(conv_ios[i].used_sz).c_str() );
	}
	if( i == convs->size() ) { break; }
	out << strprintf( "    ----  conv=%s \n", str(convs->at(i)).c_str() );
      }
      out << strprintf( "== END CONV PIPE ==\n" );
    }
  };

  struct conv_ana_t : virtual public nesi, public has_main_t // NESI(help="blf rectangle packing",bases=["has_main_t"], type_id="conv_ana")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t out_fn; //NESI(default="%(boda_output_dir)/out.txt",help="text output filename")
    p_vect_conv_op_t convs; //NESI(help="set of convs",req=1)
    // filename_t convs_fn; NESI(help="input: filename for list of convs",req=1)
    p_uint32_t in_sz; //NESI(help="calculate sizes at all layers for the given input size and dump pipe")
    p_uint32_t out_sz; //NESI(help="calculate sizes at all layers for the given output size and dump pipe")
    uint32_t ignore_padding_for_support; //NESI(default=0,help="if 1, ignore any padding specified when calculating the support_size for a single pel for each layer")
    uint32_t ignore_padding_for_sz; //NESI(default=0,help="if 1, ignore any padding specified when calculating the sizes at each layer for the in_sz or out_sz options")
    virtual void main( nesi_init_arg_t * nia ) { 
      p_ofstream out = ofs_open( out_fn.exp );
      //(*out) << convs << "\n";
      conv_pipe_t cp{convs,ignore_padding_for_support};
      if( out_sz ) { cp.calc_sizes_back( u32_pt_t( *out_sz, *out_sz ), ignore_padding_for_sz ); cp.dump_pipe(*out); }
      if( in_sz ) { cp.calc_sizes_forward( u32_pt_t( *in_sz, *in_sz ), ignore_padding_for_sz ); cp.dump_pipe(*out); }
    }
  };

#include"gen/conv_util.cc.nesi_gen.cc"

};
