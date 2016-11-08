#ifndef _LATEX_UTIL_H_
#define _LATEX_UTIL_H_

#include"boda_base.H"
#include"geom_prim.H"
#include"str_util.H"
#include"cnn_op.H"

namespace boda 
{


  inline string dims_yxc_str( dims_t const & d, bool include_img = 0 ) { 
    return strprintf( "$ %s %s \\dx %s \\dx %s $", include_img ? (str(d.dsz("img"))+" \\dx").c_str():"",
		      str(d.dsz("y")).c_str(), str(d.dsz("x")).c_str(), str(d.dsz("chan")).c_str() ); }
  inline string mkn_str( uint64_t const & M, uint64_t const & K, uint64_t const & N  ) { 
    if( M == K && K == N ) { return strprintf( "$ %s $", str(M).c_str() ); }
    else { return strprintf( "$ %s \\dx %s \\dx %s $", str(M).c_str(), str(K).c_str(), str(N).c_str() ); }
  }


  struct conv_op_info_to_latex_t {
    p_conv_op_base_t op;    
    dims_t din;
    dims_t dout;
    uint64_t B;
    uint64_t M,N,K;
    uint64_t forward_bytes, forward_flops;
    bool emit_bw;
    uint32_t inc_op_info_in_eff;
    bool show_bytes_and_ai;
    // locally override the global pp_foo() function with member functions that can control formatting
    uint32_t print_format;
#define PP_FMT( t ) string pp_##t( double const v ) const { return (print_format == 0) ? boda::pp_##t( v ) : str(v); }
    PP_FMT( bytes ) PP_FMT( flops ) PP_FMT( val ) PP_FMT( secs ) PP_FMT( fps ) PP_FMT( bps )
#undef PP_FMT

    void base_info( std::ostream * const out ) {
      if( op->is( Convolution_coi ) ) {
	assert_st( op->kern_sz().dims_are_same() );
	assert_st( op->stride().dims_are_same() );
	(*out) << strprintf( "%s & %s & %s", str(op->kern_sz().d[0]).c_str(), str(op->stride().d[0]).c_str(), str(dout.dsz("chan")).c_str() );
      }
    }
    // MKN & Bytes & FLOPs & F/B
    void ai_mkn_row( std::ostream * const out ) {
      if( show_bytes_and_ai ) {
        double const ai = double(forward_flops)/double(forward_bytes);
        (*out) << strprintf( " %s & %s & %s & %s ", mkn_str(M,K,N).c_str(), 
                             pp_bytes(forward_bytes).c_str(), pp_flops(forward_flops).c_str(), pp_val(ai).c_str() );
      } else { // flops only
        (*out) << strprintf( " %s ", pp_flops(forward_flops).c_str() );
      }
    }
    void info_row( std::ostream * const out ) {
      base_info( out );
      if( op->is( Convolution_coi ) ) {
	(*out) << strprintf( " & %s & %s & %s & ", str(B).c_str(), dims_yxc_str(din).c_str(), dims_yxc_str(dout).c_str() );
      }
      ai_mkn_row( out );
      (*out) << "\\\\ " << std::endl;
    }

    // SGEMM comp eff row
    // MKN & Bytes & FLOPs & F/B & Runtime(comp) & GF/s(comp) & Runtime(non-comp) & GF/s(non-comp) & Speedup-of-non-comp (comp/non-comp)

    // conv eff row
    // KSZ & Stride & out_chans & $dims(in)$ & variant & MxKxN & Bytes & FLOPs & F/B & Runtime & GF/s & %Peak
    void eff_row( std::ostream * const out, string const & rtc_op_type, 
                  double const & runtime_secs, double const & peak_flops,
                  double const & runtime_secs_comp ) {
      if( op->is( sgemm_coi ) ) {
        ai_mkn_row( out );
        double const fps_comp = double(forward_flops)/runtime_secs_comp;
        (*out) << strprintf( " & %s & %s ", pp_secs(runtime_secs_comp).c_str(), pp_fps(fps_comp).c_str() );
        double const fps = double(forward_flops)/runtime_secs;
        (*out) << strprintf( " & %s & %s ", pp_secs(runtime_secs).c_str(), pp_fps(fps).c_str() ); 
        (*out) << strprintf( " & %.2fx ", double(runtime_secs_comp/runtime_secs) );
      }
      else {
        base_info( out );
        (*out) << strprintf( " & %s & \\verb|%s| & ", dims_yxc_str(din,1).c_str(), rtc_op_type.c_str() );
        if( inc_op_info_in_eff ) { ai_mkn_row( out ); (*out) << " & "; }
        double const fps = double(forward_flops)/runtime_secs;
        (*out) << strprintf( " %s & %s & %s ", pp_secs(runtime_secs).c_str(), pp_fps(fps).c_str(), pp_val(fps/peak_flops*100.0).c_str() ); 

        if( emit_bw ) {
          // HACK: emit human-readable BW #s for now, breaks later flow/latex
          double const peak_bps = 20e9;
          double const bps = double(forward_bytes)/runtime_secs;
          (*out) << strprintf( " -- %s %s --", pp_bps(bps).c_str(), pp_val(bps/peak_bps*100.0).c_str() );
        }
      }
      (*out) << "\\\\ " << std::endl;
    }
    void init( p_conv_op_base_t const & op_, uint32_t const & print_format_, uint32_t const & inc_op_info_in_eff_, 
               bool const & show_bytes_and_ai_ ) {
      print_format = print_format_;
      inc_op_info_in_eff = inc_op_info_in_eff_;
      show_bytes_and_ai = show_bytes_and_ai_;
      op = op_;
      emit_bw = 0;
      if( op->is( Convolution_coi ) ) {
	dout = op->get_dims("out");
	din = op->get_dims("in");
	B = din.dsz( "img" );
	assert_st( B == dout.dsz("img" ) );
	// AI-related calculations
	dims_t const & filts = op->get_dims("filts");
	dims_t const & biases = op->get_dims("biases");
	M = dout.dsz("img")*dout.dsz("x")*dout.dsz("y"); // note: all-imgs M
	K = filts.dsz("in_chan")*filts.dsz("x")*filts.dsz("y");
	N = filts.dsz("out_chan");
	forward_bytes = (din.dims_prod() + dout.dims_prod() + filts.dims_prod() + biases.dims_prod()) * 4;
	forward_flops = M * N * K * 2;
      } else if( op->is( sgemm_coi ) ) {
	dout = op->get_dims("c");
	dims_t a = op->get_dims("a");
	dims_t b = op->get_dims("b");
	B = 1;
	M = a.dsz("M");
	K = a.dsz("K");
	assert_st( b.dsz("K") == K );
	N = b.dsz("N");
	assert_st( dout.dsz("M") == M );
	assert_st( dout.dsz("N") == N );
	forward_flops = M * N * K * 2;
	forward_bytes = (a.dims_prod() + b.dims_prod() + dout.dims_prod()) * 4;
      } else { rt_err( "cnn-op-info: unhandled op: " + op->get_type() ); }
      
    }
  };
}

#endif /* _LATEX_UTIL_H_ */
