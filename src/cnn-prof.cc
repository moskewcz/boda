// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"timers.H"
#include<boost/filesystem.hpp>
#include<boost/lexical_cast.hpp>
#include"lexp.H"
#include"has_main.H"
#include"rtc_func_gen.H"
#include"rtc_compute.H"
#include"conv_util.H"

namespace boda 
{

  // FIXME: mostly dup'd with similar code in rtc_func_gen.cc for generated function signatures
  typedef shared_ptr< conv_op_base_t > p_conv_op_base_t; 
  p_conv_op_base_t make_p_conv_op_base_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );

  string dims_yxc_str( dims_t const & d, bool include_img = 0 ) { 
    return strprintf( "$%s%s \\dx %s \\dx %s$", include_img ? (str(d.dsz("img"))+"\\dx ").c_str():"",
		      str(d.dsz("y")).c_str(), str(d.dsz("x")).c_str(), str(d.dsz("chan")).c_str() ); }
  string mkn_str( uint64_t const & M, uint64_t const & K, uint64_t const & N  ) { 
    return strprintf( "$%s \\dx %s \\dx %s$", str(M).c_str(), str(K).c_str(), str(N).c_str() ); }


  struct conv_op_info_to_latex_t {
    p_conv_op_base_t op;    
    dims_t din;
    dims_t dout;
    uint64_t B;
    uint64_t M,N,K;
    uint64_t forward_bytes, forward_flops;

    void base_info( std::ostream * const out ) {
      assert_st( op->kern_sz().dims_are_same() );
      assert_st( op->stride().dims_are_same() );
      (*out) << strprintf( "%s & %s & %s", str(op->kern_sz().d[0]).c_str(), str(op->stride().d[0]).c_str(), str(dout.dsz("chan")).c_str() );
    }
    void info_row( std::ostream * const out ) {
      base_info( out );
      (*out) << strprintf( " & %s & %s & %s", str(B).c_str(), dims_yxc_str(din).c_str(), dims_yxc_str(dout).c_str() );
      double const ai = double(forward_flops)/double(forward_bytes);
      (*out) << strprintf( " & %s & %s & %s & %s", pp_bytes(forward_bytes).c_str(), pp_flops(forward_flops).c_str(), 
			   pp_val(ai).c_str(), mkn_str(M,K,N).c_str() );
      (*out) << "\\\\ \n";
    }
    void eff_row( std::ostream * const out, string const & rtc_op_type, double const & runtime_secs, double const & peak_flops ) {
      base_info( out );
      (*out) << strprintf( " & %s & %s ", dims_yxc_str(din,1).c_str(), rtc_op_type.c_str() );
      double const fps = double(forward_flops)/runtime_secs;
      (*out) << strprintf( " & %s & %s & %s", pp_secs(runtime_secs).c_str(), pp_fps(fps).c_str(), pp_val(fps/peak_flops*100.0).c_str() ); 
      (*out) << "\\\\ \n";
    }
    void init( p_conv_op_base_t const & op_ ) {
      op = op_;
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
    }
  };

  struct cnn_op_info_t : virtual public nesi, public has_main_t // NESI(help="print info for set of CNN operations",
			 // bases=["has_main_t"], type_id="cnn_op_info" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t cnn_func_sigs_fn; //NESI(default="%(boda_test_dir)/cnn_func_sigs_tiny.txt",help="file to read cnn ops from")
    filename_t op_info_tab_fn; //NESI(default="%(boda_output_dir)/op-info-tab.tex",help="file to write op info latex rows to")
    filename_t op_eff_tab_fn; //NESI(default="%(boda_output_dir)/op-eff-tab.tex",help="file to write op info latex rows to")

    double peak_flops; //NESI(default=6600e9,help="peak flops of platform (for computing peak %s)")

    uint32_t run_opt_variants; //NESI(default=1,help="if 1, run opt variants")

    uint32_t show_rtc_calls; //NESI(default=1,help="if 1, print rtc calls")
    p_rtc_compute_t rtc; //NESI(default="(be=ocl)",help="rtc back-end to use")
    rtc_codegen_t codegen;

    virtual void main( nesi_init_arg_t * nia );
  };

  void add_cnn_codegen_annotations( conv_op_base_t * const op, 
				    bool const & enable_ipconv, bool const & enable_k1conv, bool const & enable_tconv, 
				    bool const & force_enable_tconv, uint32_t const t_tile_sz );

  double profile_rcg_call( p_rtc_compute_t const & rtc, rtc_codegen_t & codegen, bool const & show_rtc_calls,
			   string const & func_name, p_rtc_call_gen_t const & rcg );

  void cnn_op_info_t::main( nesi_init_arg_t * nia ) {
    vect_p_conv_op_t sigs;
    p_vect_string in_lines = readlines_fn( cnn_func_sigs_fn );
    p_ofstream oit_out = ofs_open( op_info_tab_fn );
    p_ofstream oet_out = ofs_open( op_eff_tab_fn );

    rtc->init();
    bool const enable_prof = 0;
    if( enable_prof ) { rtc->profile_start(); }

    for( vect_string::const_iterator i = in_lines->begin(); i != in_lines->end(); ++i ) {
      p_conv_op_base_t op = make_p_conv_op_base_t_init_and_check_unused_from_lexp( parse_lexp( *i ), 0 );
      op->set_and_check_coi();
      conv_op_info_to_latex_t to_latex;
      to_latex.init( op );
      to_latex.info_row( oit_out.get() );
      for( uint32_t opt = 0; opt <= run_opt_variants; ++opt ) {
	// create rtc op
	p_conv_op_base_t anno_op = make_shared<conv_op_base_t>( *op );
	add_cnn_codegen_annotations( anno_op.get(), 0, opt, opt, opt, 4 );
	p_rtc_func_sig_t rtc_op = make_shared< rtc_func_sig_t >( must_find( anno_op->str_vals, "cts" ), 
								 anno_op->dims_vals, anno_op->str_vals );
	must_insert( rtc_op->str_vals, "conv_has_relu", str(1) );
      
	// profile rtc op
	codegen.gen_func( make_cnn_custom_codegen_t().get(), *rtc_op );
	assert( codegen.rtc_func_names_map.size() == 1 );
	p_rtc_call_gen_t const &rcg = codegen.rtc_func_names_map.begin()->second;
	string const & func_name = codegen.rtc_func_names_map.begin()->first;
	if( !rcg->blks ) { 
	  printf( "skipping %s; dynamic block sizes todo\n", str(rcg->type).c_str() );
	  continue; 
	}
	if( (rcg->type == "quantize") || (rcg->type == "dropout") ) {
	  printf( "skipping %s; u32 arg handling todo\n", str(rcg->type).c_str() );
	  continue; 
	}
	double const rfc_dur_secs = profile_rcg_call( rtc, codegen, show_rtc_calls, func_name, rcg ) / 1000.0;
	printf( "rfc_dur_secs=%s\n", str(rfc_dur_secs).c_str() );
	codegen.rtc_func_names_map.clear();
	codegen.rtc_prog_str.clear();
	to_latex.eff_row( oet_out.get(), rtc_op->type, rfc_dur_secs, peak_flops );
      }
    }

    if( enable_prof ) { rtc->profile_stop(); }
    rtc->finish_and_sync();
  }

  struct cnn_prof_t : virtual public nesi, public has_main_t // NESI(help="profile set of rtc functions",
		      // bases=["has_main_t"], type_id="cnn_prof" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    filename_t cnn_func_sigs_fn; //NESI(default="%(boda_test_dir)/cnn_func_sigs_tiny.txt",help="file to read cnn ops from")
    filename_t rtc_func_sigs_fn; //NESI(default="%(boda_output_dir)/cnn_rtc_func_sigs.txt",help="file to hold all generated rtc func signatures for the input cnn ops")
    virtual void main( nesi_init_arg_t * nia );

  };


  void cnn_prof_t::main( nesi_init_arg_t * nia ) {
    vect_p_conv_op_t sigs;
    p_ofstream out = ofs_open( rtc_func_sigs_fn );
    p_vect_string in_lines = readlines_fn( cnn_func_sigs_fn );

    for( vect_string::const_iterator i = in_lines->begin(); i != in_lines->end(); ++i ) {
      p_conv_op_base_t op = make_p_conv_op_base_t_init_and_check_unused_from_lexp( parse_lexp( *i ), 0 );
      op->set_and_check_coi();
      add_cnn_codegen_annotations( op.get(), 0, 0, 0, 0, 4 );
      op->type = must_find( op->str_vals, "cts" );
      must_insert( op->str_vals, "conv_has_relu", str(1) );
      (*out) << str( *op ) << "\n";
    }
  }

#include"gen/cnn-prof.cc.nesi_gen.cc"

}
