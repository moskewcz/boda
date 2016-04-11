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

  string dims_yxc_str( dims_t const & d ) { return strprintf( "$%s \\dx %s \\dx %s$", str(d.dsz("y")).c_str(), 
							      str(d.dsz("x")).c_str(), str(d.dsz("chan")).c_str() ); }
  string mkn_str( uint64_t const & M, uint64_t const & K, uint64_t const & N  ) { 
    return strprintf( "$%s \\dx %s \\dx %s$", str(M).c_str(), str(K).c_str(), str(N).c_str() ); }

  string conv_op_info_as_latex_tab_row( p_conv_op_base_t const & op ) {
    string ret;
    assert_st( op->kern_sz().dims_are_same() );
    assert_st( op->stride().dims_are_same() );
    dims_t const & dout = op->get_dims("out");
    ret += strprintf( "%s & %s & %s", str(op->kern_sz().d[0]).c_str(), str(op->stride().d[0]).c_str(), 
		      str(dout.dsz("chan")).c_str() );
    dims_t const & din = op->get_dims("in");
    uint64_t const B = din.dsz( "img" );
    assert_st( B == dout.dsz("img" ) );
    ret += strprintf( " & %s & %s & %s", str(B).c_str(), dims_yxc_str(din).c_str(), dims_yxc_str(dout).c_str() );

    // AI-related calculations
    dims_t const & filts = op->get_dims("filts");
    dims_t const & biases = op->get_dims("biases");

    uint64_t const M = dout.dsz("img")*dout.dsz("x")*dout.dsz("y"); // note: all-imgs M
    uint64_t const K = filts.dsz("in_chan")*filts.dsz("x")*filts.dsz("y");
    uint64_t const N = filts.dsz("out_chan");

    uint64_t const forward_bytes = (din.dims_prod() + dout.dims_prod() + filts.dims_prod() + biases.dims_prod()) * 4;

    uint64_t const forward_flops = M * N * K * 2;
    
    double const ai = double(forward_flops)/double(forward_bytes);
    ret += strprintf( " & %s & %s & %s & %s", pp_bytes(forward_bytes).c_str(), pp_flops(forward_flops).c_str(), 
		      pp_val(ai).c_str(), mkn_str(M,K,N).c_str() );

    return ret + " \\\\";
  }

  struct cnn_op_info_t : virtual public nesi, public has_main_t // NESI(help="print info for set of CNN operations",
			 // bases=["has_main_t"], type_id="cnn_op_info" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t cnn_func_sigs_fn; //NESI(default="%(boda_test_dir)/cnn_func_sigs_tiny.txt",help="file to read cnn ops from")
    virtual void main( nesi_init_arg_t * nia );
  };

  void cnn_op_info_t::main( nesi_init_arg_t * nia ) {
    vect_p_conv_op_t sigs;
    p_vect_string in_lines = readlines_fn( cnn_func_sigs_fn );

    for( vect_string::const_iterator i = in_lines->begin(); i != in_lines->end(); ++i ) {
      p_conv_op_base_t op = make_p_conv_op_base_t_init_and_check_unused_from_lexp( parse_lexp( *i ), 0 );
      op->set_and_check_coi();
      string const s = conv_op_info_as_latex_tab_row( op );
      printstr( s + "\n" );
    }
  }

  struct cnn_prof_t : virtual public nesi, public has_main_t // NESI(help="profile set of rtc functions",
		      // bases=["has_main_t"], type_id="cnn_prof" )

  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

    filename_t cnn_func_sigs_fn; //NESI(default="%(boda_test_dir)/cnn_func_sigs_tiny.txt",help="file to read cnn ops from")
    filename_t rtc_func_sigs_fn; //NESI(default="%(boda_output_dir)/cnn_rtc_func_sigs.txt",help="file to hold all generated rtc func signatures for the input cnn ops")
    virtual void main( nesi_init_arg_t * nia );

  };

  void add_cnn_codegen_annotations( conv_op_base_t * const op, 
				    bool const & enable_ipconv, bool const & enable_k1conv, bool const & enable_tconv, 
				    bool const & force_enable_tconv, uint32_t const t_tile_sz );

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
