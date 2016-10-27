// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"op-tuner.H"
#include"str_util.H"
#include"nesi.H"
#include"lexp.H"
#include"has_main.H"
#include<sstream>
#include<boost/regex.hpp>

namespace boda 
{

  using boost::regex;
  using boost::regex_search;

  bool read_string_or_EOF( p_istream const & in, string const & s ) {
    string line;
    if( ifs_getline( "", in, line ) ) { return true; } // EOF
    if( line != s ) { rt_err( strprintf( "error reading line-oriented text stream. expected a line with '%s', but saw '%s'.", 
                                         str(s).c_str(), str(line).c_str() ) ); }
    return false;
  }
  void must_read_string( p_istream const & in, string const & s ) {
    bool const at_eof = read_string_or_EOF( in, s );
    if( at_eof ) { rt_err( strprintf( "error reading line-oriented text stream. expected a line with '%s', but got EOF.", 
                                      str(s).c_str() ) ); }
  }
  string must_getline( p_istream const & in ) {
    string ret;
    bool const at_eof = ifs_getline( "", in, ret );
    if( at_eof ) { rt_err( "error reading line-oriented text stream. expected a non-empty line, but got EOF." ); }
    return ret;
  }

  template< typename T > inline void bread_from_str( string const & in, T & v ) {
    std::istringstream iss( in );
    bread( iss, v );
  }

  void read_op_run_t( op_run_t & v, p_istream const & in ) {
    v.be_plat_tag = must_getline( in );
    v.rt_secs = lc_str_d( must_getline( in ) );
    v.err = must_getline( in );
    if( v.err.empty() ) { v.op = make_p_op_base_t_init_and_check_unused_from_lexp( parse_lexp( must_getline( in ) ), 0 ); }
  }

  void read_op_tune_wisdom( op_tune_wisdom_t & v, p_istream const & in ) {
    v.op_tune = make_p_op_tune_t_init_and_check_unused_from_lexp( parse_lexp( must_getline( in ) ), 0 );     
    string line;
    while( 1 ) {
      line = must_getline( in );
      if( 0 ) {
      } else if( line == "/op_tune_wisdom_t" ) { 
        return;
      } else if( line == "op_run_t" ) {
        op_run_t vv;
        read_op_run_t( vv, in );
        must_insert( v.runs, vv.be_plat_tag, vv );
      } else {
        rt_err( strprintf( "unknown op_tune_wisdom_t text format stream command read '%s'", line.c_str() ) );
      }
    }
  }

  // returns null if no more wisdoms in stream
  p_op_wisdom_t read_next_wisdom( p_istream const & in ) {
    p_op_wisdom_t ret;
    if( read_string_or_EOF( in, "op_wisdom_t" ) ) { return ret; } // EOF
    ret = make_shared<op_wisdom_t>();
    ret->op = make_p_op_base_t_init_and_check_unused_from_lexp( parse_lexp( must_getline( in ) ), 0 ); 
    string line;
    while( 1 ) {
      line = must_getline( in );
      if( 0 ) {
      } else if( line == "/op_wisdom_t" ) { 
        return ret;
      } else if( line == "kg" ) { 
        pair_str_p_nda_digest_t kg;
        kg.first = must_getline( in );
        bread_from_str( unhex(must_getline( in )), kg.second );
        ret->kgs.push_back( kg );
      } else if( line == "op_tune_wisdom_t" ) { 
        p_op_tune_wisdom_t v = make_shared< op_tune_wisdom_t >();
        read_op_tune_wisdom( *v, in );
        ret->wisdoms.push_back( v );
      } else { 
        rt_err( strprintf( "unknown op_wisdom_t text format stream command read '%s'", line.c_str() ) );
      }
    }
    return ret; // may be null if at eof
  }

  void write_op_run( op_run_t const & v, std::ostream & out ) {
    out << "op_run_t\n";
    out << v.be_plat_tag << "\n";
    out << v.rt_secs << "\n";
    out << v.err << "\n";
    if( v.err.empty() ) { assert_st( v.op ); out << str(v.op) << "\n"; }
  }

  void write_op_tune_wisdom( op_tune_wisdom_t const & v, std::ostream & out ) {
    out << "op_tune_wisdom_t\n";
    out << str(v.op_tune) << "\n";
    for( map_str_op_run_t::const_iterator i = v.runs.begin(); i != v.runs.end(); ++i ) { write_op_run( i->second, out ); }
    out << "/op_tune_wisdom_t\n";
  }

  template< typename T > inline string bwrite_to_str( T const & v ) {
    std::ostringstream oss;
    bwrite( oss, v );
    return oss.str();
  }

  void write_op_wisdom( op_wisdom_t const & v, std::ostream & out ) {
    out << "op_wisdom_t\n";
    out << str(v.op) << "\n";
    for( vect_pair_str_p_nda_digest_t::const_iterator i = v.kgs.begin(); i != v.kgs.end(); ++i ) {
      out << "kg\n" << i->first << "\n" << hex(bwrite_to_str(i->second)) << "\n"; 
    }
    for( vect_p_op_tune_wisdom_t::const_iterator i = v.wisdoms.begin(); i != v.wisdoms.end(); ++i ) {
      write_op_tune_wisdom( **i, out );
    }
    out << "/op_wisdom_t\n";
  }


  struct wis_ana_t : virtual public nesi, public has_main_t // NESI(help="analyses wisdom file, output data in format for plotting",
           // bases=["has_main_t"], type_id="wis-ana" )
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    filename_t wisdom_in_fn; //NESI(help="wisdom input file (to add to, may contain known-good results for checking)",req=1)
    p_filename_t csv_out_fn; //NESI(help="csv output filename")

    uint32_t s_img; //NESI(default="0",help="0 == all # of imgs; otherwise, only ops with the specified #")
    string s_plat; //NESI(default=".*",help="regex to select targ plat tag")

    map_str_uint32_t op_tunes;

    uint32_t s_tix; //NESI(default=0,help="0 == all tixs, othewise, only specified tix")
    uint32_t best_tix; //NESI(default=0,help="0 == all tixs, othewise, only lowest-time tix")

    uint32_t get_tix( op_tune_t const & ot ) { return op_tunes.insert( make_pair( str(ot), op_tunes.size()+1 ) ).first->second; }

    // FIXME: cut-n-paste from cnn-prof ...
    uint64_t get_op_flops( p_op_base_t const & op ) {

      // assert( op->is( Convolution_coi ) ); // FIXME

      dims_t din;
      dims_t dout;
      uint64_t B;
      uint64_t M,N,K;
      //uint64_t forward_bytes;
      uint64_t forward_flops;

      dout = op->get_dims("out");
      din = op->get_dims("in");
      B = din.dsz( "img" );
      assert_st( B == dout.dsz("img" ) );
      // AI-related calculations
      dims_t const & filts = op->get_dims("filts");
      //dims_t const & biases = op->get_dims("biases");
      M = dout.dsz("img")*dout.dsz("x")*dout.dsz("y"); // note: all-imgs M
      K = filts.dsz("in_chan")*filts.dsz("x")*filts.dsz("y");
      N = filts.dsz("out_chan");
      //forward_bytes = (din.dims_prod() + dout.dims_prod() + filts.dims_prod() + biases.dims_prod()) * 4;
      forward_flops = M * N * K * 2;

      return forward_flops;
    }

    virtual void main( nesi_init_arg_t * nia ) {
      p_istream win = ifs_open( wisdom_in_fn );
      p_ostream csv_out = csv_out_fn ? ofs_open( *csv_out_fn ) : p_ostream();

      regex r_plat( s_plat );
      double tot_time = 0;
      uint64_t tot_runs = 0;
      for( p_op_wisdom_t owi; owi = read_next_wisdom( win ); ) {
        if( s_img && (owi->op->get_dims("in").dsz("img") != s_img) ) { continue; }
        printf( "owi->op=%s\n", str(owi->op).c_str() );
        double min_time = std::numeric_limits<double>::max();
        op_run_t const * min_r = 0;
        uint32_t min_tix = 0;
        for( vect_p_op_tune_wisdom_t::const_iterator otwi = owi->wisdoms.begin(); otwi != owi->wisdoms.end(); ++otwi ) {
          for( map_str_op_run_t::const_iterator ri = (*otwi)->runs.begin(); ri != (*otwi)->runs.end(); ++ri ) {
            op_run_t const & r = ri->second;
            if( !r.err.empty() ) { continue; }
            if( !regex_search( r.be_plat_tag, r_plat ) ) { continue; }
            uint32_t const tix = get_tix(*(*otwi)->op_tune);
            if( s_tix && (tix != s_tix) ) { continue; }
            min_eq( min_time, r.rt_secs );
            if( best_tix ) {
              if( r.rt_secs == min_time ) { min_r = &r; min_tix = tix;}
            } else {
              printf( "r.be_plat_tag=%s r.rt_secs=%s tix=%s\n", str(r.be_plat_tag).c_str(), str(r.rt_secs).c_str(), 
                      str(tix).c_str() );
              tot_time += r.rt_secs;
              ++tot_runs;
            }
          }
        }
        if( best_tix && min_r ) {
          op_run_t const & r = *min_r;
          printf( "r.be_plat_tag=%s r.rt_secs=%s min_tix=%s\n", str(r.be_plat_tag).c_str(), str(r.rt_secs).c_str(), 
                  str(min_tix).c_str() );
          tot_time += r.rt_secs;
          ++tot_runs;
          if( csv_out ) {
            (*csv_out) << strprintf( "%s %s %s %s\n", str(min_tix).c_str(), str(r.rt_secs).c_str(), str(get_op_flops(owi->op)).c_str(),
                                     str(owi->op).c_str() );
          }
        }
      }
      printf( "\n----- tot_time=%s tot_runs=%s ------\n", str(tot_time).c_str(), str(tot_runs).c_str() );
      printstr( "\n-- LEGEND --\n" );
      for( map_str_uint32_t::const_iterator i = op_tunes.begin(); i != op_tunes.end(); ++i ) {
        printf( "tix=%s op_tune=%s\n", str(i->second).c_str(), str(i->first).c_str() );
      }

    }
  };

#include"gen/op-tuner.cc.nesi_gen.cc"

}
